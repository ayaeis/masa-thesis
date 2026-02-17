#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


EXPECTED_KPS = 133


@dataclass
class VideoItem:
    split: str
    class_name: str
    class_id: int
    src_dir: Path
    dst_video_id: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare MASA-compatible WLASL-style data from split/class/video JPG folders."
    )
    p.add_argument(
        "--preproc-root",
        required=True,
        help="Root like: .../preprocessing with train/val/test and frames/ underneath each split.",
    )
    p.add_argument(
        "--out-root",
        required=True,
        help="Output MASA dataset root (will contain traintestlist/, jpg_video_ori/, Keypoints_2d_mmpose/).",
    )
    p.add_argument(
        "--subset-num",
        type=int,
        default=100,
        help="Subset tag for traintestlist folder name WLASL{subset_num}.",
    )
    p.add_argument(
        "--det-model",
        default="whole_image",
        help="MMPose detector alias/path. Use 'whole_image' to avoid mmdet dependency (default: whole_image).",
    )
    p.add_argument(
        "--pose-model",
        default="wholebody",
        help="MMPose pose2d alias/path passed to MMPoseInferencer (default: wholebody).",
    )
    p.add_argument(
        "--pose-weights",
        default=None,
        help="Optional pose checkpoint path/URL. Strongly recommended when pose-model is a config path.",
    )
    p.add_argument(
        "--det-weights",
        default=None,
        help="Optional detector checkpoint path/URL.",
    )
    p.add_argument(
        "--copy-frames",
        action="store_true",
        help="Copy frames into jpg_video_ori. Default is symlink (faster, less disk).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing pose pkl/list outputs.",
    )
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stable_class_map(preproc_root: Path) -> Dict[str, int]:
    classes = set()
    for split in ("train", "val", "test"):
        split_frames = preproc_root / split / "frames"
        if not split_frames.exists():
            continue
        for cls_dir in split_frames.iterdir():
            if cls_dir.is_dir():
                classes.add(cls_dir.name)
    return {name: i for i, name in enumerate(sorted(classes))}


def collect_items(preproc_root: Path, class_to_id: Dict[str, int]) -> List[VideoItem]:
    items: List[VideoItem] = []
    for split in ("train", "val", "test"):
        split_frames = preproc_root / split / "frames"
        if not split_frames.exists():
            continue
        for cls_dir in sorted(split_frames.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name
            cls_id = class_to_id[cls_name]
            for video_dir in sorted(cls_dir.iterdir()):
                if not video_dir.is_dir():
                    continue
                # Keep ids unique across classes/splits.
                dst_video_id = f"{cls_name}__{video_dir.name}"
                items.append(
                    VideoItem(
                        split=split,
                        class_name=cls_name,
                        class_id=cls_id,
                        src_dir=video_dir,
                        dst_video_id=dst_video_id,
                    )
                )
    return items


def frame_paths(video_dir: Path) -> List[Path]:
    frames = [p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    return sorted(frames, key=lambda p: p.name)


def materialize_standardized_frames(
    src_dir: Path,
    dst_dir: Path,
    copy_frames: bool,
    overwrite: bool,
) -> List[Path]:
    """Create MASA-compatible frame names (img_00001.jpg, ...).

    The WLASL loader inside this repo hardcodes ``img_00001.jpg`` for reading
    image size, so we always normalize names in the destination tree.
    """
    src_frames = frame_paths(src_dir)
    if not src_frames:
        return []

    if dst_dir.exists():
        if overwrite:
            if dst_dir.is_symlink() or dst_dir.is_file():
                dst_dir.unlink()
            else:
                shutil.rmtree(dst_dir)
        else:
            existing = frame_paths(dst_dir)
            return existing

    ensure_dir(dst_dir)
    dst_frames: List[Path] = []
    for i, src_frame in enumerate(src_frames, start=1):
        dst_name = f"img_{i:05d}.jpg"
        dst_path = dst_dir / dst_name
        if copy_frames:
            shutil.copy2(src_frame, dst_path)
        else:
            rel = os.path.relpath(src_frame, dst_path.parent)
            dst_path.symlink_to(rel)
        dst_frames.append(dst_path)
    return dst_frames


def pick_best_instance(instances: Sequence[Dict]) -> Optional[Dict]:
    best = None
    best_score = -1.0
    for ins in instances:
        kps = np.asarray(ins.get("keypoints", []), dtype=np.float32)
        if kps.ndim != 2 or kps.shape[0] == 0:
            continue
        if "keypoint_scores" in ins:
            score = float(np.mean(np.asarray(ins["keypoint_scores"], dtype=np.float32)))
        elif kps.shape[1] >= 3:
            score = float(np.mean(kps[:, 2]))
        else:
            score = 0.0
        if score > best_score:
            best = ins
            best_score = score
    return best


def inst_to_133x3(inst: Optional[Dict]) -> np.ndarray:
    if inst is None:
        return np.zeros((EXPECTED_KPS, 3), dtype=np.float32)
    kps = np.asarray(inst.get("keypoints", []), dtype=np.float32)
    if kps.ndim != 2 or kps.shape[0] != EXPECTED_KPS:
        raise ValueError(f"Expected keypoints shape [133,2/3], got {kps.shape}")
    if kps.shape[1] >= 3:
        return kps[:, :3].astype(np.float32)
    scores = np.asarray(inst.get("keypoint_scores", np.zeros((EXPECTED_KPS,), dtype=np.float32)), dtype=np.float32)
    if scores.shape[0] != EXPECTED_KPS:
        raise ValueError(f"Expected 133 keypoint_scores, got {scores.shape}")
    return np.concatenate([kps.astype(np.float32), scores[:, None]], axis=1)


def parse_pred_record(record: Dict) -> np.ndarray:
    # mmpose inferencer output wrapper:
    # {'predictions': [[{'keypoints': ..., 'keypoint_scores': ...}, ...]], ...}
    if isinstance(record, dict) and "predictions" in record:
        preds = record.get("predictions", [])
        if isinstance(preds, list) and len(preds) > 0:
            first = preds[0]
            if isinstance(first, list) and len(first) > 0 and isinstance(first[0], dict):
                return inst_to_133x3(pick_best_instance(first))
            if isinstance(first, dict):
                return parse_pred_record(first)
        return np.zeros((EXPECTED_KPS, 3), dtype=np.float32)

    # direct instance dict from mmpose json-like output
    if isinstance(record, dict) and "keypoints" in record:
        return inst_to_133x3(record)

    instances = record.get("instances")
    if isinstance(instances, list):
        return inst_to_133x3(pick_best_instance(instances))

    pred_instances = record.get("pred_instances")
    if isinstance(pred_instances, dict):
        arr = np.asarray(pred_instances.get("keypoints", []), dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] > 0:
            scores = pred_instances.get("keypoint_scores")
            score_arr = None if scores is None else np.asarray(scores, dtype=np.float32)
            candidates = []
            for i in range(arr.shape[0]):
                cand = {"keypoints": arr[i]}
                if score_arr is not None and score_arr.ndim >= 2:
                    cand["keypoint_scores"] = score_arr[i]
                candidates.append(cand)
            return inst_to_133x3(pick_best_instance(candidates))
    return np.zeros((EXPECTED_KPS, 3), dtype=np.float32)


def build_masa_pose_pkl(frame_files: List[Path], inferencer) -> Dict[str, np.ndarray]:
    keypoints: List[np.ndarray] = []
    img_list: List[str] = []

    for i, frame in enumerate(frame_files):
        gen = inferencer(str(frame), show=False, return_vis=False)
        pred = next(gen)
        kps = parse_pred_record(pred)
        keypoints.append(kps)
        img_list.append(frame.name)
        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(frame_files)} frames")

    if not keypoints:
        arr = np.zeros((1, EXPECTED_KPS, 3), dtype=np.float32)
        img_list = ["img_00001.jpg"]
    else:
        arr = np.stack(keypoints, axis=0).astype(np.float32)

    # WLASL loader uses ['img_list'][:-1], so store len T+1.
    img_list_plus = img_list + [img_list[-1]]
    return {"keypoints": arr, "img_list": img_list_plus}


def save_split_lists(out_root: Path, subset_num: int, items: List[VideoItem]) -> None:
    list_dir = out_root / "traintestlist" / f"WLASL{subset_num}"
    ensure_dir(list_dir)

    by_split: Dict[str, List[VideoItem]] = {"train": [], "val": [], "test": []}
    for it in items:
        by_split[it.split].append(it)

    for split in ("train", "val", "test"):
        out_file = list_dir / f"{split}list01.txt"
        with out_file.open("w", encoding="utf-8") as f:
            for it in by_split[split]:
                f.write(f"{it.dst_video_id}.mp4 {it.class_id}\n")


def main() -> None:
    args = parse_args()
    preproc_root = Path(args.preproc_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    if not preproc_root.exists():
        raise FileNotFoundError(f"preproc root not found: {preproc_root}")

    try:
        from mmpose.apis import MMPoseInferencer
    except Exception as e:
        raise RuntimeError(
            "mmpose is required. Install it in the active environment first. "
            "Example: pip install -U openmim && mim install mmengine mmcv mmdet && pip install mmpose"
        ) from e

    inferencer = MMPoseInferencer(
        pose2d=args.pose_model,
        pose2d_weights=args.pose_weights,
        det_model=args.det_model,
        det_weights=args.det_weights,
    )

    ensure_dir(out_root)
    class_to_id = stable_class_map(preproc_root)
    if not class_to_id:
        raise ValueError(f"No class folders found under {preproc_root}/<split>/frames/")

    items = collect_items(preproc_root, class_to_id)
    if not items:
        raise ValueError("No videos found in split/class/video directory structure.")

    for split in ("train", "val", "test"):
        ensure_dir(out_root / "jpg_video_ori" / split)
        ensure_dir(out_root / "Keypoints_2d_mmpose" / split)

    # Save class mapping for reproducibility.
    with (out_root / "class_to_id.json").open("w", encoding="utf-8") as f:
        json.dump(class_to_id, f, indent=2, ensure_ascii=True)

    save_split_lists(out_root, args.subset_num, items)

    done = 0
    fail = 0
    for idx, it in enumerate(items, start=1):
        print(f"[{idx}/{len(items)}] {it.split}/{it.class_name}/{it.src_dir.name} -> {it.dst_video_id}")
        dst_frames_dir = out_root / "jpg_video_ori" / it.split / it.dst_video_id
        dst_pose_pkl = out_root / "Keypoints_2d_mmpose" / it.split / f"{it.dst_video_id}.pkl"

        try:
            dst_frames = materialize_standardized_frames(
                it.src_dir, dst_frames_dir, copy_frames=args.copy_frames, overwrite=args.overwrite
            )
            if dst_pose_pkl.exists() and not args.overwrite:
                done += 1
                continue

            if not dst_frames:
                print("  [skip] no frames found")
                fail += 1
                continue

            video_data = build_masa_pose_pkl(dst_frames, inferencer)
            with dst_pose_pkl.open("wb") as f:
                pickle.dump(video_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            done += 1
        except Exception as e:
            print(f"  [fail] {e}")
            fail += 1

    print(f"[DONE] ok={done} fail={fail} out_root={out_root}")


if __name__ == "__main__":
    main()
