#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


EXPECTED_KPS = 133
# Paper-required MMPose setting: Topdown Heatmap + HRNet + DARK on COCO-WholeBody
DEFAULT_POSE_CONFIG = (
    "configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/"
    "td-hm_hrnet-w32_dark-8xb64-210e_coco-wholebody-256x192.py"
)
DEFAULT_POSE_WEIGHTS = (
    "https://download.openmmlab.com/mmpose/top_down/hrnet/"
    "hrnet_w32_coco_wholebody_256x192_dark-469327ef_20200922.pth"
)


@dataclass
class VideoItem:
    split: str
    class_name: str
    class_id: int
    src_dir: Path
    video_id: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract WLASL100 keypoints with MMPose (Topdown Heatmap + HRNet + DARK, "
            "COCO-WholeBody) and write Data/WLASL/{Video,Pose,Annotations}."
        )
    )
    p.add_argument(
        "--preproc-root",
        required=True,
        help="Path like .../preprocessing containing train/val/test/frames.",
    )
    p.add_argument(
        "--out-root",
        required=True,
        help="Path where WLASL folder will be created.",
    )
    p.add_argument(
        "--subset-num",
        type=int,
        default=100,
        help="Subset tag in list folder name, e.g., WLASL100.",
    )
    p.add_argument(
        "--pose-config",
        default=DEFAULT_POSE_CONFIG,
        help="MMPose config path (default pinned to HRNet32 DARK COCO-WholeBody).",
    )
    p.add_argument(
        "--pose-weights",
        default=DEFAULT_POSE_WEIGHTS,
        help="MMPose checkpoint URL/path (default pinned to HRNet32 DARK COCO-WholeBody).",
    )
    p.add_argument(
        "--det-model",
        default=None,
        help=(
            "Detector alias/path for MMPoseInferencer. Default uses MMPose auto detector "
            "for top-down pose (detector-backed). Set to whole_image only if you explicitly "
            "want no detector."
        ),
    )
    p.add_argument(
        "--det-weights",
        default=None,
        help="Optional detector checkpoint URL/path.",
    )
    p.add_argument(
        "--copy-frames",
        action="store_true",
        help="Copy frames into output Video tree. Default uses symlinks.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    p.add_argument(
        "--emit-masa-aliases",
        action="store_true",
        help=(
            "Also create MASA loader aliases under WLASL root: "
            "jpg_video_ori -> Video, Keypoints_2d_mmpose -> Pose, traintestlist -> Annotations."
        ),
    )
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def frame_paths(video_dir: Path) -> List[Path]:
    frames = [p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    return sorted(frames, key=lambda p: p.name)


def stable_class_map(preproc_root: Path) -> Dict[str, int]:
    classes = set()
    for split in ("train", "val", "test"):
        frames_root = preproc_root / split / "frames"
        if not frames_root.exists():
            continue
        for cls_dir in frames_root.iterdir():
            if cls_dir.is_dir():
                classes.add(cls_dir.name)
    return {c: i for i, c in enumerate(sorted(classes))}


def collect_items(preproc_root: Path, class_to_id: Dict[str, int]) -> List[VideoItem]:
    items: List[VideoItem] = []
    for split in ("train", "val", "test"):
        split_root = preproc_root / split / "frames"
        if not split_root.exists():
            continue
        for cls_dir in sorted(split_root.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls = cls_dir.name
            cid = class_to_id[cls]
            for vid_dir in sorted(cls_dir.iterdir()):
                if not vid_dir.is_dir():
                    continue
                vid = f"{cls}__{vid_dir.name}"
                items.append(VideoItem(split=split, class_name=cls, class_id=cid, src_dir=vid_dir, video_id=vid))
    return items


def materialize_frames(src_dir: Path, dst_dir: Path, copy_frames: bool, overwrite: bool) -> List[Path]:
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
    out: List[Path] = []
    for i, src in enumerate(src_frames, start=1):
        dst = dst_dir / f"img_{i:05d}.jpg"
        if copy_frames:
            shutil.copy2(src, dst)
        else:
            rel = os.path.relpath(src, dst.parent)
            dst.symlink_to(rel)
        out.append(dst)
    return out


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
        raise ValueError(f"Expected keypoints [133,2/3], got {kps.shape}")
    if kps.shape[1] >= 3:
        return kps[:, :3].astype(np.float32)
    scores = np.asarray(inst.get("keypoint_scores", np.zeros((EXPECTED_KPS,), dtype=np.float32)), dtype=np.float32)
    if scores.shape[0] != EXPECTED_KPS:
        raise ValueError(f"Expected 133 keypoint_scores, got {scores.shape}")
    return np.concatenate([kps.astype(np.float32), scores[:, None]], axis=1)


def parse_pred_record(record: Dict) -> np.ndarray:
    if isinstance(record, dict) and "predictions" in record:
        preds = record.get("predictions", [])
        if isinstance(preds, list) and len(preds) > 0:
            first = preds[0]
            if isinstance(first, list) and len(first) > 0 and isinstance(first[0], dict):
                return inst_to_133x3(pick_best_instance(first))
            if isinstance(first, dict):
                return parse_pred_record(first)
        return np.zeros((EXPECTED_KPS, 3), dtype=np.float32)

    if isinstance(record, dict) and "keypoints" in record:
        return inst_to_133x3(record)

    pred_instances = record.get("pred_instances")
    if isinstance(pred_instances, dict):
        arr = np.asarray(pred_instances.get("keypoints", []), dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] > 0:
            scores = pred_instances.get("keypoint_scores")
            score_arr = None if scores is None else np.asarray(scores, dtype=np.float32)
            cand = []
            for i in range(arr.shape[0]):
                item = {"keypoints": arr[i]}
                if score_arr is not None and score_arr.ndim >= 2:
                    item["keypoint_scores"] = score_arr[i]
                cand.append(item)
            return inst_to_133x3(pick_best_instance(cand))
    return np.zeros((EXPECTED_KPS, 3), dtype=np.float32)


def build_pose_pkl(frame_files: List[Path], inferencer) -> Dict[str, np.ndarray]:
    keypoints: List[np.ndarray] = []
    img_list: List[str] = []
    for frame in frame_files:
        pred = next(inferencer(str(frame)))
        keypoints.append(parse_pred_record(pred))
        img_list.append(frame.name)
    if not keypoints:
        arr = np.zeros((1, EXPECTED_KPS, 3), dtype=np.float32)
        img_list = ["img_00001.jpg"]
    else:
        arr = np.stack(keypoints, axis=0).astype(np.float32)
    # MASA loader uses img_list[:-1], so keep T+1 convention.
    return {"keypoints": arr, "img_list": img_list + [img_list[-1]]}


def write_split_lists(ann_root: Path, subset_num: int, items: List[VideoItem]) -> None:
    list_root = ann_root / f"WLASL{subset_num}"
    ensure_dir(list_root)
    per = {"train": [], "val": [], "test": []}
    for it in items:
        per[it.split].append(it)
    for split in ("train", "val", "test"):
        out = list_root / f"{split}list01.txt"
        with out.open("w", encoding="utf-8") as f:
            for it in per[split]:
                f.write(f"{it.video_id}.mp4 {it.class_id}\n")


def emit_masa_aliases(wlasl_root: Path, overwrite: bool) -> None:
    aliases = {
        "jpg_video_ori": wlasl_root / "Video",
        "Keypoints_2d_mmpose": wlasl_root / "Pose",
        "traintestlist": wlasl_root / "Annotations",
    }
    for name, target in aliases.items():
        link = wlasl_root / name
        if link.exists() or link.is_symlink():
            if overwrite:
                if link.is_symlink() or link.is_file():
                    link.unlink()
                else:
                    shutil.rmtree(link)
            else:
                continue
        rel = os.path.relpath(target, link.parent)
        link.symlink_to(rel)


def main() -> None:
    args = parse_args()
    preproc_root = Path(args.preproc_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    wlasl_root = out_root / "WLASL"

    if not preproc_root.exists():
        raise FileNotFoundError(f"preproc root not found: {preproc_root}")

    try:
        from mmpose.apis import MMPoseInferencer
    except Exception as e:
        raise RuntimeError(
            "mmpose is required in current env. Install mmpose/mmengine/mmcv(-lite)/mmdet as needed."
        ) from e

    inferencer = MMPoseInferencer(
        pose2d=args.pose_config,
        pose2d_weights=args.pose_weights,
        det_model=args.det_model,
        det_weights=args.det_weights,
    )

    class_to_id = stable_class_map(preproc_root)
    if not class_to_id:
        raise ValueError(f"No classes found under {preproc_root}/<split>/frames")
    items = collect_items(preproc_root, class_to_id)
    if not items:
        raise ValueError("No videos found under preprocessing split/class/video tree.")

    video_root = wlasl_root / "Video"
    pose_root = wlasl_root / "Pose"
    ann_root = wlasl_root / "Annotations"
    for split in ("train", "val", "test"):
        ensure_dir(video_root / split)
        ensure_dir(pose_root / split)
    ensure_dir(ann_root)

    # Persist extraction metadata for reproducibility.
    meta = {
        "pose_config": args.pose_config,
        "pose_weights": args.pose_weights,
        "det_model": args.det_model,
        "det_weights": args.det_weights,
        "expected_keypoints": EXPECTED_KPS,
        "class_count": len(class_to_id),
    }
    with (ann_root / "extract_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with (ann_root / "class_to_id.json").open("w", encoding="utf-8") as f:
        json.dump(class_to_id, f, indent=2, ensure_ascii=True)
    write_split_lists(ann_root, args.subset_num, items)

    ok = 0
    fail = 0
    for i, it in enumerate(items, start=1):
        print(f"[{i}/{len(items)}] {it.split}/{it.class_name}/{it.src_dir.name} -> {it.video_id}")
        dst_frames_dir = video_root / it.split / it.video_id
        dst_pose_pkl = pose_root / it.split / f"{it.video_id}.pkl"
        try:
            frames = materialize_frames(it.src_dir, dst_frames_dir, args.copy_frames, args.overwrite)
            if dst_pose_pkl.exists() and not args.overwrite:
                ok += 1
                continue
            if not frames:
                print("  [skip] no frames found")
                fail += 1
                continue
            data = build_pose_pkl(frames, inferencer)
            with dst_pose_pkl.open("wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            ok += 1
        except Exception as e:
            print(f"  [fail] {e}")
            fail += 1

    if args.emit_masa_aliases:
        emit_masa_aliases(wlasl_root, overwrite=args.overwrite)

    print(f"[DONE] ok={ok} fail={fail} root={wlasl_root}")


if __name__ == "__main__":
    main()
