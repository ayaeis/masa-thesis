#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


EXPECTED_KPS = 133
EXPECTED_DIMS = 3


@dataclass
class ParsedVideo:
    keypoints: np.ndarray  # [T, 133, 3]
    img_list: List[str]    # MASA expects this and WLASL loader slices [: -1]


def read_split_video_ids(wlasl_root: Path, subset_num: int, split: str) -> List[str]:
    split_file = wlasl_root / "traintestlist" / f"WLASL{subset_num}" / f"{split}list01.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")
    video_ids: List[str] = []
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            # Format usually: "<video_name> <label>"
            video_name = tokens[0]
            video_ids.append(Path(video_name).stem)
    return video_ids


def build_prediction_index(mmpose_root: Path) -> Dict[str, List[Path]]:
    idx: Dict[str, List[Path]] = {}
    for p in mmpose_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".json", ".pkl"}:
            continue
        idx.setdefault(p.stem, []).append(p)
    return idx


def pick_prediction_file(
    video_id: str,
    split: str,
    pred_index: Dict[str, List[Path]],
) -> Optional[Path]:
    candidates = pred_index.get(video_id, [])
    if not candidates:
        return None
    # Prefer paths that include split directory name, then shortest path.
    split_hits = [p for p in candidates if f"/{split}/" in p.as_posix()]
    preferred = split_hits if split_hits else candidates
    preferred = sorted(preferred, key=lambda x: len(x.as_posix()))
    return preferred[0]


def load_obj(path: Path) -> Any:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    with path.open("rb") as f:
        return pickle.load(f)


def choose_best_instance(instances: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not instances:
        return None
    best = None
    best_score = -1.0
    for ins in instances:
        kps = np.asarray(ins.get("keypoints", []), dtype=np.float32)
        if kps.size == 0:
            continue
        kps_score = ins.get("keypoint_scores")
        if kps_score is None and kps.ndim == 2 and kps.shape[1] >= 3:
            score = float(np.mean(kps[:, 2]))
        else:
            score_arr = np.asarray(kps_score, dtype=np.float32) if kps_score is not None else np.zeros((kps.shape[0],), dtype=np.float32)
            score = float(np.mean(score_arr))
        if score > best_score:
            best_score = score
            best = ins
    return best


def as_kps133x3(instance: Dict[str, Any]) -> np.ndarray:
    kps = np.asarray(instance.get("keypoints", []), dtype=np.float32)
    scores = instance.get("keypoint_scores")
    if kps.ndim != 2:
        raise ValueError(f"Invalid keypoints shape: {kps.shape}")

    if kps.shape[0] != EXPECTED_KPS:
        raise ValueError(f"Expected {EXPECTED_KPS} keypoints, got {kps.shape[0]}")

    if kps.shape[1] >= 3:
        out = kps[:, :3].astype(np.float32)
        return out

    if kps.shape[1] != 2:
        raise ValueError(f"Keypoints must have 2 or 3 dims, got {kps.shape[1]}")

    if scores is None:
        conf = np.zeros((EXPECTED_KPS, 1), dtype=np.float32)
    else:
        conf = np.asarray(scores, dtype=np.float32).reshape(EXPECTED_KPS, 1)
    out = np.concatenate([kps.astype(np.float32), conf], axis=1)
    return out


def extract_frame_name(record: Dict[str, Any], frame_idx: int) -> str:
    for key in ("img_name", "image_name", "frame_name"):
        val = record.get(key)
        if isinstance(val, str) and val:
            return Path(val).name
    for key in ("img_path", "image_path"):
        val = record.get(key)
        if isinstance(val, str) and val:
            return Path(val).name
    return f"img_{frame_idx + 1:05d}.jpg"


def normalize_records(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, dict):
        if "keypoints" in obj and "img_list" in obj:
            # Already in MASA-like form.
            kp = np.asarray(obj["keypoints"], dtype=np.float32)
            img_list = [str(x) for x in obj["img_list"]]
            records: List[Dict[str, Any]] = []
            for i in range(kp.shape[0]):
                records.append(
                    {
                        "frame_name": img_list[i] if i < len(img_list) else f"img_{i + 1:05d}.jpg",
                        "instances": [{"keypoints": kp[i, :, :2], "keypoint_scores": kp[i, :, 2]}],
                    }
                )
            return records
        if isinstance(obj.get("instance_info"), list):
            return obj["instance_info"]
        if isinstance(obj.get("predictions"), list):
            preds = obj["predictions"]
            records = []
            for i, frame_pred in enumerate(preds):
                if isinstance(frame_pred, list):
                    records.append({"frame_name": f"img_{i + 1:05d}.jpg", "instances": frame_pred})
                elif isinstance(frame_pred, dict):
                    records.append(frame_pred)
            return records
    if isinstance(obj, list):
        return obj
    raise ValueError("Unsupported prediction format. Expected dict/list from MMPose.")


def parse_video_prediction(obj: Any) -> ParsedVideo:
    records = normalize_records(obj)
    frame_items: List[Tuple[int, str, np.ndarray]] = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        frame_name = extract_frame_name(rec, i)

        instances = rec.get("instances")
        if instances is None and isinstance(rec.get("pred_instances"), dict):
            pred_ins = rec["pred_instances"]
            kps = np.asarray(pred_ins.get("keypoints", []), dtype=np.float32)
            kps_scores = pred_ins.get("keypoint_scores")
            if kps.ndim == 3:
                instances = []
                score_arr = None if kps_scores is None else np.asarray(kps_scores, dtype=np.float32)
                for n in range(kps.shape[0]):
                    inst: Dict[str, Any] = {"keypoints": kps[n]}
                    if score_arr is not None and score_arr.ndim >= 2:
                        inst["keypoint_scores"] = score_arr[n]
                    instances.append(inst)
            else:
                instances = []

        if not isinstance(instances, list):
            instances = []

        best = choose_best_instance(instances)
        if best is None:
            kps = np.zeros((EXPECTED_KPS, EXPECTED_DIMS), dtype=np.float32)
        else:
            kps = as_kps133x3(best)

        # stable numeric ordering by img index if available
        m = re.search(r"(\d+)", Path(frame_name).stem)
        order = int(m.group(1)) if m else i + 1
        frame_items.append((order, frame_name, kps))

    if not frame_items:
        raise ValueError("No frame predictions found after parsing.")

    frame_items.sort(key=lambda x: x[0])
    frame_names = [x[1] for x in frame_items]
    keypoints = np.stack([x[2] for x in frame_items], axis=0).astype(np.float32)

    # WLASL loader uses video_data['img_list'][:-1], so append one trailing frame token.
    if frame_names:
        img_list = frame_names + [frame_names[-1]]
    else:
        img_list = ["img_00001.jpg", "img_00001.jpg"]

    return ParsedVideo(keypoints=keypoints, img_list=img_list)


def validate_parsed(parsed: ParsedVideo) -> None:
    if parsed.keypoints.ndim != 3:
        raise ValueError(f"keypoints must be 3D, got shape {parsed.keypoints.shape}")
    t, k, d = parsed.keypoints.shape
    if k != EXPECTED_KPS or d != EXPECTED_DIMS:
        raise ValueError(f"keypoints must be [T,{EXPECTED_KPS},{EXPECTED_DIMS}], got {parsed.keypoints.shape}")
    if len(parsed.img_list) != t + 1:
        raise ValueError(
            f"img_list must be len T+1 for this MASA loader path. got len(img_list)={len(parsed.img_list)}, T={t}"
        )


def convert_split(
    wlasl_root: Path,
    mmpose_root: Path,
    out_root: Path,
    subset_num: int,
    split: str,
    overwrite: bool,
) -> Tuple[int, int]:
    video_ids = read_split_video_ids(wlasl_root, subset_num, split)
    pred_index = build_prediction_index(mmpose_root)
    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    miss = 0
    for vid in video_ids:
        out_file = out_dir / f"{vid}.pkl"
        if out_file.exists() and not overwrite:
            ok += 1
            continue

        pred_file = pick_prediction_file(vid, split, pred_index)
        if pred_file is None:
            miss += 1
            print(f"[MISS] {split}/{vid}: prediction file not found")
            continue

        try:
            obj = load_obj(pred_file)
            parsed = parse_video_prediction(obj)
            validate_parsed(parsed)
            data = {"keypoints": parsed.keypoints, "img_list": parsed.img_list}
            with out_file.open("wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            ok += 1
        except Exception as e:
            miss += 1
            print(f"[FAIL] {split}/{vid} from {pred_file}: {e}")
    return ok, miss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert MMPose per-video predictions to MASA WLASL Keypoints_2d_mmpose format."
    )
    p.add_argument("--wlasl-root", required=True, help="WLASL root containing traintestlist/ and jpg_video_ori/")
    p.add_argument("--mmpose-root", required=True, help="Root folder containing MMPose predictions (.json/.pkl)")
    p.add_argument(
        "--out-root",
        default=None,
        help="Output root for MASA keypoints. Default: <wlasl-root>/Keypoints_2d_mmpose",
    )
    p.add_argument("--subset-num", type=int, default=100, help="WLASL subset number (e.g., 100 for WLASL100)")
    p.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated splits to convert. Defaults to train,val,test",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output pkl files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    wlasl_root = Path(args.wlasl_root).expanduser().resolve()
    mmpose_root = Path(args.mmpose_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else (wlasl_root / "Keypoints_2d_mmpose")

    if not wlasl_root.exists():
        raise FileNotFoundError(f"--wlasl-root does not exist: {wlasl_root}")
    if not mmpose_root.exists():
        raise FileNotFoundError(f"--mmpose-root does not exist: {mmpose_root}")

    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    for s in splits:
        if s not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {s}. Use train,val,test")

    total_ok = 0
    total_miss = 0
    for split in splits:
        ok, miss = convert_split(
            wlasl_root=wlasl_root,
            mmpose_root=mmpose_root,
            out_root=out_root,
            subset_num=args.subset_num,
            split=split,
            overwrite=args.overwrite,
        )
        total_ok += ok
        total_miss += miss
        print(f"[{split}] converted={ok} failed_or_missing={miss}")

    print(f"[DONE] output_root={out_root} converted={total_ok} failed_or_missing={total_miss}")


if __name__ == "__main__":
    main()
