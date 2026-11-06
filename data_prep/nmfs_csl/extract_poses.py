#!/usr/bin/env python3
"""Plan, preview, and shard NMFs-CSL extraction using the WLASL pose pipeline."""
import argparse
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import pickle
import re
import socket
import sys
import tempfile
import time

import numpy as np

from data_prep.wlasl.extract_mmpose import build_pose_pkl

CONFIG = "td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py"
WEIGHTS = "hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def read_json(path):
    return json.loads(Path(path).read_text())


@contextlib.contextmanager
def lock(path):
    # Advisory locks are released by the OS on process exit, including interruption.
    import fcntl
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another worker holds " + str(path))
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def read_splits(list_root, classes=1067):
    rows = []
    seen = set()
    for split in ("train", "test"):
        labels = set()
        for line in (Path(list_root) / (split + "list01.txt")).read_text().splitlines():
            fields = line.split()
            if len(fields) != (2 if split == "train" else 1):
                raise ValueError("Invalid split row: " + line)
            path = PurePosixPath(fields[0])
            if len(path.parts) != 2 or path.is_absolute() or path.suffix != ".mp4":
                raise ValueError("Invalid clip path: " + line)
            label = int(path.parts[0])
            if not 0 <= label < classes or (split == "train" and int(fields[1]) != label):
                raise ValueError("Invalid label: " + line)
            clip = str(path.with_suffix(""))
            if clip in seen:
                raise ValueError("Duplicate or cross-split clip: " + clip)
            seen.add(clip)
            labels.add(label)
            rows.append({"clip": clip, "label": label, "split": split})
        if labels != set(range(classes)):
            raise ValueError("Missing classes in " + split)
    return sorted(rows, key=lambda row: row["clip"])


def checked_frames(root, row):
    # Same lexical ordering/extensions as WLASL, using cached directory entries.
    with os.scandir(Path(root) / row["clip"]) as entries:
        frames = sorted((Path(e.path) for e in entries if
                         Path(e.name).suffix.lower() in {".jpg", ".jpeg", ".png"}
                         and e.is_file()), key=lambda p: p.name)
    names = [p.name for p in frames]
    expected = ["image_{:05d}.jpg".format(i) for i in range(1, len(frames) + 1)]
    if not frames or names != expected:
        raise ValueError("Empty or nonconsecutive frames: " + row["clip"])
    if "frames" in row and len(frames) != row["frames"]:
        raise ValueError("Frame count changed: " + row["clip"])
    return frames


def balance(rows, shards):
    if shards < 1:
        raise ValueError("Shards must be positive")
    groups = [[] for _ in range(shards)]
    totals = [0] * shards
    for row in sorted(rows, key=lambda row: (-row["frames"], row["clip"])):
        shard = min(range(shards), key=lambda i: (totals[i], i))
        groups[shard].append(dict(row, shard=shard))
        totals[shard] += row["frames"]
    return sorted([r for group in groups for r in group], key=lambda row: row["clip"])


def plan(args):
    root, lists = args.frames_root.resolve(), args.lists_root.resolve()
    for path in (args.pose_config, args.pose_weights, args.frame_report, args.repair_record):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.pose_config.name != CONFIG or args.pose_weights.name != WEIGHTS:
        raise ValueError("Explicit W48 DARK 384x288 config and checkpoint required")
    report = read_json(args.frame_report)
    if report.get("passed") is not True or report.get("checked") != 2640656:
        raise ValueError("Full JPEG readability validation must pass first")
    rows = read_splits(lists)
    for i, row in enumerate(rows, 1):
        row["frames"] = len(checked_frames(root, row))
        if i % 500 == 0:
            print("[PLAN {}/{}]".format(i, len(rows)), flush=True)
    if sum(r["frames"] for r in rows) != 2640656:
        raise ValueError("Unexpected total frames")
    if sum(r["split"] == "train" for r in rows) != 25608 or len(rows) != 32010:
        raise ValueError("Unexpected official split counts")
    assets = {str(p.resolve()): digest(p) for p in (
        args.pose_config, args.pose_weights, args.frame_report, args.repair_record,
        lists / "trainlist01.txt", lists / "testlist01.txt",
        Path(__file__), Path(importlib.import_module("data_prep.wlasl.extract_mmpose").__file__),
    )}
    manifest = {
        "schema": 1, "frames_root": str(root), "lists_root": str(lists),
        "out_root": str(args.out_root.resolve()), "shards": args.shards,
        "pose_config": str(args.pose_config.resolve()),
        "pose_weights": str(args.pose_weights.resolve()), "assets": assets,
        "det_model": None, "det_weights": None,
        "items": balance(rows, args.shards),
        "serialization": "float32 [T,133,3]; img_list has T names, no WLASL sentinel",
    }
    manifest["id"] = object_hash(manifest)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with lock(str(args.manifest) + ".lock"):
        if args.manifest.exists() and read_json(args.manifest) != manifest:
            raise ValueError("Existing plan differs; use a new manifest/output directory")
        write_json(args.manifest, manifest)
    for shard in range(args.shards):
        selected = [r for r in manifest["items"] if r["shard"] == shard]
        print("Shard {}: {} clips, {} frames".format(
            shard, len(selected), sum(r["frames"] for r in selected)))
    print("PLAN READY:", args.manifest)


def load_plan(path):
    manifest = read_json(path)
    body = dict(manifest)
    saved = body.pop("id")
    if object_hash(body) != saved:
        raise ValueError("Manifest contents changed")
    for name, expected in manifest["assets"].items():
        if digest(name) != expected:
            raise ValueError("Input/source changed: " + name)
    return manifest


def model_metadata(inferencer, manifest):
    import torch
    inner = inferencer.inferencer
    detector = inner.detector
    if detector is None:
        raise ValueError("Automatic WLASL person detector is missing")
    # Hash resolved configs (including inherited settings) and actual loaded tensors.
    def state_hash(model):
        h = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            arr = tensor.detach().cpu().contiguous().numpy()
            h.update(name.encode())
            h.update(str((arr.dtype, arr.shape)).encode())
            h.update(arr.tobytes())
        return h.hexdigest()
    versions = {n: importlib.import_module(n).__version__ for n in
                ("torch", "numpy", "mmcv", "mmengine", "mmdet", "mmpose", "cv2")}
    wanted = {"torch": "1.8.1+cu111", "numpy": "1.22.4", "mmcv": "2.1.0",
              "mmengine": "0.10.7", "mmdet": "3.2.0", "mmpose": "1.3.2", "cv2": "5.0.0"}
    if versions != wanted:
        raise ValueError("Environment differs from verified WLASL environment: " + repr(versions))
    return {
        "manifest_id": manifest["id"], "versions": versions,
        "resolved_pose_config": inner.cfg.pretty_text,
        "resolved_detector_config": detector.model.cfg.pretty_text,
        "pose_tensor_sha256": state_hash(inner.model),
        "detector_tensor_sha256": state_hash(detector.model),
        "detector_categories": list(inner.det_cat_ids),
        "call": "inferencer(str(frame)); default thresholds; batch size 1",
        "person_selection": "WLASL pick_best_instance: highest mean keypoint confidence",
        "cuda_runtime": torch.version.cuda,
    }


def init_model(manifest):
    import torch
    from mmpose.apis import MMPoseInferencer
    if not torch.cuda.is_available():
        raise RuntimeError("A GPU allocation is required")
    inferencer = MMPoseInferencer(
        pose2d=manifest["pose_config"], pose2d_weights=manifest["pose_weights"],
        det_model=None, det_weights=None,
    )
    metadata = model_metadata(inferencer, manifest)
    path = Path(manifest["out_root"]) / "extraction_protocol.json"
    with lock(str(path) + ".lock"):
        if path.exists() and read_json(path) != metadata:
            raise ValueError("Resolved models or environment differ from existing extraction")
        write_json(path, metadata)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    return inferencer, object_hash(metadata)


def validate_pose(data, names, manifest_id, protocol_id):
    if not isinstance(data, dict):
        raise ValueError("Pose must be a dictionary")
    arr = data.get("keypoints")
    if not isinstance(arr, np.ndarray) or arr.dtype != np.float32 or arr.shape != (len(names), 133, 3):
        raise ValueError("Invalid pose shape/dtype")
    if not np.isfinite(arr).all() or data.get("img_list") != names:
        raise ValueError("Invalid values or frame alignment")
    if data.get("extraction_id") != manifest_id or data.get("protocol_id") != protocol_id:
        raise ValueError("Pose provenance differs")


def pose_data(frames, inferencer, manifest_id, protocol_id):
    data = build_pose_pkl(frames, inferencer)
    # The NMFs loader indexes every name; only WLASL removes the T+1 sentinel.
    data["img_list"] = data["img_list"][:-1]
    data.update(extraction_id=manifest_id, protocol_id=protocol_id)
    validate_pose(data, [p.name for p in frames], manifest_id, protocol_id)
    return data


def setup_layout(manifest):
    out = Path(manifest["out_root"])
    for name, target in (("jpg_video", Path(manifest["frames_root"]).resolve()),
                         ("traintestlist", Path(manifest["lists_root"]).resolve())):
        path = out / name
        if path.is_symlink():
            if path.resolve() != target:
                raise ValueError("Wrong existing alias: " + str(path))
        elif path.exists():
            raise ValueError("Refusing existing non-symlink: " + str(path))
        else:
            try:
                path.symlink_to(os.path.relpath(target, out), target_is_directory=True)
            except FileExistsError:
                if not path.is_symlink() or path.resolve() != target:
                    raise


def run(args):
    manifest = load_plan(args.manifest)
    if not 0 <= args.shard < manifest["shards"] or args.limit < 0:
        raise ValueError("Invalid shard or limit")
    out = Path(manifest["out_root"])
    with lock(out / ("shard-{}.lock".format(args.shard))):
        inferencer, protocol_id = init_model(manifest)
        setup_layout(manifest)
        import torch
        write_json(out / "attempts" / ("shard-{}-{}.json".format(args.shard, time.time_ns())), {
            "host": socket.gethostname(), "python": sys.version,
            "executable": sys.executable, "gpu": torch.cuda.get_device_name(0),
            "shard": args.shard, "limit": args.limit, "protocol_id": protocol_id,
            "manifest_id": manifest["id"], "started_unix": time.time(),
        })
        items = [r for r in manifest["items"] if r["shard"] == args.shard]
        if args.limit:
            items = items[:args.limit]
        failure_path = out / ("failures-{}.jsonl".format(args.shard))
        stats = {"selected": len(items), "written": 0, "skipped": 0, "failed": 0,
                 "new_frames": 0, "zero_detection_frames": 0}
        start = time.perf_counter()
        with failure_path.open("a") as failures:
            for i, row in enumerate(items, 1):
                try:
                    frames = checked_frames(manifest["frames_root"], row)
                    names = [p.name for p in frames]
                    path = out / "Joint_2D_mmpose" / (row["clip"] + ".pkl")
                    if path.exists():
                        # Only read locally produced pickles; never load untrusted pickle files.
                        with path.open("rb") as f:
                            data = pickle.load(f)
                        validate_pose(data, names, manifest["id"], protocol_id)
                        stats["skipped"] += 1
                    else:
                        data = pose_data(frames, inferencer, manifest["id"], protocol_id)
                        atomic_bytes(path, pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
                        stats["written"] += 1
                        stats["new_frames"] += len(frames)
                    stats["zero_detection_frames"] += int(
                        (~np.any(data["keypoints"][:, :, 2] > 0, axis=1)).sum())
                except Exception as exc:
                    stats["failed"] += 1
                    failures.write(json.dumps({"clip": row["clip"], "error": repr(exc),
                                               "time": time.time()}) + "\n")
                    failures.flush()
                    print("[FAIL]", row["clip"], repr(exc), flush=True)
                print("[{}/{}] {} {}".format(i, len(items), row["clip"], stats), flush=True)
        stats.update(elapsed_seconds=time.perf_counter() - start, pilot=bool(args.limit),
                     manifest_id=manifest["id"], protocol_id=protocol_id)
        stats["new_frames_per_second"] = stats["new_frames"] / stats["elapsed_seconds"]
        write_json(out / ("summary-{}-{}.json".format(
            args.shard, "pilot" if args.limit else "full")), stats)
        print("[DONE]", json.dumps(stats), flush=True)
        if stats["failed"]:
            raise SystemExit(1)


def overlay(args):
    from PIL import Image, ImageDraw
    manifest = load_plan(args.manifest)
    rows = [r for r in manifest["items"] if r["clip"] == args.clip]
    if len(rows) != 1:
        raise ValueError("Clip not in official splits")
    inferencer, _ = init_model(manifest)
    from data_prep.wlasl.extract_mmpose import parse_pred_record
    frames = checked_frames(manifest["frames_root"], rows[0])
    indices = sorted(set((0, len(frames) // 2, len(frames) - 1)))
    out = Path(manifest["out_root"]) / "overlays" / args.clip
    out.mkdir(parents=True, exist_ok=True)
    for index in indices:
        frame = frames[index]
        points = parse_pred_record(next(inferencer(str(frame))))
        with Image.open(frame) as src:
            image = src.convert("RGB")
        draw = ImageDraw.Draw(image)
        for j, (x, y, confidence) in enumerate(points):
            if confidence < 0.3:
                continue
            color = "lime" if j < 23 else "yellow" if j < 91 else "cyan" if j < 112 else "red"
            draw.ellipse((float(x) - 2, float(y) - 2, float(x) + 2, float(y) + 2), fill=color)
        target = out / frame.name
        image.save(target)
        print("OVERLAY:", target, flush=True)


def audit(args):
    manifest = load_plan(args.manifest)
    out = Path(manifest["out_root"])
    metadata = read_json(out / "extraction_protocol.json")
    if metadata["manifest_id"] != manifest["id"]:
        raise ValueError("Wrong extraction protocol")
    protocol_id = object_hash(metadata)
    issues = []
    zero_frames = 0
    for i, row in enumerate(manifest["items"], 1):
        try:
            frames = checked_frames(manifest["frames_root"], row)
            path = out / "Joint_2D_mmpose" / (row["clip"] + ".pkl")
            with path.open("rb") as f:
                data = pickle.load(f)
            validate_pose(data, [p.name for p in frames], manifest["id"], protocol_id)
            zero_frames += int((~np.any(data["keypoints"][:, :, 2] > 0, axis=1)).sum())
        except Exception as exc:
            issues.append({"clip": row["clip"], "error": repr(exc)})
        if i % 500 == 0:
            print("[AUDIT {}/{}] issues={}".format(i, len(manifest["items"]), len(issues)), flush=True)
    report = {"manifest_id": manifest["id"], "clips": len(manifest["items"]),
              "issues": issues, "zero_detection_frames": zero_frames, "passed": not issues}
    write_json(out / "pose_audit.json", report)
    print(json.dumps(report), flush=True)
    if issues:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    for name in ("frames-root", "lists-root", "out-root", "pose-config", "pose-weights",
                 "frame-report", "repair-record", "manifest"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--shards", type=int, default=3)
    for name in ("run", "overlay"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True)
        if name == "run":
            p.add_argument("--shard", type=int, required=True)
            p.add_argument("--limit", type=int, default=0, help="Pilot clip limit; 0 processes full shard")
        else:
            p.add_argument("--clip", required=True)
    p = sub.add_parser("audit")
    p.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    {"plan": plan, "run": run, "overlay": overlay, "audit": audit}[args.command](args)


if __name__ == "__main__":
    main()
