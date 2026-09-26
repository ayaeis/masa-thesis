#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

EXPECTED_SPLITS = {"train": 3549, "val": 901, "test": 668}
EXPECTED_CLASS_IDS = set(range(300))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", required=True)
    p.add_argument("--preproc-root", required=True)
    p.add_argument("--old-preproc-root", required=True)
    p.add_argument("--old-masa-root", required=True)
    p.add_argument("--masa-root", required=True)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def safe_gloss(gloss):
    return gloss.replace("/", "_").strip()


def link_relative(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(os.path.relpath(src, dst.parent))


def extract_frames(video_path, frame_dir):
    video_path = Path(video_path)
    frame_dir = Path(frame_dir)
    marker = frame_dir / ".complete"

    if marker.exists():
        return "skipped", video_path.stem, ""

    if frame_dir.exists():
        for image in frame_dir.glob("img_*.jpg"):
            image.unlink()
    frame_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video_path),
            "-q:v", "2",
            str(frame_dir / "img_%06d.jpg"),
        ],
        text=True,
        capture_output=True,
    )

    if result.returncode != 0 or not any(frame_dir.glob("img_*.jpg")):
        return "failed", video_path.stem, result.stderr.strip()

    marker.touch()
    return "done", video_path.stem, ""


def main():
    args = parse_args()
    raw_root = Path(args.raw_root)
    preproc_root = Path(args.preproc_root)
    old_preproc_root = Path(args.old_preproc_root)
    old_masa_root = Path(args.old_masa_root)
    masa_root = Path(args.masa_root)

    with (raw_root / "nslt_300.json").open() as f:
        nslt300 = json.load(f)
    with (raw_root / "WLASL_v0.3.json").open() as f:
        metadata = json.load(f)
    with (old_preproc_root / "official_wlasl100_manifest.json").open() as f:
        old_rows = {row["video_id"]: row for row in json.load(f)}

    video_to_gloss = {
        str(int(str(instance["video_id"]))): entry["gloss"]
        for entry in metadata
        for instance in entry.get("instances", [])
    }

    split_counts = Counter(item["subset"] for item in nslt300.values())
    class_ids = {int(item["action"][0]) for item in nslt300.values()}

    if dict(split_counts) != EXPECTED_SPLITS:
        raise RuntimeError(f"Unexpected split counts: {dict(split_counts)}")
    if class_ids != EXPECTED_CLASS_IDS:
        raise RuntimeError("Official WLASL300 class IDs are not exactly 0-299")

    # A small number of nslt_300 IDs are absent from WLASL_v0.3.json.
    # Recover their gloss from another clip with the same official class ID.
    class_id_to_gloss = {}
    for candidate_id, candidate in nslt300.items():
        candidate_class_id = int(candidate["action"][0])
        candidate_gloss = video_to_gloss.get(str(int(candidate_id)))
        if candidate_gloss is None:
            continue
        previous = class_id_to_gloss.get(candidate_class_id)
        if previous is not None and previous != candidate_gloss:
            raise RuntimeError(
                f"Conflicting glosses for official class {candidate_class_id}: "
                f"{previous!r}, {candidate_gloss!r}"
            )
        class_id_to_gloss[candidate_class_id] = candidate_gloss

    if set(class_id_to_gloss) != EXPECTED_CLASS_IDS:
        missing_ids = sorted(EXPECTED_CLASS_IDS - set(class_id_to_gloss))
        raise RuntimeError(f"No recoverable gloss for class IDs: {missing_ids}")

    rows = []
    for video_id, item in nslt300.items():
        class_id = int(item["action"][0])
        gloss = (
            video_to_gloss.get(str(int(video_id)))
            or class_id_to_gloss.get(class_id)
        )
        if gloss is None:
            raise RuntimeError(f"No gloss mapping for {video_id}")

        split = item["subset"]
        source = raw_root / "videos" / f"{video_id}.mp4"
        if not source.is_file():
            raise RuntimeError(f"Missing raw MP4: {source}")

        rows.append({
            "video_id": video_id,
            "split": split,
            "official_class_id": class_id,
            "gloss": gloss,
            "class_folder": f"{class_id:03d}__{safe_gloss(gloss)}",
            "source_video": str(source),
            "reused_from_wlasl100": video_id in old_rows,
        })

    rows.sort(key=lambda row: (row["split"], row["official_class_id"], row["video_id"]))
    preproc_root.mkdir(parents=True, exist_ok=True)

    with (preproc_root / "official_wlasl300_manifest.json").open("w") as f:
        json.dump(rows, f, indent=2)

    reused_frames = reused_poses = 0
    extract_tasks = []

    for row in rows:
        frame_dir = (
            preproc_root / row["split"] / "frames"
            / row["class_folder"] / row["video_id"]
        )

        pose_path = (
            masa_root / "WLASL" / "Pose" / row["split"]
            / f"{row['class_folder']}__{row['video_id']}.pkl"
        )

        if row["reused_from_wlasl100"]:
            old = old_rows[row["video_id"]]
            if old["split"] != row["split"]:
                raise RuntimeError(f"Split mismatch for reused video {row['video_id']}")

            old_frame_dir = (
                old_preproc_root / old["split"] / "frames"
                / old["class_folder"] / old["video_id"]
            )
            old_pose_path = (
                old_masa_root / "Pose" / old["split"]
                / f"{old['class_folder']}__{old['video_id']}.pkl"
            )

            if not old_frame_dir.is_dir() or not (old_frame_dir / ".complete").exists():
                raise RuntimeError(f"Invalid reusable frame directory: {old_frame_dir}")
            if not old_pose_path.is_file():
                raise RuntimeError(f"Missing reusable pose file: {old_pose_path}")

            link_relative(old_frame_dir, frame_dir)
            link_relative(old_pose_path, pose_path)
            reused_frames += 1
            reused_poses += 1
        else:
            extract_tasks.append((row["source_video"], str(frame_dir)))

    print(f"Official WLASL300 clips: {len(rows)}")
    print(f"Reused WLASL100 frames and poses: {reused_frames}")
    print(f"New clips requiring frame extraction: {len(extract_tasks)}")

    done = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(extract_frames, *task) for task in extract_tasks]
        for i, future in enumerate(as_completed(futures), start=1):
            status, video_id, error = future.result()
            if status == "done":
                done += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                print(f"[FAIL] {video_id}: {error}", flush=True)

            if i % 25 == 0 or i == len(extract_tasks):
                print(
                    f"[{i}/{len(extract_tasks)}] "
                    f"done={done} skipped={skipped} failed={failed}",
                    flush=True,
                )

    if failed:
        raise RuntimeError(f"Frame extraction failed for {failed} clips")

    print(
        f"[DONE] WLASL300 preparation complete: "
        f"reused={reused_frames}, extracted={done}, skipped={skipped}"
    )


if __name__ == "__main__":
    main()
