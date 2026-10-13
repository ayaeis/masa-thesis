#!/usr/bin/env python3
"""Prepare official WLASL2000 frames while reusing WLASL300 artifacts.

WLASL100 is a subset of WLASL300, so the WLASL300 manifest covers every
previously prepared clip. Reused frame directories and pose files are linked
instead of copied; only unseen official videos are decoded with FFmpeg.
"""

import argparse
import json
import os
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


EXPECTED_SPLITS = {"train": 14296, "val": 3920, "test": 2879}
EXPECTED_CLASS_IDS = set(range(2000))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--preproc-root", required=True)
    parser.add_argument("--reuse-preproc-root", required=True)
    parser.add_argument("--reuse-masa-root", required=True)
    parser.add_argument("--masa-root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def safe_gloss(gloss):
    return gloss.replace("/", "_").strip()


def link_relative(src, dst):
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise RuntimeError(f"Missing reuse source: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        raise RuntimeError(f"Conflicting reuse destination: {dst}; expected a link to {src}")
    dst.symlink_to(os.path.relpath(src, dst.parent))


def has_frames(frame_dir):
    return any(image.is_file() and image.stat().st_size > 0
               for image in frame_dir.glob("img_*.jpg"))


def completed_frames(frame_dir):
    return (frame_dir.is_dir() and (frame_dir / ".complete").is_file()
            and has_frames(frame_dir))


def extract_frames(video_path, frame_dir):
    video_path = Path(video_path)
    frame_dir = Path(frame_dir)
    marker = frame_dir / ".complete"
    if marker.exists() or marker.is_symlink():
        if completed_frames(frame_dir):
            return "skipped", video_path.stem, ""
        raise RuntimeError(f"Invalid completed frame directory: {frame_dir}")

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
    if result.returncode != 0 or not has_frames(frame_dir):
        return "failed", video_path.stem, result.stderr.strip()

    marker.touch()
    return "done", video_path.stem, ""


def main():
    args = parse_args()
    raw_root = Path(args.raw_root)
    preproc_root = Path(args.preproc_root)
    reuse_preproc_root = Path(args.reuse_preproc_root)
    reuse_masa_root = Path(args.reuse_masa_root)
    masa_root = Path(args.masa_root)
    videos_root = raw_root / "videos"

    with (raw_root / "nslt_2000.json").open() as handle:
        nslt2000 = json.load(handle)
    with (raw_root / "WLASL_v0.3.json").open() as handle:
        metadata = json.load(handle)
    with (reuse_preproc_root / "official_wlasl300_manifest.json").open() as handle:
        reusable_rows = {row["video_id"]: row for row in json.load(handle)}

    split_counts = Counter(item["subset"] for item in nslt2000.values())
    class_ids = {int(item["action"][0]) for item in nslt2000.values()}
    if dict(split_counts) != EXPECTED_SPLITS:
        raise RuntimeError(f"Unexpected official split counts: {dict(split_counts)}")
    if class_ids != EXPECTED_CLASS_IDS:
        raise RuntimeError("Official WLASL2000 class IDs are not exactly 0-1999")

    video_to_gloss = {
        str(int(str(instance["video_id"]))): entry["gloss"]
        for entry in metadata
        for instance in entry.get("instances", [])
    }
    class_id_to_gloss = {}
    for candidate_id, candidate in nslt2000.items():
        class_id = int(candidate["action"][0])
        gloss = video_to_gloss.get(str(int(str(candidate_id))))
        if gloss is None:
            continue
        previous = class_id_to_gloss.get(class_id)
        if previous is not None and previous != gloss:
            raise RuntimeError(
                f"Conflicting glosses for class {class_id}: {previous!r}, {gloss!r}"
            )
        class_id_to_gloss[class_id] = gloss
    if set(class_id_to_gloss) != EXPECTED_CLASS_IDS:
        missing = sorted(EXPECTED_CLASS_IDS - set(class_id_to_gloss))
        raise RuntimeError(f"No recoverable gloss for class IDs: {missing[:20]}")

    rows = []
    for video_id, item in nslt2000.items():
        class_id = int(item["action"][0])
        reused = reusable_rows.get(video_id)
        if reused is not None:
            if reused["split"] != item["subset"]:
                raise RuntimeError(
                    f"Split mismatch for reusable clip {video_id}: "
                    f"{reused['split']} != {item['subset']}"
                )
            if int(reused["official_class_id"]) != class_id:
                raise RuntimeError(f"Class mismatch for reusable clip {video_id}")
            gloss = reused["gloss"]
        else:
            gloss = (
                video_to_gloss.get(str(int(str(video_id))))
                or class_id_to_gloss[class_id]
            )
        # Four-digit prefixes preserve numeric ordering for all 2,000 classes.
        # Reused source directories retain their original three-digit names.
        class_folder = f"{class_id:04d}__{safe_gloss(gloss)}"

        source = videos_root / f"{video_id}.mp4"
        if not source.is_file():
            raise RuntimeError(f"Missing raw MP4: {source}")
        rows.append(
            {
                "video_id": video_id,
                "split": item["subset"],
                "official_class_id": class_id,
                "gloss": gloss,
                "class_folder": class_folder,
                "source_video": str(source),
                "reused_from_wlasl300": reused is not None,
            }
        )

    rows.sort(key=lambda row: (row["split"], row["official_class_id"], row["video_id"]))
    preproc_root.mkdir(parents=True, exist_ok=True)
    with (preproc_root / "official_wlasl2000_manifest.json").open("w") as handle:
        json.dump(rows, handle, indent=2)

    reused = 0
    extract_tasks = []
    for row in rows:
        frame_dir = (
            preproc_root / row["split"] / "frames" / row["class_folder"] / row["video_id"]
        )
        pose_path = (
            masa_root / "WLASL" / "Pose" / row["split"]
            / f"{row['class_folder']}__{row['video_id']}.pkl"
        )
        if row["reused_from_wlasl300"]:
            old = reusable_rows[row["video_id"]]
            old_frame_dir = (
                reuse_preproc_root / old["split"] / "frames"
                / old["class_folder"] / old["video_id"]
            )
            old_pose_path = (
                reuse_masa_root / "Pose" / old["split"]
                / f"{old['class_folder']}__{old['video_id']}.pkl"
            )
            if not completed_frames(old_frame_dir):
                raise RuntimeError(f"Invalid reusable frame directory: {old_frame_dir}")
            if not old_pose_path.is_file():
                raise RuntimeError(f"Missing reusable pose file: {old_pose_path}")
            link_relative(old_frame_dir, frame_dir)
            link_relative(old_pose_path, pose_path)
            reused += 1
        else:
            extract_tasks.append((row["source_video"], str(frame_dir)))

    print(f"Official WLASL2000 clips: {len(rows)}")
    print(f"Reused WLASL300 frames and poses: {reused}")
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
                    f"[{i}/{len(extract_tasks)}] done={done} skipped={skipped} failed={failed}",
                    flush=True,
                )

    if failed:
        raise RuntimeError(f"Frame extraction failed for {failed} clips")
    print(
        f"[DONE] WLASL2000 frame preparation complete: "
        f"reused={reused}, extracted={done}, skipped={skipped}"
    )


if __name__ == "__main__":
    main()
