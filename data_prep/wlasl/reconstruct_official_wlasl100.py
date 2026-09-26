#!/usr/bin/env python3
import argparse
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


EXPECTED_SPLITS = {"train": 1442, "val": 338, "test": 258}
EXPECTED_CLASS_IDS = set(range(100))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--preproc-root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def safe_gloss(gloss):
    return gloss.replace("/", "_").strip()


def extract_frames(task):
    video_path, frame_dir = map(Path, task)
    marker = frame_dir / ".complete"

    if marker.exists():
        return "skipped", video_path.stem, ""

    # A missing marker means a previous attempt was interrupted.
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
    videos_root = raw_root / "videos"

    with (raw_root / "nslt_100.json").open() as f:
        nslt100 = json.load(f)
    with (raw_root / "WLASL_v0.3.json").open() as f:
        entries = json.load(f)

    video_to_gloss = {
        str(instance["video_id"]): entry["gloss"]
        for entry in entries
        for instance in entry.get("instances", [])
    }

    split_counts = Counter(item["subset"] for item in nslt100.values())
    class_ids = {int(item["action"][0]) for item in nslt100.values()}

    if dict(split_counts) != EXPECTED_SPLITS:
        raise RuntimeError(f"Unexpected split counts: {dict(split_counts)}")
    if class_ids != EXPECTED_CLASS_IDS:
        raise RuntimeError(
            f"Official class IDs are not exactly 0-99: {sorted(class_ids)}"
        )

    rows = []
    id_to_gloss = {}
    missing = []

    for video_id, item in nslt100.items():
        official_id = int(item["action"][0])
        gloss = video_to_gloss.get(video_id)
        if gloss is None:
            raise RuntimeError(f"No gloss mapping for video {video_id}")

        clean_gloss = safe_gloss(gloss)
        if official_id in id_to_gloss and id_to_gloss[official_id] != clean_gloss:
            raise RuntimeError(
                f"Official class {official_id} maps to multiple glosses: "
                f"{id_to_gloss[official_id]!r}, {clean_gloss!r}"
            )
        id_to_gloss[official_id] = clean_gloss

        source_video = videos_root / f"{video_id}.mp4"
        if not source_video.is_file():
            missing.append(video_id)

        rows.append(
            {
                "video_id": video_id,
                "official_class_id": official_id,
                "gloss": gloss,
                "split": item["subset"],
                "action": item["action"],
                # Lexicographic sorting of these folders reproduces IDs 0-99.
                "class_folder": f"{official_id:03d}__{clean_gloss}",
                "source_video": str(source_video),
            }
        )

    if missing:
        raise RuntimeError(f"Missing raw videos: {missing[:20]}")
    if len(id_to_gloss) != 100:
        raise RuntimeError(f"Expected 100 glosses, found {len(id_to_gloss)}")

    rows.sort(
        key=lambda row: (
            row["split"], row["official_class_id"], row["video_id"]
        )
    )

    preproc_root.mkdir(parents=True, exist_ok=True)
    with (preproc_root / "official_wlasl100_manifest.json").open("w") as f:
        json.dump(rows, f, indent=2)
    with (preproc_root / "official_class_id_to_gloss.json").open("w") as f:
        json.dump(
            {str(key): value for key, value in sorted(id_to_gloss.items())},
            f,
            indent=2,
        )

    print(f"Official clips: {len(rows)}")
    print(f"Official splits: {dict(sorted(split_counts.items()))}")
    print(f"Official classes: {len(id_to_gloss)}")
    print("Using all frames because Kaggle MP4s are already trimmed.")

    tasks = [
        (
            row["source_video"],
            str(
                preproc_root
                / row["split"]
                / "frames"
                / row["class_folder"]
                / row["video_id"]
            ),
        )
        for row in rows
    ]

    done = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(extract_frames, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            status, video_id, error = future.result()
            if status == "done":
                done += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                print(f"[FAIL] {video_id}: {error}", flush=True)

            if index % 25 == 0 or index == len(tasks):
                print(
                    f"[{index}/{len(tasks)}] "
                    f"done={done} skipped={skipped} failed={failed}",
                    flush=True,
                )

    if failed:
        raise RuntimeError(f"Frame extraction failed for {failed} videos")

    print(
        f"[DONE] WLASL100 reconstruction complete: "
        f"done={done} skipped={skipped} root={preproc_root}"
    )


if __name__ == "__main__":
    main()
