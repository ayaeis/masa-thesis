#!/usr/bin/env python3
"""Convert the third-party MSASL1000 pose archive to MASA's native layout."""

import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import h5py
import numpy as np
from PIL import Image


ROOT = Path("/workspace/MSASL")
METADATA = ROOT / "official_metadata"
ARCHIVE = ROOT / "third_party_pose_archive" / "msasl1000"
REUSE_ARCHIVE = ROOT / "third_party_pose_archive"
OUT_PARENT = ROOT / "native_masa_ready"
OUT = OUT_PARENT / "MSASL1000"
STAGE = OUT_PARENT / "MSASL1000.building"
REUSE_ROOT = OUT_PARENT / "MSASL"

CLASS_COUNT = 1000
HDF_SPLITS = (("Train", "train"), ("Val", "val"), ("Test", "test"))
NAME_RE = re.compile(
    r"^(?P<video_id>.+)_(?P<start>\d+)_(?P<end>\d+)-msasl-(?P<gloss>.+)\.mp4$"
)


def canonical_id(value):
    return value.split("&", 1)[0].split("?", 1)[0]


def url_video_id(url):
    parsed = urlparse(url)
    if parsed.netloc.endswith("youtu.be"):
        return canonical_id(parsed.path.strip("/"))
    return canonical_id(parse_qs(parsed.query)["v"][0])


def metadata_key(split, item):
    return (split, url_video_id(item["url"]), int(item["start"]), int(item["end"]))


def hdf_key(split, filename):
    match = NAME_RE.match(filename)
    if not match:
        raise RuntimeError(f"Cannot parse archive filename: {filename}")
    value = match.groupdict()
    return (
        split,
        canonical_id(value["video_id"]),
        int(value["start"]),
        int(value["end"]),
    )


def strict_validity(coords):
    """Return a binary availability mask; this is not MMPose confidence."""
    valid = np.isfinite(coords).all(axis=-1)
    valid &= np.any(np.abs(coords) > 1e-6, axis=-1)

    # Preserve the MSASL200 treatment of collapsed hand detections.
    for start, stop in ((91, 112), (112, 133)):
        for frame in range(len(coords)):
            present = valid[frame, start:stop]
            if present.sum() < 10:
                continue
            points = coords[frame, start:stop][present]
            span = points.max(axis=0) - points.min(axis=0)
            if span[0] <= 1e-4 or span[1] <= 1e-4:
                valid[frame, start:stop] = False
    return valid


def load_source_lists():
    result = {}
    for split in ("train", "val", "test"):
        path = METADATA / "MS-ASL" / f"MSASL_{split}.json"
        if not path.is_file():
            raise RuntimeError(f"Missing official metadata: {path}")
        result[split] = json.loads(path.read_text())
    return result


def load_reusable_msasl200_clips():
    """Map the verified MSASL200 archive clips by split and source filename."""
    result = {}
    for hdf_split, split in HDF_SPLITS:
        path = REUSE_ARCHIVE / f"MSASL200_135-{hdf_split}.hdf5"
        if not path.is_file():
            raise RuntimeError(f"Missing reusable MSASL200 archive: {path}")
        with h5py.File(path, "r") as h5:
            for group_id in h5:
                group = h5[group_id]
                filename = Path(group["video_name"][()].decode()).name
                gloss = group["label"][()].decode()
                result[(split, filename)] = gloss
    return result


def link_reused_artifacts(split, stem):
    """Reuse the already validated MSASL200 conversion without duplicating files."""
    source_pose = REUSE_ROOT / "Keypoints_2d_mmpose" / split / f"{stem}.pkl"
    source_image = REUSE_ROOT / "jpg_video_ori" / split / stem / "img_00001.jpg"
    if not source_pose.is_file() or not source_image.is_file():
        raise RuntimeError(
            f"Missing converted MSASL200 artifacts for reusable clip: {split}/{stem}"
        )

    pose_path = STAGE / "Keypoints_2d_mmpose" / split / f"{stem}.pkl"
    pose_path.parent.mkdir(parents=True, exist_ok=True)
    pose_path.symlink_to(source_pose)

    image_path = STAGE / "jpg_video_ori" / split / stem / "img_00001.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.symlink_to(source_image)


def main():
    if OUT.exists() or STAGE.exists():
        raise RuntimeError(f"Refusing to overwrite existing output: {OUT} or {STAGE}")

    source_lists = load_source_lists()
    reusable_msasl200 = load_reusable_msasl200_clips()
    if not REUSE_ROOT.is_dir():
        raise RuntimeError(f"Missing converted MSASL200 dataset: {REUSE_ROOT}")
    label_map = json.loads((ARCHIVE / "msasl_1000_maplabels.json").read_text())
    gloss_to_id = label_map["id_to_label"]
    if len(gloss_to_id) != CLASS_COUNT or len(label_map["label_to_id"]) != CLASS_COUNT:
        raise RuntimeError("MSASL1000 label map does not contain exactly 1,000 classes")

    official_by_key = defaultdict(list)
    for split, records in source_lists.items():
        for index, item in enumerate(records):
            official_by_key[metadata_key(split, item)].append((index, item))

    STAGE.mkdir(parents=True)
    pose_root = STAGE / "Keypoints_2d_mmpose"
    image_root = STAGE / "jpg_video_ori"
    list_root = STAGE / "traintestlist"

    archive_name_by_key = {}
    archive_counts = Counter()
    reused_counts = Counter()
    converted_counts = Counter()
    collapsed_frames = 0

    for hdf_split, split in HDF_SPLITS:
        hdf_path = ARCHIVE / f"MSASL1000_135-{hdf_split}.hdf5"
        if not hdf_path.is_file():
            raise RuntimeError(f"Missing archive: {hdf_path}")

        with h5py.File(hdf_path, "r") as h5:
            for group_id in sorted(h5.keys(), key=int):
                group = h5[group_id]
                filename = Path(group["video_name"][()].decode()).name
                key = hdf_key(split, filename)
                gloss = group["label"][()].decode()
                class_id = int(gloss_to_id[gloss])

                matches = [
                    item for _, item in official_by_key[key]
                    if int(item["label"]) == class_id
                ]
                if not matches:
                    raise RuntimeError(
                        f"No official numeric-label match for archive clip: {filename}"
                    )
                if key in archive_name_by_key:
                    raise RuntimeError(f"Duplicate archive identity: {key}")
                archive_name_by_key[key] = filename

                stem = Path(filename).stem
                reusable_gloss = reusable_msasl200.get((split, filename))
                if reusable_gloss is not None:
                    if reusable_gloss != gloss:
                        raise RuntimeError(
                            f"MSASL200/MSASL1000 gloss mismatch for {filename}: "
                            f"{reusable_gloss!r} != {gloss!r}"
                        )
                    link_reused_artifacts(split, stem)
                    archive_counts[split] += 1
                    reused_counts[split] += 1
                    continue

                data = np.asarray(group["data"][()], dtype=np.float32)
                if data.ndim != 3 or data.shape[1:] != (2, 135):
                    raise RuntimeError(f"Unexpected pose shape {data.shape}: {filename}")
                width, height = int(group["width"][()]), int(group["height"][()])
                if width <= 0 or height <= 0:
                    raise RuntimeError(f"Invalid dimensions {width}x{height}: {filename}")

                # Archive: [T, 2, 135] normalized x/y. MASA: [T, 133, 3].
                coords = np.moveaxis(data[:, :, :133], 1, 2).copy()
                coords[..., 0] *= width
                coords[..., 1] *= height
                valid = strict_validity(coords)
                collapsed_frames += int(
                    np.sum(
                        np.any(np.isfinite(coords), axis=(1, 2))
                        & (
                            ~valid[:, 91:112].any(axis=1)
                            | ~valid[:, 112:133].any(axis=1)
                        )
                    )
                )
                coords[~valid] = 0.0
                keypoints = np.concatenate(
                    [coords, valid[..., None].astype(np.float32)], axis=-1
                ).astype(np.float32)

                pose_path = pose_root / split / f"{stem}.pkl"
                pose_path.parent.mkdir(parents=True, exist_ok=True)
                img_list = [f"img_{frame:05d}.jpg" for frame in range(1, len(keypoints) + 1)]
                with pose_path.open("wb") as f:
                    pickle.dump(
                        {"keypoints": keypoints, "img_list": img_list},
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

                # The native feeder opens this image only to read dimensions.
                frame_dir = image_root / split / stem
                frame_dir.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (width, height), (0, 0, 0)).save(
                    frame_dir / "img_00001.jpg", quality=80
                )
                archive_counts[split] += 1
                converted_counts[split] += 1

    unavailable = []
    list_counts = {}
    list_root.mkdir(parents=True, exist_ok=True)
    for split, records in source_lists.items():
        list_path = list_root / f"{split}list01.txt"
        with list_path.open("w") as f:
            for index, item in enumerate(records):
                class_id = int(item["label"])
                if not 0 <= class_id < CLASS_COUNT:
                    raise RuntimeError(f"Unexpected official label {class_id}")
                key = metadata_key(split, item)
                filename = archive_name_by_key.get(key)
                if filename is None:
                    filename = f"unavailable_{split}_{index:05d}.mp4"
                    unavailable.append({
                        "split": split,
                        "source_index": index,
                        "filename": filename,
                        "class_id": class_id,
                        "gloss": item["clean_text"],
                        "url": item["url"],
                        "start": int(item["start"]),
                        "end": int(item["end"]),
                    })
                f.write(f"{filename} {class_id}\n")
        list_counts[split] = len(records)

    (STAGE / "unavailable_records.json").write_text(json.dumps(unavailable, indent=2))
    (STAGE / "native_feeder_exclusions.json").write_text("[]\n")
    (STAGE / "class_id_to_gloss.json").write_text(
        json.dumps(label_map["label_to_id"], indent=2)
    )
    (STAGE / "build_summary.json").write_text(json.dumps({
        "source": "third_party_pose_archive/msasl1000",
        "class_count": CLASS_COUNT,
        "pose_shape": "[T, 133, 3]",
        "img_list_convention": "T entries (native MSASL feeder)",
        "validity_channel": (
            "binary availability mask derived from finite/nonzero coordinates; "
            "collapsed hand detections are marked invalid"
        ),
        "archive_pose_files": dict(archive_counts),
        "reused_msasl200_pose_files": dict(reused_counts),
        "newly_converted_pose_files": dict(converted_counts),
        "full_official_list_rows": list_counts,
        "unavailable_records": unavailable,
        "collapsed_hand_frames_invalidated": collapsed_frames,
    }, indent=2))

    STAGE.rename(OUT)
    print("[DONE] Native MSASL1000 dataset created.")
    print(f"Output: {OUT}")
    print(f"Pose files: {dict(archive_counts)}")
    print(f"Reused MSASL200 pose files: {dict(reused_counts)}")
    print(f"Newly converted pose files: {dict(converted_counts)}")
    print(f"Full-list rows retained: {list_counts}")
    print(f"Unavailable records: {len(unavailable)}")
    print(f"Collapsed-hand frames invalidated: {collapsed_frames}")


if __name__ == "__main__":
    main()
