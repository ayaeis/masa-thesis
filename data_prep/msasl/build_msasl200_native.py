#!/usr/bin/env python3
import json
import pickle
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import h5py
import numpy as np
from PIL import Image


ROOT = Path("/workspace/MSASL")
METADATA = ROOT / "official_metadata"
ARCHIVE = ROOT / "third_party_pose_archive"
OUT_PARENT = ROOT / "native_masa_ready"
OUT = OUT_PARENT / "MSASL"
STAGE = OUT_PARENT / "MSASL.building"

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
    """Return a binary availability mask; it is not a pose confidence score."""
    valid = np.isfinite(coords).all(axis=-1)
    valid &= np.any(np.abs(coords) > 1e-6, axis=-1)

    # Reject only mathematically collapsed hand detections. MASA otherwise
    # interprets them as visible hands and produces zero-sized crops.
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


def main():
    if OUT.exists() or STAGE.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing output: {OUT} or {STAGE}"
        )

    source_lists = load_source_lists()
    label_map = json.loads((ARCHIVE / "msasl_200_maplabels.json").read_text())
    gloss_to_id = label_map["id_to_label"]

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
    collapsed_frames = 0

    for hdf_split, split in HDF_SPLITS:
        hdf_path = ARCHIVE / f"MSASL200_135-{hdf_split}.hdf5"
        with h5py.File(hdf_path, "r") as h5:
            for group_id in sorted(h5.keys(), key=int):
                group = h5[group_id]
                filename = Path(group["video_name"][()].decode()).name
                key = hdf_key(split, filename)
                gloss = group["label"][()].decode()
                class_id = gloss_to_id[gloss]

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

                data = np.asarray(group["data"][()], dtype=np.float32)
                if data.ndim != 3 or data.shape[1:] != (2, 135):
                    raise RuntimeError(f"Unexpected pose shape {data.shape}: {filename}")

                width = int(group["width"][()])
                height = int(group["height"][()])
                if width <= 0 or height <= 0:
                    raise RuntimeError(f"Invalid dimensions {width}x{height}: {filename}")

                # Archive: [T, 2, 135] normalized x/y coordinates.
                # MASA: [T, 133, 3] pixel coordinates plus binary validity.
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

                stem = Path(filename).stem
                pose_path = pose_root / split / f"{stem}.pkl"
                pose_path.parent.mkdir(parents=True, exist_ok=True)
                img_list = [f"img_{frame:05d}.jpg" for frame in range(1, len(keypoints) + 1)]

                with pose_path.open("wb") as f:
                    pickle.dump(
                        {"keypoints": keypoints, "img_list": img_list},
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

                # MSASL.py only opens img_00001.jpg to recover width and height.
                frame_dir = image_root / split / stem
                frame_dir.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (width, height), (0, 0, 0)).save(
                    frame_dir / "img_00001.jpg", quality=80
                )

                archive_counts[split] += 1

    unavailable = []
    list_counts = {}

    # Preserve original full-list order. MSASL.py deletes its native eight
    # positions before filtering to class IDs below 200.
    for split, records in source_lists.items():
        list_root.mkdir(parents=True, exist_ok=True)
        list_path = list_root / f"{split}list01.txt"

        with list_path.open("w") as f:
            for index, item in enumerate(records):
                class_id = int(item["label"])
                key = metadata_key(split, item)

                if class_id < 200:
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
                else:
                    # These entries maintain original positions but are removed
                    # by MSASL.py's class_num=200 filtering before any file I/O.
                    filename = f"unused_{split}_{index:05d}.mp4"

                f.write(f"{filename} {class_id}\n")

        list_counts[split] = len(records)

    if len(unavailable) != 1:
        raise RuntimeError(
            f"Expected exactly one unavailable MSASL200 record, found {len(unavailable)}"
        )

    unavailable_record = unavailable[0]
    if not (
        unavailable_record["split"] == "train"
        and unavailable_record["class_id"] == 94
        and unavailable_record["gloss"] == "right"
    ):
        raise RuntimeError(f"Unexpected unavailable record: {unavailable_record}")

    (STAGE / "unavailable_msasl200_records.json").write_text(
        json.dumps(unavailable, indent=2)
    )
    (STAGE / "class_id_to_gloss.json").write_text(
        json.dumps(label_map["label_to_id"], indent=2)
    )
    (STAGE / "build_summary.json").write_text(json.dumps({
        "source": "third_party_pose_archive",
        "pose_shape": "[T, 133, 3]",
        "img_list_convention": "T entries (native MSASL feeder)",
        "validity_channel": (
            "binary availability mask derived from finite/nonzero coordinates; "
            "collapsed hand detections are marked invalid"
        ),
        "archive_pose_files": dict(archive_counts),
        "full_official_list_rows": list_counts,
        "unavailable_records": unavailable,
    }, indent=2))

    expected_archive = {"train": 6296, "val": 2041, "test": 1359}
    if dict(archive_counts) != expected_archive:
        raise RuntimeError(
            f"Unexpected archive counts: {dict(archive_counts)}"
        )

    STAGE.rename(OUT)
    print("[DONE] Native MSASL200 dataset created.")
    print(f"Output: {OUT}")
    print(f"Pose files: {dict(archive_counts)}")
    print(f"Full-list rows retained: {list_counts}")
    print(f"Unavailable record: {unavailable_record}")
    print(f"Collapsed-hand frames invalidated: {collapsed_frames}")


if __name__ == "__main__":
    main()
