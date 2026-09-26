#!/usr/bin/env python3
import json
from pathlib import Path

import torch

from msasl200_archive_loader import MSASLArchive

ROOT = "/workspace/MSASL/native_masa_ready/MSASL"
REPORT = Path("/workspace/MSASL/native_masa_ready/loader_preflight.json")


def scan(split):
    dataset = MSASLArchive(
        data_root=ROOT,
        data_split=split,
        class_num=200,
        use_cache=False,
    )

    failures = []
    lengths = []

    for index in range(len(dataset)):
        try:
            sample = dataset.get_sample(index)
            right = sample["right"]["kp2d"]
            left = sample["left"]["kp2d"]
            body = sample["body"]["body_pose"]

            if right.ndim != 3 or right.shape[1:] != (21, 2):
                raise ValueError(f"bad right shape: {tuple(right.shape)}")
            if left.shape != right.shape:
                raise ValueError(
                    f"left/right mismatch: {tuple(left.shape)} vs {tuple(right.shape)}"
                )
            if body.shape != (right.shape[0], 7, 2):
                raise ValueError(f"bad body shape: {tuple(body.shape)}")
            if not (
                torch.isfinite(right).all()
                and torch.isfinite(left).all()
                and torch.isfinite(body).all()
            ):
                raise ValueError("non-finite output tensor")

            lengths.append(int(right.shape[0]))
        except Exception as exc:
            failures.append({
                "index": index,
                "file": dataset.video_list[index].tolist(),
                "error": repr(exc),
            })

        if (index + 1) % 250 == 0 or index + 1 == len(dataset):
            print(
                f"[{split} {index + 1}/{len(dataset)}] "
                f"failures={len(failures)}",
                flush=True,
            )

    return {
        "clips": len(dataset),
        "train_boundary": getattr(dataset, "flag", None),
        "failures": failures,
        "min_frames": min(lengths) if lengths else None,
        "max_frames": max(lengths) if lengths else None,
    }


def main():
    result = {
        "train_and_val": scan("train"),
        "test": scan("test"),
    }
    REPORT.write_text(json.dumps(result, indent=2))

    failures = sum(len(value["failures"]) for value in result.values())
    print(f"[DONE] failures={failures} report={REPORT}")


if __name__ == "__main__":
    main()
