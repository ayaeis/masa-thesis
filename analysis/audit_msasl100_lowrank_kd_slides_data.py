#!/usr/bin/env python3
"""Verify the MSASL100 low-rank KD slide JSON against source reports."""

import argparse
import json
from pathlib import Path

from export_msasl100_lowrank_kd_slides_data import build_payload


def first_difference(actual, expected, path="root"):
    if type(actual) is not type(expected):
        return f"{path}: type {type(actual).__name__} != {type(expected).__name__}"
    if isinstance(actual, dict):
        if set(actual) != set(expected):
            return f"{path}: keys {sorted(actual)} != {sorted(expected)}"
        for key in sorted(actual):
            difference = first_difference(actual[key], expected[key], f"{path}.{key}")
            if difference:
                return difference
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            return f"{path}: length {len(actual)} != {len(expected)}"
        for index, (left, right) in enumerate(zip(actual, expected)):
            difference = first_difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
    elif actual != expected:
        return f"{path}: {actual!r} != {expected!r}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check MSASL100 low-rank KD slide JSON against source reports."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/workspace/masa-thesis/fall_results/msasl100"),
    )
    parser.add_argument(
        "--slides-json",
        type=Path,
        default=Path(
            "/workspace/masa-thesis/fall_results/msasl100/"
            "msasl100_lowrank_kd_slides_data.json"
        ),
    )
    args = parser.parse_args()

    actual = json.loads(args.slides_json.read_text(encoding="utf-8"))
    expected = build_payload(args.results_root)
    difference = first_difference(actual, expected)
    if difference:
        raise SystemExit(f"[FAIL] Slide JSON differs from source reports: {difference}")

    print("[PASS] Low-rank KD slide JSON exactly matches source reports.")
    print("Experiments:", len(actual["experiments"]))
    print("Output:", args.slides_json)


if __name__ == "__main__":
    main()
