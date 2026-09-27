#!/usr/bin/env python3
"""Verify WLASL100 slide JSON against the original experiment reports."""

import argparse
import json
from pathlib import Path

from export_wlasl100_slides_data import build_payload


def first_difference(actual, expected, path="root"):
    if type(actual) is not type(expected):
        return "%s: type %s != %s" % (path, type(actual).__name__, type(expected).__name__)
    if isinstance(actual, dict):
        if set(actual) != set(expected):
            return "%s: keys %s != %s" % (path, sorted(actual), sorted(expected))
        for key in sorted(actual):
            difference = first_difference(actual[key], expected[key], "%s.%s" % (path, key))
            if difference:
                return difference
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            return "%s: length %d != %d" % (path, len(actual), len(expected))
        for index, (left, right) in enumerate(zip(actual, expected)):
            difference = first_difference(left, right, "%s[%d]" % (path, index))
            if difference:
                return difference
    elif actual != expected:
        return "%s: %r != %r" % (path, actual, expected)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Check that the WLASL100 Slides JSON exactly matches its source reports."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/workspace/masa-thesis/fall_results/wlasl100/overnight_final_run"),
    )
    parser.add_argument(
        "--slides-json",
        type=Path,
        default=Path("/workspace/masa-thesis/fall_results/wlasl100/wlasl100_slides_data.json"),
    )
    parser.add_argument("--deck-title", default="WLASL100")
    parser.add_argument("--dataset-label", default="Official reconstructed WLASL100")
    args = parser.parse_args()

    actual = json.loads(args.slides_json.read_text(encoding="utf-8"))
    expected = build_payload(args.results_root, args.deck_title, args.dataset_label)
    difference = first_difference(actual, expected)
    if difference:
        raise SystemExit("[FAIL] Slide JSON differs from source reports: %s" % difference)

    print("[PASS] Slide JSON exactly matches source reports.")
    print("Experiments:", len(actual["experiments"]))
    print("Output:", args.slides_json)


if __name__ == "__main__":
    main()
