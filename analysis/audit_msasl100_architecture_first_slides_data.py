#!/usr/bin/env python3
"""Verify the MSASL100 architecture-first slide JSON against source reports."""

import argparse
import json
from pathlib import Path

from export_msasl100_architecture_first_slides_data import build_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
            "msasl100_architecture_first_slides_data.json"
        ),
    )
    args = parser.parse_args()

    expected = build_payload(args.results_root)
    actual = json.loads(args.slides_json.read_text(encoding="utf-8"))
    if actual != expected:
        raise SystemExit("[FAIL] Slide JSON does not exactly match current source reports.")
    print("[PASS] Architecture-first slide JSON exactly matches source reports.")


if __name__ == "__main__":
    main()
