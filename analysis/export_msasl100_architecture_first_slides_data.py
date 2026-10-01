#!/usr/bin/env python3
"""Export the MSASL100 post-training versus architecture-first comparison."""

import argparse
import json
from pathlib import Path


METRIC_KEYS = (
    "accuracy",
    "param_count_tensors",
    "latency_ms_per_batch",
    "model_size_mb",
    "flops_per_batch",
)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(payload: dict) -> dict:
    return {key: float(payload[key]) for key in METRIC_KEYS}


def report_metrics(report: dict) -> dict:
    return metrics(report["metrics"])


def report_baseline(report: dict) -> dict:
    return metrics(report["baseline"]["metrics"])


def assert_same_metrics(reference: dict, candidate: dict, label: str) -> None:
    for key in METRIC_KEYS:
        if reference[key] != candidate[key]:
            raise ValueError(
                f"Baseline mismatch in {label} for {key}: "
                f"{candidate[key]} != {reference[key]}"
            )


def entry(label: str, stage: str, payload: dict) -> dict:
    return {
        "label": label,
        "stage": stage,
        "metrics": report_metrics(payload),
        "baseline_metrics": report_baseline(payload),
    }


def build_payload(results_root: Path) -> dict:
    post_root = results_root / "overnight_final_run"
    architecture_root = results_root / "architecture_first"

    baseline_report = read_json(post_root / "reports" / "baseline_report.json")
    baseline = report_metrics(baseline_report)

    source_paths = {
        "ghost": {
            "post_training": post_root / "reports" / "ghost_allk_vs_baseline.json",
            "architecture_first_60": architecture_root / "reports" / "ghost_allk_epoch60_vs_baseline.json",
            "architecture_first_120": architecture_root / "reports" / "ghost_allk_epoch120_vs_baseline.json",
        },
        "low_rank": {
            "post_training": post_root / "lowrank" / "all_r0125" / "report_vs_baseline.json",
            "architecture_first_60": architecture_root / "reports" / "lowrank_all_r0125_epoch60_vs_baseline.json",
            "architecture_first_120": architecture_root / "reports" / "lowrank_all_r0125_epoch120_vs_baseline.json",
        },
    }

    labels = {
        "post_training": "Post-training",
        "architecture_first_60": "Architecture-first, 60 epochs",
        "architecture_first_120": "Architecture-first, two 60-epoch stages",
    }
    methods = {}
    for method, paths in source_paths.items():
        items = []
        for stage, path in paths.items():
            report = read_json(path)
            item = entry(labels[stage], stage, report)
            assert_same_metrics(baseline, item["baseline_metrics"], str(path))
            items.append(item)
        methods[method] = items

    return {
        "deck_title": "MSASL100: Post-Training vs Architecture-First Compression",
        "dataset": "Archive-derived MSASL100, 100 classes",
        "metrics": ["Accuracy", "Parameter count", "Latency", "Model size", "FLOPs"],
        "baseline": baseline,
        "methods": methods,
        "protocol": {
            "post_training": "Dense MASA fine-tuning for 60 epochs, then compressed-model recovery for 60 epochs.",
            "architecture_first_60": "Generic MASA pretraining, then compressed-model fine-tuning for 60 epochs.",
            "architecture_first_120": "Generic MASA pretraining, then two compressed-model stages of 60 epochs each.",
            "controlled": "Same MSASL100 split, seed, optimizer, batch size, temporal sampling, and per-stage learning-rate schedule.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/workspace/masa-thesis/fall_results/msasl100"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/workspace/masa-thesis/fall_results/msasl100/"
            "msasl100_architecture_first_slides_data.json"
        ),
    )
    args = parser.parse_args()

    payload = build_payload(args.results_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[PASS] Exported MSASL100 architecture-first slide data: {args.out}")


if __name__ == "__main__":
    main()
