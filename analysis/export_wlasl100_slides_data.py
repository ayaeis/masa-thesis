#!/usr/bin/env python3
"""Export completed WLASL100 experiment reports into Google Slides-ready JSON."""

import argparse
import json
from pathlib import Path
from typing import Optional


METRIC_KEYS = (
    "accuracy",
    "param_count_tensors",
    "latency_ms_per_batch",
    "model_size_mb",
    "flops_per_batch",
)

GHOST_CONFIGS = {
    "allk": "All eligible convolutions",
    "k1": "1x1 convolutions",
    "gt1": "Spatial convolutions (>1x1)",
}

LOW_RANK_CONFIGS = (
    ("transformer", "r0125", "Transformer layers", "0.125"),
    ("transformer", "r25", "Transformer layers", "0.25"),
    ("transformer", "r05", "Transformer layers", "0.50"),
    ("transformer", "r075", "Transformer layers", "0.75"),
    ("all", "r0125", "Transformer, convolution, and projection layers", "0.125"),
    ("all", "r25", "Transformer, convolution, and projection layers", "0.25"),
    ("all", "r05", "Transformer, convolution, and projection layers", "0.50"),
    ("all", "r075", "Transformer, convolution, and projection layers", "0.75"),
)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def metric_values(payload: dict) -> dict:
    """Keep the presentation schema narrow and stable."""
    return {key: float(payload[key]) for key in METRIC_KEYS}


def report_metrics(report: dict) -> dict:
    return metric_values(report["metrics"])


def report_baseline(report: dict) -> dict:
    return metric_values(report["baseline"]["metrics"])


def quant_metrics(summary: dict) -> dict:
    return metric_values(summary["after_int8_supported_fp16_fallback"])


def quant_baseline(summary: dict, fallback: dict) -> dict:
    baseline = summary.get("baseline", {}).get("metrics")
    return metric_values(baseline) if baseline else fallback


def deltas(experiment: dict, baseline: dict) -> dict:
    return {
        "accuracy_pp": (experiment["accuracy"] - baseline["accuracy"]) * 100.0,
        "param_count": experiment["param_count_tensors"] - baseline["param_count_tensors"],
        "latency_ms": experiment["latency_ms_per_batch"] - baseline["latency_ms_per_batch"],
        "latency_speedup_x": (
            baseline["latency_ms_per_batch"] / experiment["latency_ms_per_batch"]
            if experiment["latency_ms_per_batch"] > 0
            else None
        ),
        "model_size_mb": experiment["model_size_mb"] - baseline["model_size_mb"],
        "flops_per_batch": experiment["flops_per_batch"] - baseline["flops_per_batch"],
    }


def experiment_entry(
    *,
    experiment_id: str,
    family: str,
    title: str,
    configuration: str,
    precision: str,
    metrics: dict,
    baseline_metrics: dict,
    recovery_reference: Optional[dict] = None,
    recovery_label: Optional[str] = None,
) -> dict:
    entry = {
        "id": experiment_id,
        "family": family,
        "title": title,
        "configuration": configuration,
        "precision": precision,
        "metrics": metrics,
        "baseline_label": "Dense baseline (FP32)",
        "baseline_metrics": baseline_metrics,
        "delta_vs_baseline": deltas(metrics, baseline_metrics),
    }
    if recovery_reference is not None:
        entry["recovery_reference_label"] = recovery_label
        entry["recovery_reference_metrics"] = recovery_reference
        entry["delta_vs_recovery_reference"] = deltas(metrics, recovery_reference)
    return entry


def build_payload(
    results_root: Path,
    deck_title: str = "WLASL100",
    dataset_label: str = "Official reconstructed WLASL100",
) -> dict:
    report_root = results_root / "reports"
    dense_report = read_json(report_root / "baseline_report.json")
    dense_fp32 = report_metrics(dense_report)

    quant_dense = read_json(results_root / "quant_baseline" / "summary.json")
    dense_quantized = quant_metrics(quant_dense)

    experiments = []

    for tag, description in GHOST_CONFIGS.items():
        report = read_json(report_root / f"ghost_{tag}_vs_baseline.json")
        ghost_fp32_metrics = report_metrics(report)
        experiments.append(experiment_entry(
            experiment_id=f"ghost_{tag}_fp32",
            family="Ghost Convolution",
            title="Ghost Convolution",
            configuration=description,
            precision="FP32",
            metrics=ghost_fp32_metrics,
            baseline_metrics=report_baseline(report),
        ))

        quant = read_json(results_root / f"quant_ghost_{tag}" / "summary.json")
        experiments.append(experiment_entry(
            experiment_id=f"ghost_{tag}_quantized",
            family="Ghost Convolution",
            title="Ghost Convolution",
            configuration=description,
            precision="INT8 supported layers with FP16 fallback",
            metrics=quant_metrics(quant),
            baseline_metrics=quant_baseline(quant, dense_fp32),
        ))

        kd_report = read_json(report_root / f"kd_ghost_{tag}_vs_baseline.json")
        # Retain the report as an existence check.  Use the direct Ghost report
        # below so latency is measured from the same report as the displayed model.
        read_json(report_root / f"kd_ghost_{tag}_vs_ghost.json")
        experiments.append(experiment_entry(
            experiment_id=f"kd_ghost_{tag}_fp32",
            family="Knowledge Distillation Recovery",
            title="Knowledge Distillation Recovery",
            configuration=description,
            precision="FP32",
            metrics=report_metrics(kd_report),
            baseline_metrics=report_baseline(kd_report),
            recovery_reference=ghost_fp32_metrics,
            recovery_label="Matching Ghost model (FP32)",
        ))

        kd_quant = read_json(results_root / f"quant_kd_ghost_{tag}" / "summary.json")
        ghost_quant = read_json(results_root / f"quant_ghost_{tag}" / "summary.json")
        experiments.append(experiment_entry(
            experiment_id=f"kd_ghost_{tag}_quantized",
            family="Knowledge Distillation Recovery",
            title="Knowledge Distillation Recovery",
            configuration=description,
            precision="INT8 supported layers with FP16 fallback",
            metrics=quant_metrics(kd_quant),
            baseline_metrics=quant_baseline(kd_quant, dense_fp32),
            recovery_reference=quant_metrics(ghost_quant),
            recovery_label="Matching Ghost model (quantized)",
        ))

    for target, rank_tag, target_description, rank in LOW_RANK_CONFIGS:
        root = results_root / "lowrank" / f"{target}_{rank_tag}"
        report = read_json(root / "report_vs_baseline.json")
        configuration = f"{target_description}; r = {rank}"
        experiments.append(experiment_entry(
            experiment_id=f"lowrank_{target}_{rank_tag}_fp32",
            family="Low-Rank Factorization",
            title="Low-Rank Factorization",
            configuration=configuration,
            precision="FP32",
            metrics=report_metrics(report),
            baseline_metrics=report_baseline(report),
        ))

        quant = read_json(root / "quantized" / "summary.json")
        experiments.append(experiment_entry(
            experiment_id=f"lowrank_{target}_{rank_tag}_quantized",
            family="Low-Rank Factorization",
            title="Low-Rank Factorization",
            configuration=configuration,
            precision="INT8 supported layers with FP16 fallback",
            metrics=quant_metrics(quant),
            baseline_metrics=quant_baseline(quant, dense_fp32),
        ))

    return {
        "deck_title": deck_title,
        "dataset": dataset_label,
        "metrics": ["Accuracy", "Parameter count", "Latency", "Model size", "FLOPs"],
        "baseline": {
            "fp32": dense_fp32,
            "quantized": dense_quantized,
        },
        "experiments": experiments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export WLASL100 results into a Google Slides-ready JSON payload."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/workspace/masa-thesis/fall_results/wlasl100/overnight_final_run"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/workspace/masa-thesis/fall_results/wlasl100/wlasl100_slides_data.json"),
    )
    parser.add_argument("--deck-title", default="WLASL100")
    parser.add_argument("--dataset-label", default="Official reconstructed WLASL100")
    args = parser.parse_args()

    payload = build_payload(args.results_root, args.deck_title, args.dataset_label)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[DONE] Exported {len(payload['experiments'])} experiment entries")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
