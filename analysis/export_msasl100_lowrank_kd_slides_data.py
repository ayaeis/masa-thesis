#!/usr/bin/env python3
"""Export the targeted MSASL100 low-rank KD ablation for an existing deck."""

import argparse
import json
from pathlib import Path

from export_wlasl100_slides_data import deltas, report_baseline, report_metrics


RANKS = (
    ("r0125", "0.125"),
    ("r25", "0.25"),
    ("r05", "0.50"),
)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_payload(results_root: Path) -> dict:
    """Build a small, auditable supplement from matched per-rank source files."""
    report_root = results_root / "lowrank_kd_recovery" / "reports"
    lowrank_root = results_root / "overnight_final_run" / "lowrank"
    kd_root = results_root / "lowrank_kd_recovery"
    experiments = []

    for rank_tag, rank in RANKS:
        report_path = report_root / f"all_{rank_tag}_vs_post_training_student.json"
        report = read_json(report_path)
        summary_path = kd_root / f"all_{rank_tag}" / "summary.json"
        summary = read_json(summary_path)
        expected_kd = str(kd_root / f"all_{rank_tag}" / "best.pth.tar")
        expected_student = str(lowrank_root / f"all_{rank_tag}" / "best.pth.tar")
        expected_teacher = str(results_root / "baseline" / "best.pth.tar")
        expected_targets = "transformer,conv,project_head"

        # One report evaluates both models in the same invocation. This keeps
        # displayed latency values matched and proves the intended checkpoint
        # assignment before exporting anything to Slides.
        if report.get("checkpoint") != expected_kd:
            raise ValueError(f"Wrong KD checkpoint in {report_path}: {report.get('checkpoint')}")
        if report.get("baseline", {}).get("checkpoint") != expected_student:
            raise ValueError(f"Wrong pre-KD checkpoint in {report_path}")
        if not report.get("use_low_rank") or not report.get("baseline", {}).get("use_low_rank"):
            raise ValueError(f"Low-rank structure missing in {report_path}")
        if float(report.get("low_rank_ratio")) != float(rank):
            raise ValueError(f"Wrong KD rank in {report_path}")
        if float(report["baseline"].get("low_rank_ratio")) != float(rank):
            raise ValueError(f"Wrong pre-KD rank in {report_path}")
        if report.get("low_rank_targets") != expected_targets:
            raise ValueError(f"Wrong KD target layers in {report_path}")
        if report["baseline"].get("low_rank_targets") != expected_targets:
            raise ValueError(f"Wrong pre-KD target layers in {report_path}")
        if summary.get("teacher_ckpt") != expected_teacher:
            raise ValueError(f"Wrong dense teacher in {summary_path}")
        if summary.get("student_init_ckpt") != expected_student:
            raise ValueError(f"Wrong low-rank student in {summary_path}")
        if float(summary.get("student_low_rank_ratio")) != float(rank):
            raise ValueError(f"Wrong student rank in {summary_path}")
        if summary.get("student_low_rank_targets") != expected_targets:
            raise ValueError(f"Wrong student target layers in {summary_path}")

        kd_metrics = report_metrics(report)
        student_metrics = report_baseline(report)
        experiments.append(
            {
                "id": f"lowrank_all_{rank_tag}",
                "rank": rank,
                "configuration": "Transformer, convolution, and projection layers",
                "pre_kd_metrics": student_metrics,
                "kd_metrics": kd_metrics,
                "delta_vs_pre_kd": deltas(kd_metrics, student_metrics),
                "source_report": str(report_path),
                "source_summary": str(summary_path),
            }
        )

    return {
        "deck_title": "MSASL100",
        "dataset": "Archive-derived MSASL100",
        "section_title": "Low-rank KD recovery",
        "setup": {
            "teacher": "Dense FP32 MSASL100 baseline",
            "student": "Completed post-training low-rank model",
            "targets": "Transformer, convolution, and projection layers",
            "epochs": 60,
            "kd_alpha": 0.5,
            "kd_temperature": 4.0,
        },
        "experiments": experiments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export MSASL100 low-rank KD results for an existing Slides deck."
    )
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
            "msasl100_lowrank_kd_slides_data.json"
        ),
    )
    args = parser.parse_args()

    payload = build_payload(args.results_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[PASS] Exported {len(payload['experiments'])} MSASL100 low-rank KD entries")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
