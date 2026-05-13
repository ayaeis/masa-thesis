#!/usr/bin/env python3
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path("/workspace/masa-thesis/final_result")
OUT_DIR = ROOT / "plots"


@dataclass
class ModelPoint:
    name: str
    family: str
    accuracy_pct: float
    flops_per_batch: float
    latency_ms_per_batch: float
    model_size_mb: float


FAMILY_STYLES: Dict[str, Dict[str, object]] = {
    "Baseline": {"color": "#1F4E79", "marker": "o"},
    "Quantized Baseline": {"color": "#5B8E7D", "marker": "s"},
    "Ghost": {"color": "#D17A22", "marker": "^"},
    "Quantized Ghost": {"color": "#A05A2C", "marker": "D"},
    "KD Ghost": {"color": "#7A6AA6", "marker": "P"},
    "Quantized KD Ghost": {"color": "#4C6A92", "marker": "X"},
}


DISPLAY_NAMES = {
    "Baseline": "Baseline",
    "Quant-Baseline": "Q-Baseline",
    "Ghost-all": "G-all",
    "Ghost-k1": "G-k1",
    "Ghost-gt1": "G-gt1",
    "QGhost-all": "QG-all",
    "QGhost-k1": "QG-k1",
    "QGhost-gt1": "QG-gt1",
    "KD-Ghost-all": "KDG-all",
    "KD-Ghost-k1": "KDG-k1",
    "KD-Ghost-gt1": "KDG-gt1",
    "QKD-Ghost-all": "QKDG-all",
    "QKD-Ghost-k1": "QKDG-k1",
    "QKD-Ghost-gt1": "QKDG-gt1",
}


LABEL_OFFSETS = {
    "Baseline": (8, 6),
    "Quant-Baseline": (8, -12),
    "Ghost-all": (8, 6),
    "Ghost-k1": (8, -12),
    "Ghost-gt1": (8, 6),
    "QGhost-all": (8, -12),
    "QGhost-k1": (8, 6),
    "QGhost-gt1": (8, -12),
    "KD-Ghost-all": (8, 6),
    "KD-Ghost-k1": (8, -12),
    "KD-Ghost-gt1": (8, 6),
    "QKD-Ghost-all": (8, -12),
    "QKD-Ghost-k1": (8, 6),
    "QKD-Ghost-gt1": (8, -12),
}


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def add_point(points: List[ModelPoint], name: str, family: str, metrics: Dict) -> None:
    points.append(
        ModelPoint(
            name=name,
            family=family,
            accuracy_pct=100.0 * float(metrics["accuracy"]),
            flops_per_batch=float(metrics["flops_per_batch"]),
            latency_ms_per_batch=float(metrics["latency_ms_per_batch"]),
            model_size_mb=float(metrics["model_size_mb"]),
        )
    )


def collect_points() -> List[ModelPoint]:
    points: List[ModelPoint] = []

    baseline = load_json(ROOT / "reports" / "baseline_report.json")
    add_point(points, "Baseline", "Baseline", baseline["metrics"])

    quant_baseline = load_json(ROOT / "quant_baseline" / "summary.json")
    add_point(
        points,
        "Quant-Baseline",
        "Quantized Baseline",
        quant_baseline["after_int8_supported_fp16_fallback"],
    )

    for tag, suffix in [("allk", "all"), ("k1", "k1"), ("gt1", "gt1")]:
        ghost = load_json(ROOT / "reports" / f"ghost_{tag}_vs_baseline.json")
        add_point(points, f"Ghost-{suffix}", "Ghost", ghost["metrics"])

        qghost = load_json(ROOT / f"quant_ghost_{tag}" / "summary.json")
        add_point(
            points,
            f"QGhost-{suffix}",
            "Quantized Ghost",
            qghost["after_int8_supported_fp16_fallback"],
        )

        kdghost = load_json(ROOT / "reports" / f"kd_ghost_{tag}_vs_baseline.json")
        add_point(points, f"KD-Ghost-{suffix}", "KD Ghost", kdghost["metrics"])

        qkdghost = load_json(ROOT / f"quant_kd_ghost_{tag}" / "summary.json")
        add_point(
            points,
            f"QKD-Ghost-{suffix}",
            "Quantized KD Ghost",
            qkdghost["after_int8_supported_fp16_fallback"],
        )

    return points


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(axis="both", labelsize=10, colors="#333333")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35, color="#B0B7C3")
    ax.set_axisbelow(True)


def annotate_points(ax, points: List[ModelPoint], x_attr: str, y_attr: str) -> None:
    for p in points:
        dx, dy = LABEL_OFFSETS.get(p.name, (8, 6))
        ax.annotate(
            DISPLAY_NAMES.get(p.name, p.name),
            (getattr(p, x_attr), getattr(p, y_attr)),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8.5,
            color="#222222",
        )


def make_legend() -> List[Line2D]:
    handles = []
    for family, style in FAMILY_STYLES.items():
        handles.append(
            Line2D(
                [],
                [],
                linestyle="",
                marker=style["marker"],
                markersize=7,
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                label=family,
            )
        )
    return handles


def plot_accuracy_vs_flops(points: List[ModelPoint]) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for p in points:
        style = FAMILY_STYLES[p.family]
        ax.scatter(
            p.flops_per_batch,
            p.accuracy_pct,
            s=62,
            color=style["color"],
            marker=style["marker"],
            edgecolors="white",
            linewidths=0.6,
        )

    annotate_points(ax, points, "flops_per_batch", "accuracy_pct")
    style_axes(ax)
    ax.set_xlabel("FLOPs per batch", fontsize=11, color="#222222")
    ax.set_ylabel("Accuracy (%)", fontsize=11, color="#222222")
    ax.set_title("Accuracy–FLOPs Trade-off", fontsize=13, color="#111111", pad=10)
    ax.legend(
        handles=make_legend(),
        loc="lower left",
        frameon=False,
        fontsize=9,
        ncol=2,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "accuracy_vs_flops.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT_DIR / "accuracy_vs_flops.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_vs_latency(points: List[ModelPoint]) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for p in points:
        style = FAMILY_STYLES[p.family]
        ax.scatter(
            p.latency_ms_per_batch,
            p.accuracy_pct,
            s=62,
            color=style["color"],
            marker=style["marker"],
            edgecolors="white",
            linewidths=0.6,
        )

    annotate_points(ax, points, "latency_ms_per_batch", "accuracy_pct")
    style_axes(ax)
    ax.set_xlabel("Latency per batch (ms)", fontsize=11, color="#222222")
    ax.set_ylabel("Accuracy (%)", fontsize=11, color="#222222")
    ax.set_title("Accuracy–Latency Trade-off", fontsize=13, color="#111111", pad=10)
    ax.legend(
        handles=make_legend(),
        loc="lower left",
        frameon=False,
        fontsize=9,
        ncol=2,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "accuracy_vs_latency.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT_DIR / "accuracy_vs_latency.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    points = collect_points()
    plot_accuracy_vs_flops(points)
    plot_accuracy_vs_latency(points)
    print(f"Saved plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()
