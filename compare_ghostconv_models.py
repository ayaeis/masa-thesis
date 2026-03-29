#!/usr/bin/env python3
import argparse
import copy
import json
import time
from pathlib import Path

import torch

from moco.builder_dist import MASA
from quantize_finetuned_int8_fp16_report import FlopsCounter, checkpoint_to_state_dict


def load_model(ckpt_path, num_class, dropout, use_ghost_conv, ghost_ratio):
    sd = checkpoint_to_state_dict(ckpt_path)
    model = MASA(
        skeleton_representation="graph-based",
        num_class=num_class,
        pretrain=False,
        dropout=dropout,
        use_ghost_conv=use_ghost_conv,
        ghost_ratio=ghost_ratio,
    )
    msd = model.state_dict()
    loadable = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    skipped = len(sd) - len(loadable)
    msd.update(loadable)
    model.load_state_dict(msd, strict=False)
    return model, {"loaded": len(loadable), "skipped": skipped}


def state_size_mb(model):
    total = 0
    for tensor in model.state_dict().values():
        if torch.is_tensor(tensor):
            total += tensor.numel() * tensor.element_size()
    return total / (1024 * 1024)


def param_count(model):
    return sum(p.numel() for p in model.parameters())


def make_inputs(batch_size, target_t, device):
    return {
        "rh": torch.randn(batch_size, target_t, 21, 2, device=device),
        "lh": torch.randn(batch_size, target_t, 21, 2, device=device),
        "body": torch.randn(batch_size, target_t, 7, 2, device=device),
        "mask": torch.ones(batch_size, target_t * 2, 21, 2, device=device),
    }


def benchmark(model, device, batch_size, target_t, warmup_steps, bench_steps):
    model = model.to(device).eval()
    x = make_inputs(batch_size, target_t, device)

    meter = FlopsCounter()
    meter.add_hooks(model)
    with torch.no_grad():
        _ = model(x)
    flops_per_batch = meter.flops
    meter.clear()

    for _ in range(warmup_steps):
        with torch.no_grad():
            _ = model(x)

    times = []
    for _ in range(bench_steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "params": param_count(model),
        "state_size_mb": state_size_mb(model),
        "flops_per_batch": flops_per_batch,
        "latency_ms_per_batch": sum(times) / len(times),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Compare baseline MASA against first-pass GhostConv architecture.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--num-class", type=int, default=100)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--ghost-ratio", type=int, default=2)
    p.add_argument("--target-t", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--bench-steps", type=int, default=30)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="ghostconv_compare.json")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    num_class = args.num_class

    baseline, baseline_load = load_model(args.ckpt, num_class, args.dropout, False, args.ghost_ratio)
    ghost, ghost_load = load_model(args.ckpt, num_class, args.dropout, True, args.ghost_ratio)

    baseline_stats = benchmark(copy.deepcopy(baseline), device, args.batch_size, args.target_t, args.warmup_steps, args.bench_steps)
    ghost_stats = benchmark(copy.deepcopy(ghost), device, args.batch_size, args.target_t, args.warmup_steps, args.bench_steps)
    baseline_stats["load_info"] = baseline_load
    ghost_stats["load_info"] = ghost_load
    ghost_stats["ghost_ratio"] = args.ghost_ratio

    summary = {
        "checkpoint": args.ckpt,
        "num_class": num_class,
        "baseline": baseline_stats,
        "ghost_first_pass": ghost_stats,
        "delta": {
            "param_change": ghost_stats["params"] - baseline_stats["params"],
            "state_size_mb_change": ghost_stats["state_size_mb"] - baseline_stats["state_size_mb"],
            "flops_change": ghost_stats["flops_per_batch"] - baseline_stats["flops_per_batch"],
            "latency_speedup_x": (
                baseline_stats["latency_ms_per_batch"] / ghost_stats["latency_ms_per_batch"]
                if ghost_stats["latency_ms_per_batch"] > 0
                else None
            ),
        },
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
