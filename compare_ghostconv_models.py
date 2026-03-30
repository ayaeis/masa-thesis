#!/usr/bin/env python3
import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from feeder.single_dataset.WLASL import WLASL
from quantize_finetuned_int8_fp16_report import (
    WLASLSupervisedEval,
    collate_supervised,
    evaluate_model,
    checkpoint_to_state_dict,
)
from moco.builder_dist import MASA


def load_model(ckpt_path, num_class, dropout, use_ghost_conv, ghost_ratio, ghost_mode):
    sd = checkpoint_to_state_dict(ckpt_path)
    model = MASA(
        skeleton_representation="graph-based",
        num_class=num_class,
        pretrain=False,
        dropout=dropout,
        use_ghost_conv=use_ghost_conv,
        ghost_ratio=ghost_ratio,
        ghost_mode=ghost_mode,
    )
    msd = model.state_dict()
    loadable = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    skipped = len(sd) - len(loadable)
    msd.update(loadable)
    model.load_state_dict(msd, strict=False)
    return model, {"loaded": len(loadable), "skipped": skipped}


def to_dict(res):
    return {
        "flops_per_batch": res.flops_per_batch,
        "latency_ms_per_batch": res.latency_ms_per_batch,
        "model_size_mb": res.model_size_mb,
        "param_count_tensors": res.param_count_tensors,
        "loss": res.loss,
        "accuracy": res.acc,
        "dtype_numel": res.dtype_numel,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Compare baseline MASA against GhostConv using the same evaluation protocol as quantization.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--subset-num", type=int, default=100)
    p.add_argument("--num-class", type=int, default=100)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--ghost-ratio", type=int, default=2)
    p.add_argument("--ghost-mode", type=str, default="all", choices=["kernel1", "all", "gt1"])
    p.add_argument("--target-t", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--temporal-sampling", type=str, default="index", choices=["index", "interpolate"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="ghostconv_compare.json")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    criterion = nn.CrossEntropyLoss()

    base_test = WLASL(data_root=args.data_root, data_split="test", subset_num=args.subset_num, use_cache=False)
    ds_test = WLASLSupervisedEval(base_test, target_t=args.target_t, temporal_sampling=args.temporal_sampling)
    test_loader = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_supervised,
    )

    baseline, baseline_load = load_model(args.ckpt, args.num_class, args.dropout, False, args.ghost_ratio, args.ghost_mode)
    ghost, ghost_load = load_model(args.ckpt, args.num_class, args.dropout, True, args.ghost_ratio, args.ghost_mode)

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_stats = evaluate_model(
        copy.deepcopy(baseline),
        test_loader,
        criterion,
        device,
        out_dir / "compare_baseline_state_dict.pth",
        args.warmup_steps,
    )
    ghost_stats = evaluate_model(
        copy.deepcopy(ghost),
        test_loader,
        criterion,
        device,
        out_dir / "compare_ghost_state_dict.pth",
        args.warmup_steps,
    )

    summary = {
        "checkpoint": args.ckpt,
        "num_class": args.num_class,
        "baseline": {**to_dict(baseline_stats), "load_info": baseline_load},
        "ghost_first_pass": {**to_dict(ghost_stats), "load_info": ghost_load, "ghost_ratio": args.ghost_ratio, "ghost_mode": args.ghost_mode},
        "delta": {
            "param_change": ghost_stats.param_count_tensors - baseline_stats.param_count_tensors,
            "state_size_mb_change": ghost_stats.model_size_mb - baseline_stats.model_size_mb,
            "flops_change": ghost_stats.flops_per_batch - baseline_stats.flops_per_batch,
            "latency_speedup_x": (
                baseline_stats.latency_ms_per_batch / ghost_stats.latency_ms_per_batch
                if ghost_stats.latency_ms_per_batch > 0
                else None
            ),
            "accuracy_drop": baseline_stats.acc - ghost_stats.acc,
        },
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
