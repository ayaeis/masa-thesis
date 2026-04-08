#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from feeder.single_dataset.WLASL import WLASL
from moco.builder_dist import MASA
from quantize_finetuned_int8_fp16_report import (
    WLASLSupervisedEval,
    checkpoint_to_state_dict,
    collate_supervised,
    evaluate_model,
    infer_num_class,
    sanitize_json,
    to_dict,
)


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


def load_eval_result(
    ckpt_path,
    data_root,
    subset_num,
    target_t,
    batch_size,
    workers,
    num_class,
    dropout,
    warmup_steps,
    temporal_sampling,
    device,
    state_path,
    use_ghost_conv,
    ghost_ratio,
    ghost_mode,
):
    criterion = nn.CrossEntropyLoss()
    base_test = WLASL(data_root=data_root, data_split="test", subset_num=subset_num, use_cache=False)
    ds_test = WLASLSupervisedEval(base_test, target_t=target_t, temporal_sampling=temporal_sampling)
    test_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_supervised,
    )

    sd = checkpoint_to_state_dict(ckpt_path)
    inferred_num_class = infer_num_class(sd, num_class)
    model, load_info = load_model(
        ckpt_path,
        inferred_num_class,
        dropout,
        use_ghost_conv,
        ghost_ratio,
        ghost_mode,
    )
    result = evaluate_model(model, test_loader, criterion, device, state_path, warmup_steps)
    return inferred_num_class, load_info, result


def parse_args():
    p = argparse.ArgumentParser(description="Report matched evaluation metrics for a single fine-tuned checkpoint.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--baseline-ckpt", default=None)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--subset-num", type=int, default=100)
    p.add_argument("--target-t", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--num-class", type=int, default=100)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--temporal-sampling", type=str, default="index", choices=["index", "interpolate"])
    p.add_argument("--use-ghost-conv", action="store_true")
    p.add_argument("--ghost-ratio", type=int, default=2)
    p.add_argument("--ghost-mode", type=str, default="all", choices=["kernel1", "all", "gt1"])
    p.add_argument("--baseline-use-ghost-conv", action="store_true")
    p.add_argument("--baseline-ghost-ratio", type=int, default=2)
    p.add_argument("--baseline-ghost-mode", type=str, default="all", choices=["kernel1", "all", "gt1"])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    num_class, load_info, result = load_eval_result(
        args.ckpt,
        args.data_root,
        args.subset_num,
        args.target_t,
        args.batch_size,
        args.workers,
        args.num_class,
        args.dropout,
        args.warmup_steps,
        args.temporal_sampling,
        device,
        out_path := Path(args.out).with_name(Path(args.out).stem + "_state_dict.pth"),
        args.use_ghost_conv,
        args.ghost_ratio,
        args.ghost_mode,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "checkpoint": args.ckpt,
        "num_class": num_class,
        "load_info": load_info,
        "use_ghost_conv": args.use_ghost_conv,
        "ghost_ratio": args.ghost_ratio,
        "ghost_mode": args.ghost_mode,
        "metrics": to_dict(result),
    }

    if args.baseline_ckpt:
        _, baseline_load_info, baseline_result = load_eval_result(
            args.baseline_ckpt,
            args.data_root,
            args.subset_num,
            args.target_t,
            args.batch_size,
            args.workers,
            args.num_class,
            args.dropout,
            args.warmup_steps,
            args.temporal_sampling,
            device,
            out_path.with_name(out_path.stem + "_baseline_state_dict.pth"),
            args.baseline_use_ghost_conv,
            args.baseline_ghost_ratio,
            args.baseline_ghost_mode,
        )
        summary["baseline"] = {
            "checkpoint": args.baseline_ckpt,
            "load_info": baseline_load_info,
            "use_ghost_conv": args.baseline_use_ghost_conv,
            "ghost_ratio": args.baseline_ghost_ratio,
            "ghost_mode": args.baseline_ghost_mode,
            "metrics": to_dict(baseline_result),
        }
        summary["delta_vs_baseline"] = {
            "loss_change": result.loss - baseline_result.loss,
            "accuracy_drop": baseline_result.acc - result.acc,
            "latency_speedup_x": (
                baseline_result.latency_ms_per_batch / result.latency_ms_per_batch
                if result.latency_ms_per_batch > 0
                else None
            ),
            "flops_change": result.flops_per_batch - baseline_result.flops_per_batch,
            "param_count_change": result.param_count_tensors - baseline_result.param_count_tensors,
            "model_size_mb_change": result.model_size_mb - baseline_result.model_size_mb,
        }

    summary = sanitize_json(summary)
    out_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
