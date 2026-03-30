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


def parse_args():
    p = argparse.ArgumentParser(description="Report matched evaluation metrics for a single fine-tuned checkpoint.")
    p.add_argument("--ckpt", required=True)
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
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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

    sd = checkpoint_to_state_dict(args.ckpt)
    num_class = infer_num_class(sd, args.num_class)
    model, load_info = load_model(
        args.ckpt,
        num_class,
        args.dropout,
        args.use_ghost_conv,
        args.ghost_ratio,
        args.ghost_mode,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = out_path.with_name(out_path.stem + "_state_dict.pth")
    result = evaluate_model(model, test_loader, criterion, device, state_path, args.warmup_steps)

    summary = sanitize_json(
        {
            "checkpoint": args.ckpt,
            "num_class": num_class,
            "load_info": load_info,
            "use_ghost_conv": args.use_ghost_conv,
            "ghost_ratio": args.ghost_ratio,
            "ghost_mode": args.ghost_mode,
            "metrics": to_dict(result),
        }
    )
    out_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
