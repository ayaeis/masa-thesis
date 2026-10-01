#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data_prep.msasl.msasl200_archive_loader import MSASLArchive as MSASL
from finetune import MSASLSupervised, collate_supervised, run_epoch, set_seed, strip_prefix_if_present
from moco.builder_dist import MASA
from moco.low_rank_modules import apply_low_rank_from_dense, apply_low_rank_structure


def load_checkpoint_compatible(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    state_dict = strip_prefix_if_present(state_dict, "module.")
    model_state = model.state_dict()
    loadable = {}
    skipped = 0
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            loadable[k] = v
        else:
            skipped += 1
    model_state.update(loadable)
    model.load_state_dict(model_state, strict=False)
    print(f"[weights] loaded={len(loadable)} skipped={skipped}")
    return {"loaded": len(loadable), "skipped": skipped}


def load_low_rank_checkpoint_strict(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    state_dict = strip_prefix_if_present(state_dict, "module.")
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def validate_resume_configuration(checkpoint, args):
    if not isinstance(checkpoint, dict):
        raise ValueError("A low-rank resume checkpoint must include configuration metadata.")
    expected = {
        "rank_ratio": float(args.rank_ratio),
        "low_rank_targets": args.low_rank_targets,
        "low_rank_min_features": int(args.low_rank_min_features),
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f"Resume checkpoint {key}={checkpoint.get(key)!r} does not match requested {value!r}."
            )


def parse_args():
    p = argparse.ArgumentParser(description="Architecture-first low-rank MASA fine-tuning from generic pretraining or a prior factorized stage.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--init-ckpt", required=True)
    p.add_argument(
        "--init-mode",
        required=True,
        choices=["pretrained_dense", "resume_low_rank"],
        help="Start from generic pretraining and factorize, or strictly load a prior factorized checkpoint.",
    )
    p.add_argument("--num-class", type=int, default=100)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--optim", type=str, default="sgd", choices=["sgd", "adam", "adamw"])
    p.add_argument("--scheduler", type=str, default="multistep", choices=["multistep", "cosine", "none"])
    p.add_argument("--milestones", type=int, nargs="+", default=[20, 40])
    p.add_argument("--lr-gamma", type=float, default=0.1)
    p.add_argument("--freeze-epochs", type=int, default=0)
    p.add_argument("--head-lr-mult", type=float, default=1.0)
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--train-temporal-crop-min", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=999)
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--target-t", type=int, default=32)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--temporal-sampling", type=str, default="index", choices=["index", "interpolate"])
    p.add_argument("--rank-ratio", type=float, default=0.25)
    p.add_argument("--low-rank-targets", type=str, default="transformer")
    p.add_argument("--low-rank-min-features", type=int, default=64)
    p.add_argument("--out-dir", type=str, default="./ckpt_lowrank")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_train = MSASL(
        args.data_root, data_split="train", class_num=args.num_class, use_cache=False
    )
    base_test = MSASL(
        args.data_root, data_split="test", class_num=args.num_class, use_cache=False
    )

    n_train = int(base_train.flag)
    n_total = len(base_train)
    train_idx = list(range(0, n_train))
    val_idx = list(range(n_train, n_total))

    ds_train_all = MSASLSupervised(base_train, target_t=args.target_t, train=True, temporal_sampling=args.temporal_sampling)
    ds_val_all = MSASLSupervised(base_train, target_t=args.target_t, train=False, temporal_sampling=args.temporal_sampling)
    ds_train = Subset(ds_train_all, train_idx)
    ds_val = Subset(ds_val_all, val_idx)
    ds_test = MSASLSupervised(base_test, target_t=args.target_t, train=False, temporal_sampling=args.temporal_sampling)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_supervised)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_supervised)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_supervised)

    model = MASA(
        skeleton_representation="graph-based",
        num_class=args.num_class,
        pretrain=False,
        dropout=args.dropout,
    )
    if args.init_mode == "pretrained_dense":
        load_info = load_checkpoint_compatible(model, args.init_ckpt)
        low_rank_stats = apply_low_rank_from_dense(
            model,
            rank_ratio=args.rank_ratio,
            target_spec=args.low_rank_targets,
            min_features=args.low_rank_min_features,
        )
    else:
        low_rank_stats = apply_low_rank_structure(
            model,
            rank_ratio=args.rank_ratio,
            target_spec=args.low_rank_targets,
            min_features=args.low_rank_min_features,
        )
        checkpoint = torch.load(args.init_ckpt, map_location="cpu")
        validate_resume_configuration(checkpoint, args)
        load_low_rank_checkpoint_strict(model, args.init_ckpt)
        load_info = {"loaded": len(model.state_dict()), "skipped": 0}

    print(
        f"[init] mode={args.init_mode} loaded={load_info['loaded']} "
        f"skipped={load_info['skipped']}"
    )
    print(
        f"[low-rank] replaced={low_rank_stats['num_replaced']} "
        f"param_reduction={low_rank_stats['param_reduction_total']}"
    )

    model.to(device)

    if args.train_temporal_crop_min != 1.0:
        print("[warn] --train-temporal-crop-min is accepted but not used in this script.")
    if args.warmup_epochs > 0:
        print("[warn] --warmup-epochs is accepted but not used in this script.")

    head_params = []
    base_params = []
    for name, p in model.named_parameters():
        if "encoder_q.proj.fc" in name:
            head_params.append(p)
        else:
            base_params.append(p)
    param_groups = [
        {"params": base_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ]

    if args.optim == "sgd":
        optimizer = torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == "adam":
        optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    if args.scheduler == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_gamma)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pth.tar"
    metrics_path = out_dir / "metrics.json"

    best_val = -1.0
    best_epoch = -1
    history = []
    no_improve = 0

    if args.freeze_epochs > 0:
        for name, p in model.named_parameters():
            if "encoder_q.proj.fc" not in name:
                p.requires_grad = False

    for epoch in range(1, args.epochs + 1):
        if args.freeze_epochs > 0 and epoch == (args.freeze_epochs + 1):
            for p in model.parameters():
                p.requires_grad = True

        tr = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va = run_epoch(model, val_loader, criterion, optimizer, device, False)

        if scheduler is not None:
            scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr.loss,
                "train_acc": tr.acc,
                "val_loss": va.loss,
                "val_acc": va.acc,
                "lr": lr_now,
            }
        )
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if va.acc > (best_val + args.min_delta):
            best_val = va.acc
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_acc": best_val,
                    "init_ckpt": args.init_ckpt,
                    "init_mode": args.init_mode,
                    "rank_ratio": args.rank_ratio,
                    "low_rank_targets": args.low_rank_targets,
                    "low_rank_min_features": args.low_rank_min_features,
                    "low_rank_stats": low_rank_stats,
                },
                str(best_path),
            )
        else:
            no_improve += 1

        print(
            f"Epoch {epoch:02d} | "
            f"train_acc={tr.acc:.4f} | "
            f"val_acc={va.acc:.4f} | "
            f"lr={lr_now:.6g}"
        )
        if no_improve >= args.patience:
            print(f"[early-stop] no improvement for {args.patience} epochs.")
            break

    best = torch.load(best_path, map_location="cpu")
    model.load_state_dict(best["state_dict"], strict=True)
    model.to(device)
    te = run_epoch(model, test_loader, criterion, optimizer, device, False)
    summary = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val,
        "test_loss": te.loss,
        "test_acc": te.acc,
        "best_checkpoint": str(best_path),
        "init_ckpt": args.init_ckpt,
        "init_mode": args.init_mode,
        "rank_ratio": args.rank_ratio,
        "low_rank_targets": args.low_rank_targets,
        "low_rank_min_features": args.low_rank_min_features,
        "low_rank_stats": low_rank_stats,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[test @best_epoch={best_epoch}] acc={te.acc:.4f} loss={te.loss:.4f}")
    print(f"[done] summary saved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
