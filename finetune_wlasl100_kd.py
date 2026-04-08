#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from feeder.single_dataset.WLASL import WLASL
from finetune_wlasl100 import WLASLSupervised, collate_supervised, set_seed
from moco.builder_dist import MASA


def strip_prefix_if_present(state_dict, prefix):
    out = {}
    for k, v in state_dict.items():
        out[k[len(prefix):] if k.startswith(prefix) else k] = v
    return out


def load_checkpoint_strict_false(model, ckpt_path):
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
    return {"loaded": len(loadable), "skipped": skipped}


def build_model(num_class, dropout, use_ghost_conv, ghost_ratio, ghost_mode):
    return MASA(
        skeleton_representation="graph-based",
        num_class=num_class,
        pretrain=False,
        dropout=dropout,
        use_ghost_conv=use_ghost_conv,
        ghost_ratio=ghost_ratio,
        ghost_mode=ghost_mode,
    )


@dataclass
class EpochResult:
    total_loss: float
    ce_loss: float
    kd_loss: float
    acc: float


def kd_loss_fn(student_logits, teacher_logits, temperature):
    t = temperature
    return F.kl_div(
        F.log_softmax(student_logits / t, dim=1),
        F.softmax(teacher_logits / t, dim=1),
        reduction="batchmean",
    ) * (t * t)


def run_train_epoch(student, teacher, loader, criterion, optimizer, device, kd_alpha, kd_temp):
    student.train()
    teacher.eval()

    total_total_loss = 0.0
    total_ce_loss = 0.0
    total_kd_loss = 0.0
    total = 0
    correct = 0

    for x, y in loader:
        x = {k: v.to(device) for k, v in x.items()}
        y = y.to(device)

        with torch.no_grad():
            teacher_logits = teacher(x)

        student_logits = student(x)
        ce = criterion(student_logits, y)
        kd = kd_loss_fn(student_logits, teacher_logits, kd_temp)
        loss = (1.0 - kd_alpha) * ce + kd_alpha * kd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total += bs
        total_total_loss += loss.item() * bs
        total_ce_loss += ce.item() * bs
        total_kd_loss += kd.item() * bs
        pred = torch.argmax(student_logits, dim=1)
        correct += (pred == y).sum().item()

    return EpochResult(
        total_loss=total_total_loss / total,
        ce_loss=total_ce_loss / total,
        kd_loss=total_kd_loss / total,
        acc=correct / total,
    )


def run_eval_epoch(student, loader, criterion, device):
    student.eval()

    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for x, y in loader:
            x = {k: v.to(device) for k, v in x.items()}
            y = y.to(device)
            logits = student(x)
            loss = criterion(logits, y)

            bs = y.size(0)
            total += bs
            total_loss += loss.item() * bs
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()

    return total_loss / total, correct / total


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune a Ghost MASA student with knowledge distillation from a baseline teacher.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--teacher-ckpt", required=True)
    p.add_argument("--student-ckpt", required=True)
    p.add_argument("--subset-num", type=int, default=100)
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
    p.add_argument("--kd-alpha", type=float, default=0.5)
    p.add_argument("--kd-temp", type=float, default=4.0)
    p.add_argument("--teacher-use-ghost-conv", action="store_true")
    p.add_argument("--teacher-ghost-ratio", type=int, default=2)
    p.add_argument("--teacher-ghost-mode", type=str, default="all", choices=["kernel1", "all", "gt1"])
    p.add_argument("--student-use-ghost-conv", action="store_true")
    p.add_argument("--student-ghost-ratio", type=int, default=2)
    p.add_argument("--student-ghost-mode", type=str, default="all", choices=["kernel1", "all", "gt1"])
    p.add_argument("--out-dir", type=str, default="./checkpoints_finetune_wlasl100_kd")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_train = WLASL(args.data_root, data_split="train", subset_num=args.subset_num, use_cache=False)
    base_test = WLASL(args.data_root, data_split="test", subset_num=args.subset_num, use_cache=False)

    n_train = int(base_train.flag)
    n_total = len(base_train)
    train_idx = list(range(0, n_train))
    val_idx = list(range(n_train, n_total))

    ds_train_all = WLASLSupervised(
        base_train,
        target_t=args.target_t,
        train=True,
        temporal_sampling=args.temporal_sampling,
    )
    ds_val_all = WLASLSupervised(
        base_train,
        target_t=args.target_t,
        train=False,
        temporal_sampling=args.temporal_sampling,
    )
    ds_train = Subset(ds_train_all, train_idx)
    ds_val = Subset(ds_val_all, val_idx)
    ds_test = WLASLSupervised(
        base_test,
        target_t=args.target_t,
        train=False,
        temporal_sampling=args.temporal_sampling,
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_supervised,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_supervised,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_supervised,
    )

    teacher = build_model(
        args.num_class,
        args.dropout,
        args.teacher_use_ghost_conv,
        args.teacher_ghost_ratio,
        args.teacher_ghost_mode,
    )
    student = build_model(
        args.num_class,
        args.dropout,
        args.student_use_ghost_conv,
        args.student_ghost_ratio,
        args.student_ghost_mode,
    )

    teacher_load = load_checkpoint_strict_false(teacher, args.teacher_ckpt)
    student_load = load_checkpoint_strict_false(student, args.student_ckpt)
    print(f"[teacher] loaded={teacher_load['loaded']} skipped={teacher_load['skipped']}")
    print(f"[student] loaded={student_load['loaded']} skipped={student_load['skipped']}")

    teacher.to(device)
    student.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    if args.train_temporal_crop_min != 1.0:
        print("[warn] --train-temporal-crop-min is accepted but not used in this script.")
    if args.warmup_epochs > 0:
        print("[warn] --warmup-epochs is accepted but not used in this script.")

    head_params = []
    base_params = []
    for name, p in student.named_parameters():
        if "encoder_q.proj.fc" in name:
            head_params.append(p)
        else:
            base_params.append(p)
    param_groups = [
        {"params": base_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ]

    if args.optim == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=args.weight_decay,
        )

    if args.scheduler == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args.milestones,
            gamma=args.lr_gamma,
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs),
        )
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
        for name, p in student.named_parameters():
            if "encoder_q.proj.fc" not in name:
                p.requires_grad = False

    for epoch in range(1, args.epochs + 1):
        if args.freeze_epochs > 0 and epoch == (args.freeze_epochs + 1):
            for p in student.parameters():
                p.requires_grad = True

        tr = run_train_epoch(student, teacher, train_loader, criterion, optimizer, device, args.kd_alpha, args.kd_temp)
        val_loss, val_acc = run_eval_epoch(student, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_total_loss": tr.total_loss,
                "train_ce_loss": tr.ce_loss,
                "train_kd_loss": tr.kd_loss,
                "train_acc": tr.acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": lr_now,
            }
        )
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if val_acc > (best_val + args.min_delta):
            best_val = val_acc
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_acc": best_val,
                    "kd_alpha": args.kd_alpha,
                    "kd_temp": args.kd_temp,
                    "teacher_ckpt": args.teacher_ckpt,
                    "student_init_ckpt": args.student_ckpt,
                },
                str(best_path),
            )
        else:
            no_improve += 1

        print(
            f"Epoch {epoch:02d} | "
            f"train_acc={tr.acc:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"ce={tr.ce_loss:.4f} | "
            f"kd={tr.kd_loss:.4f} | "
            f"lr={lr_now:.6g}"
        )
        if no_improve >= args.patience:
            print(f"[early-stop] no improvement for {args.patience} epochs.")
            break

    best = torch.load(best_path, map_location="cpu")
    student.load_state_dict(best["state_dict"], strict=True)
    student.to(device)
    test_loss, test_acc = run_eval_epoch(student, test_loader, criterion, device)
    summary = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "best_checkpoint": str(best_path),
        "teacher_ckpt": args.teacher_ckpt,
        "student_init_ckpt": args.student_ckpt,
        "kd_alpha": args.kd_alpha,
        "kd_temp": args.kd_temp,
        "teacher_use_ghost_conv": args.teacher_use_ghost_conv,
        "teacher_ghost_ratio": args.teacher_ghost_ratio,
        "teacher_ghost_mode": args.teacher_ghost_mode,
        "student_use_ghost_conv": args.student_use_ghost_conv,
        "student_ghost_ratio": args.student_ghost_ratio,
        "student_ghost_mode": args.student_ghost_mode,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[test @best_epoch={best_epoch}] acc={test_acc:.4f} loss={test_loss:.4f}")
    print(f"[done] summary saved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
