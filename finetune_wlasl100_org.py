#!/usr/bin/env python3
import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from feeder.single_dataset.WLASL import WLASL
from moco.builder_dist import MASA


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WLASLSupervised(Dataset):
    """Adapter from WLASL.get_sample(...) format to MASA finetune input."""

    def __init__(
        self,
        base_ds: Dataset,
        target_t: int,
        train: bool,
        train_temporal_crop_min: float,
    ):
        self.base_ds = base_ds
        self.target_t = target_t
        self.train = train
        self.train_temporal_crop_min = train_temporal_crop_min

    def __len__(self) -> int:
        return len(self.base_ds)

    @staticmethod
    def _resample_time(x: torch.Tensor, target_t: int, mode: str) -> torch.Tensor:
        # x: [T, ...]
        t = x.shape[0]
        if t == target_t:
            return x
        if t <= 1:
            return x.repeat(target_t, *([1] * (x.dim() - 1)))
        flat = x.reshape(t, -1).transpose(0, 1).unsqueeze(0)  # [1, F, T]
        y = F.interpolate(flat, size=target_t, mode=mode)
        y = y.squeeze(0).transpose(0, 1).reshape(target_t, *x.shape[1:])
        return y

    def _random_crop_bounds(self, t: int) -> Tuple[int, int]:
        if t <= 1:
            return 0, t
        if self.train_temporal_crop_min >= 1.0:
            return 0, t
        min_len = max(1, int(round(t * self.train_temporal_crop_min)))
        crop_len = random.randint(min_len, t)
        start = random.randint(0, t - crop_len)
        return start, start + crop_len

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.base_ds.get_sample(index)
        right = sample["right"]
        left = sample["left"]
        body = sample["body"]

        rh = right["kp2d"].float()                 # [T, 21, 2]
        lh = left["kp2d"].float()                  # [T, 21, 2]
        bd = body["body_pose"].float()             # [T, 7, 2]
        rm = right["mask"].float()                 # [T, 21, 2]
        lm = left["mask"].float()                  # [T, 21, 2]

        if self.train:
            s, e = self._random_crop_bounds(rh.shape[0])
            rh, lh, bd, rm, lm = rh[s:e], lh[s:e], bd[s:e], rm[s:e], lm[s:e]

        rh = self._resample_time(rh, self.target_t, mode="linear")
        lh = self._resample_time(lh, self.target_t, mode="linear")
        bd = self._resample_time(bd, self.target_t, mode="linear")
        rm = self._resample_time(rm, self.target_t, mode="nearest")
        lm = self._resample_time(lm, self.target_t, mode="nearest")
        mask = torch.cat([rm, lm], dim=0).float()  # [2T, 21, 2]

        label = int(right["label"])

        return {
            "rh": rh,
            "lh": lh,
            "body": bd,
            "mask": mask,
            "label": label,
        }


def collate_supervised(batch: List[Dict[str, torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    batch_size = len(batch)
    max_t = max(item["rh"].shape[0] for item in batch)

    rh = torch.zeros((batch_size, max_t, 21, 2), dtype=torch.float32)
    lh = torch.zeros((batch_size, max_t, 21, 2), dtype=torch.float32)
    bd = torch.zeros((batch_size, max_t, 7, 2), dtype=torch.float32)
    mk = torch.zeros((batch_size, 2 * max_t, 21, 2), dtype=torch.float32)
    y = torch.zeros((batch_size,), dtype=torch.long)

    for i, item in enumerate(batch):
        t = item["rh"].shape[0]
        rh[i, :t] = item["rh"]
        lh[i, :t] = item["lh"]
        bd[i, :t] = item["body"]
        mk[i, : 2 * t] = item["mask"]
        y[i] = int(item["label"])

    x = {"rh": rh, "lh": lh, "body": bd, "mask": mk}
    return x, y


def strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        out[k[len(prefix):] if k.startswith(prefix) else k] = v
    return out


def load_pretrained_strict_false(model: nn.Module, ckpt_path: str) -> None:
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


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    # Keep the final classifier trainable during warmup.
    for name, p in model.named_parameters():
        if "encoder_q.proj.fc" in name:
            p.requires_grad = True
        else:
            p.requires_grad = trainable


def build_optimizer(
    model: nn.Module,
    optim_name: str,
    base_lr: float,
    weight_decay: float,
    head_lr_mult: float,
    momentum: float,
) -> torch.optim.Optimizer:
    head_params = []
    body_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "encoder_q.proj.fc" in name:
            head_params.append(p)
        else:
            body_params.append(p)

    param_groups = []
    if body_params:
        param_groups.append({"params": body_params, "lr": base_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": base_lr * head_lr_mult})

    if optim_name.lower() == "sgd":
        return torch.optim.SGD(param_groups, momentum=momentum, weight_decay=weight_decay)
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    warmup_epochs: int,
    milestones: List[int],
    gamma: float,
    warmup_start_factor: float,
):
    scheduler_name = scheduler_name.lower()
    main_scheduler = None
    remain_epochs = max(1, epochs - warmup_epochs)

    if scheduler_name == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remain_epochs)
    elif scheduler_name == "multistep":
        m = [x for x in milestones if x > warmup_epochs]
        m = [x - warmup_epochs for x in m]
        if not m:
            m = [max(1, remain_epochs // 2)]
        main_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=m, gamma=gamma)
    elif scheduler_name == "none":
        main_scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    if warmup_epochs <= 0:
        return main_scheduler

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    if main_scheduler is None:
        return warmup
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, main_scheduler],
        milestones=[warmup_epochs],
    )


@dataclass
class EvalResult:
    loss: float
    acc: float


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
) -> EvalResult:
    model.train(mode=train)
    total_loss = 0.0
    total = 0
    correct = 0

    pbar = tqdm(loader, ncols=100, leave=False)
    for x, y in pbar:
        x = {k: v.to(device, non_blocking=(device.type == "cuda")) for k, v in x.items()}
        y = y.to(device, non_blocking=(device.type == "cuda"))

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch = y.size(0)
        total += batch
        total_loss += float(loss.item()) * batch
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())

        pbar.set_description(f"{'train' if train else 'eval'} loss={loss.item():.4f}")

    return EvalResult(loss=total_loss / max(total, 1), acc=correct / max(total, 1))


def build_dataloaders(
    data_root: str,
    subset_num: int,
    batch_size: int,
    workers: int,
    target_t: int,
    train_temporal_crop_min: float,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Train split in this loader includes train+val; self.flag stores split boundary.
    base_trainval = WLASL(data_root=data_root, data_split="train", subset_num=subset_num, use_cache=False)
    n_train = int(base_trainval.flag)
    n_total = len(base_trainval)
    train_idx = list(range(0, n_train))
    val_idx = list(range(n_train, n_total))

    ds_train_all = WLASLSupervised(
        base_trainval,
        target_t=target_t,
        train=True,
        train_temporal_crop_min=train_temporal_crop_min,
    )
    ds_train = Subset(ds_train_all, train_idx)
    ds_val = Subset(ds_train_all, val_idx)

    base_test = WLASL(data_root=data_root, data_split="test", subset_num=subset_num, use_cache=False)
    ds_test = WLASLSupervised(
        base_test,
        target_t=target_t,
        train=False,
        train_temporal_crop_min=1.0,
    )

    kwargs = dict(batch_size=batch_size, num_workers=workers, pin_memory=True, collate_fn=collate_supervised)
    train_loader = DataLoader(ds_train, shuffle=True, drop_last=False, **kwargs)
    val_loader = DataLoader(ds_val, shuffle=False, drop_last=False, **kwargs)
    test_loader = DataLoader(ds_test, shuffle=False, drop_last=False, **kwargs)
    return train_loader, val_loader, test_loader


def save_checkpoint(path: Path, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, best_val_acc: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
        },
        str(path),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune MASA on WLASL100 and save best checkpoint.")
    p.add_argument("--data-root", required=True, help="Prepared MASA dataset root (contains jpg_video_ori, Keypoints_2d_mmpose, traintestlist)")
    p.add_argument("--pretrained", required=True, help="Path to pretrained_model.pth.tar")
    p.add_argument("--out-dir", default="./checkpoints_finetune_wlasl100")
    p.add_argument("--subset-num", type=int, default=100)
    p.add_argument("--num-class", type=int, default=100)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--optim", type=str, default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "multistep"])
    p.add_argument("--milestones", nargs="*", type=int, default=[30, 50, 70], help="Used by multistep scheduler")
    p.add_argument("--lr-gamma", type=float, default=0.1, help="Gamma for multistep scheduler")
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--warmup-start-factor", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--head-lr-mult", type=float, default=5.0)
    p.add_argument("--freeze-epochs", type=int, default=5, help="Train classifier head only for first N epochs")
    p.add_argument("--target-t", type=int, default=64, help="Fixed temporal length after resampling")
    p.add_argument("--train-temporal-crop-min", type=float, default=0.6, help="Train random temporal crop min ratio before resampling")
    p.add_argument("--patience", type=int, default=12, help="Early stop after N epochs without val improvement")
    p.add_argument("--min-delta", type=float, default=1e-4, help="Minimum val acc improvement to reset patience")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    train_loader, val_loader, test_loader = build_dataloaders(
        data_root=args.data_root,
        subset_num=args.subset_num,
        batch_size=args.batch_size,
        workers=args.workers,
        target_t=args.target_t,
        train_temporal_crop_min=args.train_temporal_crop_min,
    )
    print(f"[data] train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    model = MASA(
        skeleton_representation="graph-based",
        num_class=args.num_class,
        pretrain=False,
        dropout=args.dropout,
    )
    model.to(device)
    load_pretrained_strict_false(model, args.pretrained)

    criterion = nn.CrossEntropyLoss()
    set_backbone_trainable(model, trainable=False)
    optimizer = build_optimizer(
        model=model,
        optim_name=args.optim,
        base_lr=args.lr,
        weight_decay=args.weight_decay,
        head_lr_mult=args.head_lr_mult,
        momentum=args.momentum,
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=args.scheduler,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        milestones=args.milestones,
        gamma=args.lr_gamma,
        warmup_start_factor=args.warmup_start_factor,
    )
    print(f"[warmup] head-only training for first {args.freeze_epochs} epochs")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / "best.pth.tar"
    last_ckpt = out_dir / "last.pth.tar"
    metrics_path = out_dir / "metrics.json"

    history: List[Dict[str, float]] = []
    best_val_acc = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n[epoch {epoch}/{args.epochs}]")

        if epoch == args.freeze_epochs + 1:
            set_backbone_trainable(model, trainable=True)
            optimizer = build_optimizer(
                model=model,
                optim_name=args.optim,
                base_lr=args.lr,
                weight_decay=args.weight_decay,
                head_lr_mult=args.head_lr_mult,
                momentum=args.momentum,
            )
            scheduler = build_scheduler(
                optimizer=optimizer,
                scheduler_name=args.scheduler,
                epochs=args.epochs - epoch + 1,
                warmup_epochs=0,
                milestones=args.milestones,
                gamma=args.lr_gamma,
                warmup_start_factor=args.warmup_start_factor,
            )
            print("[unfreeze] backbone parameters are now trainable")

        tr = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"train: loss={tr.loss:.4f} acc={tr.acc:.4f} | val: loss={va.loss:.4f} acc={va.acc:.4f} | lr={current_lr:.6g}")

        history.append(
            {
                "epoch": epoch,
                "train_loss": tr.loss,
                "train_acc": tr.acc,
                "val_loss": va.loss,
                "val_acc": va.acc,
            }
        )
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        improved = va.acc > (best_val_acc + args.min_delta)
        if improved:
            best_val_acc = va.acc
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(best_ckpt, epoch, model, optimizer, best_val_acc)
            print(f"[best] epoch={epoch} val_acc={best_val_acc:.4f} saved={best_ckpt}")
        else:
            epochs_no_improve += 1
            print(f"[plateau] no_improve={epochs_no_improve}/{args.patience}")

        save_checkpoint(last_ckpt, epoch, model, optimizer, best_val_acc)
        if scheduler is not None:
            scheduler.step()

        if epochs_no_improve >= args.patience:
            print(f"[early-stop] validation accuracy stabilized at epoch {epoch}.")
            break

    # Final test with best model.
    best = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(best["state_dict"], strict=True)
    model.to(device)
    te = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"\n[test @best_epoch={best_epoch}] loss={te.loss:.4f} acc={te.acc:.4f}")

    summary = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": te.loss,
        "test_acc": te.acc,
        "best_checkpoint": str(best_ckpt),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] summary saved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
