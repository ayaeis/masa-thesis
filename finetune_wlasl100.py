#!/usr/bin/env python3
import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from feeder.single_dataset.WLASL import WLASL
from moco.builder_dist import MASA


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class WLASLSupervised(Dataset):
    """
    Adapter from WLASL.get_sample(...) format to MASA fine-tune input.
    Implements:
        - random index sampling (train)
        - center index sampling (test)
        - optional interpolation fallback
    """

    def __init__(
        self,
        base_ds: Dataset,
        target_t: int,
        train: bool,
        temporal_sampling: str,
    ):
        self.base_ds = base_ds
        self.target_t = target_t
        self.train = train
        self.temporal_sampling = temporal_sampling

    def __len__(self):
        return len(self.base_ds)

    # ------------------------------------------------------------
    # Index Sampling (paper-aligned)
    # ------------------------------------------------------------

    def _sample_indices(self, T: int) -> torch.Tensor:
        target = self.target_t

        if T == target:
            return torch.arange(T, dtype=torch.long)

        if T < target:
            # Repeat by interpolation on indices (no synthetic feature interpolation).
            return torch.linspace(0, T - 1, steps=target).round().long()

        # T > target
        if self.train:
            # random uniform segment sampling
            segments = torch.linspace(0, T, steps=target + 1)
            indices = []
            for i in range(target):
                start = int(segments[i].item())
                end = int(segments[i + 1].item())
                if start < end:
                    indices.append(torch.randint(start, end, (1,)))
                else:
                    indices.append(torch.tensor([start]))
            return torch.cat(indices).long()
        else:
            # center deterministic segment sampling
            edges = torch.linspace(0, T, steps=target + 1)
            centers = ((edges[:-1] + edges[1:]) * 0.5).long()
            return torch.clamp(centers, 0, T - 1)

    # ------------------------------------------------------------
    # Interpolation fallback (optional)
    # ------------------------------------------------------------

    @staticmethod
    def _resample_time(x: torch.Tensor, target_t: int, mode: str) -> torch.Tensor:
        t = x.shape[0]
        if t == target_t:
            return x
        if t <= 1:
            return x.repeat(target_t, *([1] * (x.dim() - 1)))

        flat = x.reshape(t, -1).transpose(0, 1).unsqueeze(0)
        y = F.interpolate(flat, size=target_t, mode=mode)
        y = y.squeeze(0).transpose(0, 1).reshape(target_t, *x.shape[1:])
        return y

    # ------------------------------------------------------------
    # Get item
    # ------------------------------------------------------------

    def __getitem__(self, index: int):
        sample = self.base_ds.get_sample(index)

        rh = sample["right"]["kp2d"].float()
        lh = sample["left"]["kp2d"].float()
        bd = sample["body"]["body_pose"].float()
        rm = sample["right"]["mask"].float()
        lm = sample["left"]["mask"].float()

        T = rh.shape[0]

        if self.temporal_sampling == "interpolate":
            rh = self._resample_time(rh, self.target_t, mode="linear")
            lh = self._resample_time(lh, self.target_t, mode="linear")
            bd = self._resample_time(bd, self.target_t, mode="linear")
            rm = self._resample_time(rm, self.target_t, mode="nearest")
            lm = self._resample_time(lm, self.target_t, mode="nearest")
        else:
            idx = self._sample_indices(T)
            rh = rh[idx]
            lh = lh[idx]
            bd = bd[idx]
            rm = rm[idx]
            lm = lm[idx]

        mask = torch.cat([rm, lm], dim=0)
        label = int(sample["right"]["label"])

        return {
            "rh": rh,
            "lh": lh,
            "body": bd,
            "mask": mask,
            "label": label,
        }


# ------------------------------------------------------------
# Collate
# ------------------------------------------------------------

def collate_supervised(batch):
    batch_size = len(batch)
    T = batch[0]["rh"].shape[0]

    rh = torch.stack([b["rh"] for b in batch])
    lh = torch.stack([b["lh"] for b in batch])
    bd = torch.stack([b["body"] for b in batch])
    mk = torch.stack([b["mask"] for b in batch])
    y = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    return {"rh": rh, "lh": lh, "body": bd, "mask": mk}, y


# ------------------------------------------------------------
# Pretrained Loading
# ------------------------------------------------------------

def strip_prefix_if_present(state_dict, prefix):
    out = {}
    for k, v in state_dict.items():
        out[k[len(prefix):] if k.startswith(prefix) else k] = v
    return out


def load_pretrained_strict_false(model, ckpt_path):
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


# ------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------

@dataclass
class EvalResult:
    loss: float
    acc: float


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(mode=train)
    total_loss = 0.0
    total = 0
    correct = 0

    for x, y in loader:
        x = {k: v.to(device) for k, v in x.items()}
        y = y.to(device)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total += y.size(0)
        total_loss += loss.item() * y.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()

    return EvalResult(total_loss / total, correct / total)


# ------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--pretrained", required=True)
    p.add_argument("--subset-num", type=int, default=100)
    p.add_argument("--num-class", type=int, default=100)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--target-t", type=int, default=32)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-dir", type=str, default="./checkpoints_finetune_wlasl100_index")
    p.add_argument(
        "--temporal-sampling",
        type=str,
        default="index",
        choices=["index", "interpolate"],
    )
    return p.parse_args()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_train = WLASL(args.data_root, data_split="train",
                       subset_num=args.subset_num, use_cache=False)
    base_test = WLASL(args.data_root, data_split="test",
                      subset_num=args.subset_num, use_cache=False)

    # WLASL train loader includes train+val with boundary at base_train.flag
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

    model = MASA(
        skeleton_representation="graph-based",
        num_class=args.num_class,
        pretrain=False,
    )

    model.to(device)
    load_pretrained_strict_false(model, args.pretrained)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[20, 40],
        gamma=0.1,
    )

    criterion = nn.CrossEntropyLoss()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pth.tar"
    metrics_path = out_dir / "metrics.json"

    best_val = -1.0
    best_epoch = -1
    history = []

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, criterion,
                       optimizer, device, True)
        va = run_epoch(model, val_loader, criterion,
                       optimizer, device, False)

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

        if va.acc > best_val:
            best_val = va.acc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_acc": best_val,
                },
                str(best_path),
            )

        print(
            f"Epoch {epoch:02d} | "
            f"train_acc={tr.acc:.4f} | "
            f"val_acc={va.acc:.4f} | "
            f"lr={lr_now:.6g}"
        )

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
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[test @best_epoch={best_epoch}] acc={te.acc:.4f} loss={te.loss:.4f}")
    print(f"[done] summary saved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
