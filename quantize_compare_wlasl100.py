#!/usr/bin/env python3
import argparse
import copy
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from feeder.single_dataset.WLASL import WLASL
from moco.builder_dist import MASA


class WLASLSupervisedEval(Dataset):
    """Evaluation-only adapter: deterministic temporal resampling to target_t."""

    def __init__(self, base_ds: Dataset, target_t: int):
        self.base_ds = base_ds
        self.target_t = target_t

    def __len__(self) -> int:
        return len(self.base_ds)

    @staticmethod
    def _resample_time(x: torch.Tensor, target_t: int, mode: str) -> torch.Tensor:
        t = x.shape[0]
        if t == target_t:
            return x
        if t <= 1:
            return x.repeat(target_t, *([1] * (x.dim() - 1)))
        flat = x.reshape(t, -1).transpose(0, 1).unsqueeze(0)  # [1, F, T]
        if mode == "linear":
            y = F.interpolate(flat, size=target_t, mode=mode, align_corners=False)
        else:
            y = F.interpolate(flat, size=target_t, mode=mode)
        y = y.squeeze(0).transpose(0, 1).reshape(target_t, *x.shape[1:])
        return y

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.base_ds.get_sample(index)
        right = sample["right"]
        left = sample["left"]
        body = sample["body"]

        rh = right["kp2d"].float()
        lh = left["kp2d"].float()
        bd = body["body_pose"].float()
        rm = right["mask"].float()
        lm = left["mask"].float()
        label = int(right["label"])

        rh = self._resample_time(rh, self.target_t, mode="linear")
        lh = self._resample_time(lh, self.target_t, mode="linear")
        bd = self._resample_time(bd, self.target_t, mode="linear")
        rm = self._resample_time(rm, self.target_t, mode="nearest")
        lm = self._resample_time(lm, self.target_t, mode="nearest")
        mask = torch.cat([rm, lm], dim=0).float()  # [2T, 21, 2]

        return {"rh": rh, "lh": lh, "body": bd, "mask": mask, "label": label}


def collate_supervised(batch: List[Dict[str, torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    batch_size = len(batch)
    t = batch[0]["rh"].shape[0]  # fixed by resampling

    rh = torch.zeros((batch_size, t, 21, 2), dtype=torch.float32)
    lh = torch.zeros((batch_size, t, 21, 2), dtype=torch.float32)
    bd = torch.zeros((batch_size, t, 7, 2), dtype=torch.float32)
    mk = torch.zeros((batch_size, 2 * t, 21, 2), dtype=torch.float32)
    y = torch.zeros((batch_size,), dtype=torch.long)

    for i, item in enumerate(batch):
        rh[i] = item["rh"]
        lh[i] = item["lh"]
        bd[i] = item["body"]
        mk[i] = item["mask"]
        y[i] = int(item["label"])
    return {"rh": rh, "lh": lh, "body": bd, "mask": mk}, y


def load_checkpoint_state_dict(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    # Handle possible DDP checkpoints.
    out = {}
    for k, v in sd.items():
        out[k[7:] if k.startswith("module.") else k] = v
    return out


def load_model(model: nn.Module, checkpoint_path: str) -> None:
    sd = load_checkpoint_state_dict(checkpoint_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")


def model_size_mb(model: nn.Module, path: Path) -> float:
    torch.save(model.state_dict(), str(path))
    return path.stat().st_size / (1024 * 1024)


@dataclass
class EvalResult:
    loss: float
    acc: float
    latency_ms: float


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> EvalResult:
    model.eval()
    total = 0
    total_loss = 0.0
    correct = 0
    step_times: List[float] = []

    with torch.no_grad():
        for x, y in tqdm(loader, ncols=100, leave=False):
            x = {k: v.to(device) for k, v in x.items()}
            y = y.to(device)
            t0 = time.perf_counter()
            logits = model(x)
            t1 = time.perf_counter()
            step_times.append((t1 - t0) * 1000.0)

            loss = criterion(logits, y)
            bs = y.size(0)
            total += bs
            total_loss += float(loss.item()) * bs
            correct += int((torch.argmax(logits, dim=1) == y).sum().item())

    return EvalResult(
        loss=total_loss / max(total, 1),
        acc=correct / max(total, 1),
        latency_ms=float(np.mean(step_times)) if step_times else 0.0,
    )


def count_linear_modules(model: nn.Module) -> Tuple[int, int]:
    fp_linear = 0
    int8_linear = 0
    for m in model.modules():
        name = m.__class__.__name__.lower()
        if isinstance(m, nn.Linear):
            fp_linear += 1
        # torch.nn.quantized.dynamic.Linear class name variants across torch versions.
        if "dynamic" in name and "linear" in name:
            int8_linear += 1
    return fp_linear, int8_linear


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate FP32 vs Dynamic INT8 quantization on WLASL100 test set.")
    p.add_argument("--data-root", required=True, help="Prepared MASA dataset root")
    p.add_argument("--checkpoint", required=True, help="Fine-tuned checkpoint (best.pth.tar)")
    p.add_argument("--num-class", type=int, default=100)
    p.add_argument("--subset-num", type=int, default=100)
    p.add_argument("--target-t", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-dir", default="./quant_compare_out")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dynamic quantization is CPU-oriented.
    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss()

    base_test = WLASL(
        data_root=args.data_root,
        data_split="test",
        subset_num=args.subset_num,
        use_cache=False,
    )
    ds_test = WLASLSupervisedEval(base_test, target_t=args.target_t)
    test_loader = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        collate_fn=collate_supervised,
    )
    print(f"[data] test={len(ds_test)}")

    # FP32 baseline.
    fp32_model = MASA(
        skeleton_representation="graph-based",
        num_class=args.num_class,
        pretrain=False,
    ).to(device)
    load_model(fp32_model, args.checkpoint)
    fp_lin, fp_dyn = count_linear_modules(fp32_model)
    print(f"[fp32] nn.Linear={fp_lin}, dynamic_linear={fp_dyn}")

    fp32_res = evaluate(fp32_model, test_loader, device, criterion)
    fp32_size = model_size_mb(fp32_model, out_dir / "fp32_state_dict.pth")

    # Dynamic INT8: quantize only nn.Linear.
    int8_model = copy.deepcopy(fp32_model).cpu().eval()
    int8_model = torch.quantization.quantize_dynamic(
        int8_model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    int_lin, int_dyn = count_linear_modules(int8_model)
    print(f"[int8] nn.Linear={int_lin}, dynamic_linear={int_dyn} (only Linear modules quantized)")

    int8_res = evaluate(int8_model, test_loader, device, criterion)
    int8_size = model_size_mb(int8_model, out_dir / "int8_dynamic_state_dict.pth")

    summary = {
        "fp32": {
            "loss": fp32_res.loss,
            "acc": fp32_res.acc,
            "latency_ms_per_batch": fp32_res.latency_ms,
            "size_mb_state_dict": fp32_size,
            "linear_modules": fp_lin,
        },
        "int8_dynamic": {
            "loss": int8_res.loss,
            "acc": int8_res.acc,
            "latency_ms_per_batch": int8_res.latency_ms,
            "size_mb_state_dict": int8_size,
            "dynamic_linear_modules": int_dyn,
        },
        "delta": {
            "acc_drop": fp32_res.acc - int8_res.acc,
            "size_reduction_mb": fp32_size - int8_size,
            "speedup_x": (fp32_res.latency_ms / int8_res.latency_ms) if int8_res.latency_ms > 0 else None,
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== FP32 ===")
    print(f"loss={fp32_res.loss:.4f} acc={fp32_res.acc:.4f} latency_ms/batch={fp32_res.latency_ms:.2f} size_mb={fp32_size:.2f}")
    print("=== INT8 Dynamic (Linear only) ===")
    print(f"loss={int8_res.loss:.4f} acc={int8_res.acc:.4f} latency_ms/batch={int8_res.latency_ms:.2f} size_mb={int8_size:.2f}")
    print("=== Delta ===")
    print(f"acc_drop={summary['delta']['acc_drop']:.4f} size_reduction_mb={summary['delta']['size_reduction_mb']:.2f} speedup_x={summary['delta']['speedup_x']:.2f}")
    print(f"[done] {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
