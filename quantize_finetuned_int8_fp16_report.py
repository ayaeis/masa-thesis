#!/usr/bin/env python3
import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from feeder.single_dataset.WLASL import WLASL
from moco.builder_dist import MASA


class Int8Linear(nn.Module):
    def __init__(self, mod: nn.Linear):
        super().__init__()
        w = mod.weight.detach().cpu()
        w_max = w.abs().max()
        scale = float(w_max / 127.0) if w_max > 0 else 1.0
        qweight = torch.clamp((w / scale).round(), -128, 127).to(torch.int8)
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.bias = nn.Parameter(mod.bias.detach().clone(), requires_grad=False) if mod.bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = (self.qweight.float() * self.scale.float()).to(dtype=x.dtype, device=x.device)
        b = self.bias.to(dtype=x.dtype, device=x.device) if self.bias is not None else None
        return F.linear(x, w, b)


class Int8Conv1d(nn.Module):
    def __init__(self, mod: nn.Conv1d):
        super().__init__()
        w = mod.weight.detach().cpu()
        w_max = w.abs().max()
        scale = float(w_max / 127.0) if w_max > 0 else 1.0
        qweight = torch.clamp((w / scale).round(), -128, 127).to(torch.int8)
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.bias = nn.Parameter(mod.bias.detach().clone(), requires_grad=False) if mod.bias is not None else None
        self.stride = mod.stride
        self.padding = mod.padding
        self.dilation = mod.dilation
        self.groups = mod.groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = (self.qweight.float() * self.scale.float()).to(dtype=x.dtype, device=x.device)
        b = self.bias.to(dtype=x.dtype, device=x.device) if self.bias is not None else None
        return F.conv1d(
            x,
            w,
            b,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class Int8Conv2d(nn.Module):
    def __init__(self, mod: nn.Conv2d):
        super().__init__()
        w = mod.weight.detach().cpu()
        w_max = w.abs().max()
        scale = float(w_max / 127.0) if w_max > 0 else 1.0
        qweight = torch.clamp((w / scale).round(), -128, 127).to(torch.int8)
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.bias = nn.Parameter(mod.bias.detach().clone(), requires_grad=False) if mod.bias is not None else None
        self.stride = mod.stride
        self.padding = mod.padding
        self.dilation = mod.dilation
        self.groups = mod.groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = (self.qweight.float() * self.scale.float()).to(dtype=x.dtype, device=x.device)
        b = self.bias.to(dtype=x.dtype, device=x.device) if self.bias is not None else None
        return F.conv2d(
            x,
            w,
            b,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


def replace_int8_supported_layers(module: nn.Module) -> Dict[str, int]:
    stats = {"linear": 0, "conv1d": 0, "conv2d": 0}

    def _walk(m: nn.Module) -> None:
        for name, child in list(m.named_children()):
            if isinstance(child, nn.Linear):
                setattr(m, name, Int8Linear(child))
                stats["linear"] += 1
            elif isinstance(child, nn.Conv1d):
                setattr(m, name, Int8Conv1d(child))
                stats["conv1d"] += 1
            elif isinstance(child, nn.Conv2d):
                setattr(m, name, Int8Conv2d(child))
                stats["conv2d"] += 1
            else:
                _walk(child)

    _walk(module)
    return stats


class WLASLSupervisedEval(Dataset):
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
        flat = x.reshape(t, -1).transpose(0, 1).unsqueeze(0)
        if mode == "linear":
            y = F.interpolate(flat, size=target_t, mode="linear", align_corners=False)
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
        y = int(right["label"])

        rh = self._resample_time(rh, self.target_t, mode="linear")
        lh = self._resample_time(lh, self.target_t, mode="linear")
        bd = self._resample_time(bd, self.target_t, mode="linear")
        rm = self._resample_time(rm, self.target_t, mode="nearest")
        lm = self._resample_time(lm, self.target_t, mode="nearest")
        mask = torch.cat([rm, lm], dim=0).float()

        return {"rh": rh, "lh": lh, "body": bd, "mask": mask, "label": y}


def collate_supervised(batch: List[Dict[str, torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    bs = len(batch)
    t = batch[0]["rh"].shape[0]
    rh = torch.zeros((bs, t, 21, 2), dtype=torch.float32)
    lh = torch.zeros((bs, t, 21, 2), dtype=torch.float32)
    bd = torch.zeros((bs, t, 7, 2), dtype=torch.float32)
    mk = torch.zeros((bs, 2 * t, 21, 2), dtype=torch.float32)
    y = torch.zeros((bs,), dtype=torch.long)
    for i, item in enumerate(batch):
        rh[i] = item["rh"]
        lh[i] = item["lh"]
        bd[i] = item["body"]
        mk[i] = item["mask"]
        y[i] = int(item["label"])
    return {"rh": rh, "lh": lh, "body": bd, "mask": mk}, y


def checkpoint_to_state_dict(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    out = {}
    for k, v in sd.items():
        out[k[7:] if k.startswith("module.") else k] = v
    return out


def infer_num_class(sd: Dict[str, torch.Tensor], default: int) -> int:
    for k, v in sd.items():
        if k.endswith("encoder_q.proj.fc.weight") and v.ndim == 2:
            return int(v.shape[0])
    return default


def load_finetuned_model(ckpt_path: str, num_class: int, dropout: float) -> Tuple[nn.Module, Dict[str, int]]:
    sd = checkpoint_to_state_dict(ckpt_path)
    model = MASA(skeleton_representation="graph-based", num_class=num_class, pretrain=False, dropout=dropout)
    msd = model.state_dict()
    loadable = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    skipped = len(sd) - len(loadable)
    msd.update(loadable)
    model.load_state_dict(msd, strict=False)
    return model, {"loaded": len(loadable), "skipped": skipped}


def infer_model_input_dtype(model: nn.Module) -> torch.dtype:
    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype
    for _, b in model.named_buffers():
        if torch.is_tensor(b) and b.is_floating_point():
            return b.dtype
    return torch.float32


def state_dict_numel(model: nn.Module) -> int:
    total = 0
    for _, t in model.state_dict().items():
        if torch.is_tensor(t):
            total += int(t.numel())
    return total


def dtype_numel_stats(model: nn.Module) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    for _, t in model.state_dict().items():
        if not torch.is_tensor(t):
            continue
        k = str(t.dtype).replace("torch.", "")
        stats[k] = stats.get(k, 0) + int(t.numel())
    return stats


def save_state_dict_size_mb(model: nn.Module, out_path: Path) -> float:
    torch.save(model.state_dict(), str(out_path))
    return out_path.stat().st_size / (1024 * 1024)


class FlopsCounter:
    def __init__(self):
        self.macs = 0
        self.handles = []

    def _linear(self, m: nn.Module, inp: Tuple[torch.Tensor], out: torch.Tensor):
        x = inp[0]
        n = int(x.numel() / x.shape[-1])
        self.macs += int(n * x.shape[-1] * out.shape[-1])

    def _conv1d(self, m: nn.Module, inp: Tuple[torch.Tensor], out: torch.Tensor):
        x = inp[0]
        out_elems = int(out.numel())
        in_c = x.shape[1]
        if hasattr(m, "qweight"):
            k = int(m.qweight.shape[-1])
            groups = int(m.groups)
        else:
            k = int(m.kernel_size[0])
            groups = int(m.groups)
        self.macs += int(out_elems * (in_c // groups) * k)

    def _conv2d(self, m: nn.Module, inp: Tuple[torch.Tensor], out: torch.Tensor):
        x = inp[0]
        out_elems = int(out.numel())
        in_c = x.shape[1]
        if hasattr(m, "qweight"):
            kh, kw = int(m.qweight.shape[-2]), int(m.qweight.shape[-1])
            groups = int(m.groups)
        else:
            kh, kw = int(m.kernel_size[0]), int(m.kernel_size[1])
            groups = int(m.groups)
        self.macs += int(out_elems * (in_c // groups) * kh * kw)

    def add_hooks(self, model: nn.Module) -> None:
        for mod in model.modules():
            if isinstance(mod, (nn.Linear, Int8Linear)):
                self.handles.append(mod.register_forward_hook(self._linear))
            elif isinstance(mod, (nn.Conv1d, Int8Conv1d)):
                self.handles.append(mod.register_forward_hook(self._conv1d))
            elif isinstance(mod, (nn.Conv2d, Int8Conv2d)):
                self.handles.append(mod.register_forward_hook(self._conv2d))

    def clear(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    @property
    def flops(self) -> int:
        return int(self.macs * 2)


@dataclass
class EvalResult:
    loss: float
    acc: float
    latency_ms_per_batch: float
    flops_per_batch: int
    param_count_tensors: int
    model_size_mb: float
    dtype_numel: Dict[str, int]


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    out_model_path: Path,
    warmup_steps: int,
) -> EvalResult:
    model = model.to(device).eval()
    dtype = infer_model_input_dtype(model)
    size_mb = save_state_dict_size_mb(model, out_model_path)
    param_count = state_dict_numel(model)
    dtype_stats = dtype_numel_stats(model)

    meter = FlopsCounter()
    meter.add_hooks(model)

    total = 0
    total_loss = 0.0
    correct = 0
    step_times: List[float] = []
    warmup_done = 0

    with torch.no_grad():
        for x, y in tqdm(loader, ncols=100, leave=False):
            x = {k: v.to(device=device, dtype=dtype, non_blocking=(device.type == "cuda")) for k, v in x.items()}
            y = y.to(device, non_blocking=(device.type == "cuda"))

            if warmup_done < warmup_steps:
                _ = model(x)
                warmup_done += 1
                continue

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            step_times.append((t1 - t0) * 1000.0)

            loss = criterion(logits, y)
            bs = y.size(0)
            total += bs
            total_loss += float(loss.item()) * bs
            correct += int((torch.argmax(logits, dim=1) == y).sum().item())

    meter.clear()

    if total == 0:
        raise RuntimeError("No timed/evaluated batches were processed. Reduce warmup_steps or increase data.")

    return EvalResult(
        loss=total_loss / total,
        acc=correct / total,
        latency_ms_per_batch=float(sum(step_times) / len(step_times)) if step_times else 0.0,
        flops_per_batch=meter.flops,
        param_count_tensors=param_count,
        model_size_mb=size_mb,
        dtype_numel=dtype_stats,
    )


def to_dict(res: EvalResult) -> Dict[str, object]:
    return {
        "flops_per_batch": res.flops_per_batch,
        "latency_ms_per_batch": res.latency_ms_per_batch,
        "model_size_mb": res.model_size_mb,
        "param_count_tensors": res.param_count_tensors,
        "loss": res.loss,
        "accuracy": res.acc,
        "dtype_numel": res.dtype_numel,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tuned checkpoint report: INT8 where supported, FP16 fallback otherwise."
    )
    p.add_argument("--finetuned-ckpt", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", default="./quant_finetuned_int8_fp16_report")
    p.add_argument("--subset-num", type=int, default=100)
    p.add_argument("--target-t", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--num-class", type=int, default=100)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"[device] {device}")

    base_test = WLASL(data_root=args.data_root, data_split="test", subset_num=args.subset_num, use_cache=False)
    ds_test = WLASLSupervisedEval(base_test, target_t=args.target_t)
    test_loader = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_supervised,
    )
    print(f"[data] test samples={len(ds_test)}")

    sd = checkpoint_to_state_dict(args.finetuned_ckpt)
    num_class = infer_num_class(sd, args.num_class)
    model_fp32, load_info = load_finetuned_model(args.finetuned_ckpt, num_class, args.dropout)
    print(f"[load] loaded={load_info['loaded']} skipped={load_info['skipped']} num_class={num_class}")

    # Quantize supported layers to int8, then cast remaining floating tensors to fp16 fallback.
    model_hybrid = copy.deepcopy(model_fp32).eval()
    int8_applied = replace_int8_supported_layers(model_hybrid)
    model_hybrid = model_hybrid.half()
    print(f"[quant] int8_applied={int8_applied}")

    criterion = nn.CrossEntropyLoss()
    before = evaluate_model(
        model_fp32, test_loader, criterion, device, out_dir / "before_fp32_state_dict.pth", args.warmup_steps
    )
    after = evaluate_model(
        model_hybrid, test_loader, criterion, device, out_dir / "after_int8_fp16_state_dict.pth", args.warmup_steps
    )

    summary = {
        "checkpoint": args.finetuned_ckpt,
        "load_info": load_info,
        "int8_applied_modules": int8_applied,
        "before_fp32": to_dict(before),
        "after_int8_supported_fp16_fallback": to_dict(after),
        "delta": {
            "flops_change": after.flops_per_batch - before.flops_per_batch,
            "latency_speedup_x": (before.latency_ms_per_batch / after.latency_ms_per_batch) if after.latency_ms_per_batch > 0 else None,
            "size_reduction_mb": before.model_size_mb - after.model_size_mb,
            "param_count_change": after.param_count_tensors - before.param_count_tensors,
            "accuracy_drop": before.acc - after.acc,
        },
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] summary saved: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
