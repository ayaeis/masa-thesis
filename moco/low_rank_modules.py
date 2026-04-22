import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


@dataclass
class LowRankReplacement:
    name: str
    kind: str
    in_features: int
    out_features: int
    rank: int
    original_params: int
    low_rank_params: int


class LowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.first = nn.Linear(in_features, rank, bias=False)
        self.second = nn.Linear(rank, out_features, bias=bias)

    @classmethod
    def from_linear(cls, mod: nn.Linear, rank: int) -> "LowRankLinear":
        low_rank = cls(mod.in_features, mod.out_features, rank, bias=(mod.bias is not None))
        weight = mod.weight.detach().float()
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        u_r = u[:, :rank]
        s_r = s[:rank]
        vh_r = vh[:rank, :]
        sqrt_s = torch.sqrt(torch.clamp(s_r, min=0))
        first_weight = torch.diag(sqrt_s) @ vh_r
        second_weight = u_r @ torch.diag(sqrt_s)
        low_rank.first.weight.data.copy_(first_weight.to(mod.weight.dtype))
        low_rank.second.weight.data.copy_(second_weight.to(mod.weight.dtype))
        if mod.bias is not None:
            low_rank.second.bias.data.copy_(mod.bias.detach())
        return low_rank

    def forward(self, x):
        return self.second(self.first(x))


class LowRankConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        self.first = nn.Conv1d(
            in_channels,
            rank,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=False,
        )
        self.second = nn.Conv1d(rank, out_channels, kernel_size=1, bias=bias)

    @classmethod
    def from_conv(cls, mod: nn.Conv1d, rank: int) -> "LowRankConv1d":
        low_rank = cls(
            mod.in_channels,
            mod.out_channels,
            rank,
            kernel_size=mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            bias=(mod.bias is not None),
        )
        weight = mod.weight.detach().float().reshape(mod.out_channels, -1)
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        u_r = u[:, :rank]
        s_r = s[:rank]
        vh_r = vh[:rank, :]
        sqrt_s = torch.sqrt(torch.clamp(s_r, min=0))
        first_weight = torch.diag(sqrt_s) @ vh_r
        second_weight = u_r @ torch.diag(sqrt_s)
        low_rank.first.weight.data.copy_(first_weight.reshape_as(low_rank.first.weight).to(mod.weight.dtype))
        low_rank.second.weight.data.copy_(second_weight.reshape_as(low_rank.second.weight).to(mod.weight.dtype))
        if mod.bias is not None:
            low_rank.second.bias.data.copy_(mod.bias.detach())
        return low_rank

    def forward(self, x):
        return self.second(self.first(x))


class LowRankConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        self.first = nn.Conv2d(
            in_channels,
            rank,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=False,
        )
        self.second = nn.Conv2d(rank, out_channels, kernel_size=1, bias=bias)

    @classmethod
    def from_conv(cls, mod: nn.Conv2d, rank: int) -> "LowRankConv2d":
        low_rank = cls(
            mod.in_channels,
            mod.out_channels,
            rank,
            kernel_size=mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            bias=(mod.bias is not None),
        )
        weight = mod.weight.detach().float().reshape(mod.out_channels, -1)
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        u_r = u[:, :rank]
        s_r = s[:rank]
        vh_r = vh[:rank, :]
        sqrt_s = torch.sqrt(torch.clamp(s_r, min=0))
        first_weight = torch.diag(sqrt_s) @ vh_r
        second_weight = u_r @ torch.diag(sqrt_s)
        low_rank.first.weight.data.copy_(first_weight.reshape_as(low_rank.first.weight).to(mod.weight.dtype))
        low_rank.second.weight.data.copy_(second_weight.reshape_as(low_rank.second.weight).to(mod.weight.dtype))
        if mod.bias is not None:
            low_rank.second.bias.data.copy_(mod.bias.detach())
        return low_rank

    def forward(self, x):
        return self.second(self.first(x))


def parse_low_rank_targets(spec: str) -> List[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def should_factorize_linear(name: str, mod: nn.Linear, targets: List[str], min_features: int) -> bool:
    if min(mod.in_features, mod.out_features) < min_features:
        return False

    if "all_linear" in targets:
        return True

    if "transformer" in targets and "encoder_q.GCN_Tran" in name:
        return True

    if "project_head" in targets and any(
        key in name
        for key in ("encoder_q.proj.fc", "encoder_q.proj.body_fc", "encoder_q.proj.hand_fc")
    ):
        return True

    return False


def should_factorize_conv(name: str, mod, targets: List[str], min_features: int) -> bool:
    if "conv" not in targets and "all" not in targets:
        return False
    if getattr(mod, "groups", 1) != 1:
        return False
    if min(mod.in_channels, mod.out_channels) < min_features:
        return False
    return True


def compute_rank(in_dim: int, out_dim: int, rank_ratio: float) -> int:
    rank_ratio = float(rank_ratio)
    if not (0.0 < rank_ratio < 1.0):
        raise ValueError(f"low-rank ratio must be in (0, 1), got {rank_ratio}")
    rank = max(1, int(math.ceil(min(in_dim, out_dim) * rank_ratio)))
    return min(rank, min(in_dim, out_dim) - 1)


def compute_low_rank_rank(mod: nn.Linear, rank_ratio: float) -> int:
    return compute_rank(mod.in_features, mod.out_features, rank_ratio)


def would_compress(mod: nn.Linear, rank: int) -> bool:
    original_params = mod.in_features * mod.out_features + (mod.out_features if mod.bias is not None else 0)
    low_rank_params = mod.in_features * rank + rank * mod.out_features + (mod.out_features if mod.bias is not None else 0)
    return low_rank_params < original_params


def conv_param_counts(mod, rank: int) -> Tuple[int, int]:
    kernel_params = 1
    for k in mod.kernel_size:
        kernel_params *= int(k)
    original_params = mod.out_channels * (mod.in_channels // mod.groups) * kernel_params
    original_params += mod.out_channels if mod.bias is not None else 0
    low_rank_params = rank * mod.in_channels * kernel_params + mod.out_channels * rank
    low_rank_params += mod.out_channels if mod.bias is not None else 0
    return int(original_params), int(low_rank_params)


def would_compress_conv(mod, rank: int) -> bool:
    original_params, low_rank_params = conv_param_counts(mod, rank)
    return low_rank_params < original_params


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module) -> None:
    parent = model
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def apply_low_rank_from_dense(
    model: nn.Module,
    rank_ratio: float,
    target_spec: str = "transformer",
    min_features: int = 64,
) -> Dict[str, object]:
    targets = parse_low_rank_targets(target_spec)
    replacements: List[LowRankReplacement] = []

    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear):
            if not should_factorize_linear(name, mod, targets, min_features):
                continue
            rank = compute_low_rank_rank(mod, rank_ratio)
            if not would_compress(mod, rank):
                continue
            low_rank = LowRankLinear.from_linear(mod, rank)
            _set_module_by_name(model, name, low_rank)
            replacements.append(
                LowRankReplacement(
                    name=name,
                    kind="linear",
                    in_features=mod.in_features,
                    out_features=mod.out_features,
                    rank=rank,
                    original_params=mod.in_features * mod.out_features + (mod.out_features if mod.bias is not None else 0),
                    low_rank_params=mod.in_features * rank + rank * mod.out_features + (mod.out_features if mod.bias is not None else 0),
                )
            )
        elif isinstance(mod, nn.Conv1d):
            if not should_factorize_conv(name, mod, targets, min_features):
                continue
            rank = compute_rank(mod.in_channels, mod.out_channels, rank_ratio)
            if not would_compress_conv(mod, rank):
                continue
            low_rank = LowRankConv1d.from_conv(mod, rank)
            original_params, low_rank_params = conv_param_counts(mod, rank)
            _set_module_by_name(model, name, low_rank)
            replacements.append(
                LowRankReplacement(
                    name=name,
                    kind="conv1d",
                    in_features=mod.in_channels,
                    out_features=mod.out_channels,
                    rank=rank,
                    original_params=original_params,
                    low_rank_params=low_rank_params,
                )
            )
        elif isinstance(mod, nn.Conv2d):
            if not should_factorize_conv(name, mod, targets, min_features):
                continue
            rank = compute_rank(mod.in_channels, mod.out_channels, rank_ratio)
            if not would_compress_conv(mod, rank):
                continue
            low_rank = LowRankConv2d.from_conv(mod, rank)
            original_params, low_rank_params = conv_param_counts(mod, rank)
            _set_module_by_name(model, name, low_rank)
            replacements.append(
                LowRankReplacement(
                    name=name,
                    kind="conv2d",
                    in_features=mod.in_channels,
                    out_features=mod.out_channels,
                    rank=rank,
                    original_params=original_params,
                    low_rank_params=low_rank_params,
                )
            )

    return summarize_replacements(replacements)


def apply_low_rank_structure(
    model: nn.Module,
    rank_ratio: float,
    target_spec: str = "transformer",
    min_features: int = 64,
) -> Dict[str, object]:
    targets = parse_low_rank_targets(target_spec)
    replacements: List[LowRankReplacement] = []

    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear):
            if not should_factorize_linear(name, mod, targets, min_features):
                continue
            rank = compute_low_rank_rank(mod, rank_ratio)
            if not would_compress(mod, rank):
                continue
            low_rank = LowRankLinear(mod.in_features, mod.out_features, rank, bias=(mod.bias is not None))
            _set_module_by_name(model, name, low_rank)
            replacements.append(
                LowRankReplacement(
                    name=name,
                    kind="linear",
                    in_features=mod.in_features,
                    out_features=mod.out_features,
                    rank=rank,
                    original_params=mod.in_features * mod.out_features + (mod.out_features if mod.bias is not None else 0),
                    low_rank_params=mod.in_features * rank + rank * mod.out_features + (mod.out_features if mod.bias is not None else 0),
                )
            )
        elif isinstance(mod, nn.Conv1d):
            if not should_factorize_conv(name, mod, targets, min_features):
                continue
            rank = compute_rank(mod.in_channels, mod.out_channels, rank_ratio)
            if not would_compress_conv(mod, rank):
                continue
            low_rank = LowRankConv1d(mod.in_channels, mod.out_channels, rank, mod.kernel_size, mod.stride, mod.padding, mod.dilation, bias=(mod.bias is not None))
            original_params, low_rank_params = conv_param_counts(mod, rank)
            _set_module_by_name(model, name, low_rank)
            replacements.append(
                LowRankReplacement(
                    name=name,
                    kind="conv1d",
                    in_features=mod.in_channels,
                    out_features=mod.out_channels,
                    rank=rank,
                    original_params=original_params,
                    low_rank_params=low_rank_params,
                )
            )
        elif isinstance(mod, nn.Conv2d):
            if not should_factorize_conv(name, mod, targets, min_features):
                continue
            rank = compute_rank(mod.in_channels, mod.out_channels, rank_ratio)
            if not would_compress_conv(mod, rank):
                continue
            low_rank = LowRankConv2d(mod.in_channels, mod.out_channels, rank, mod.kernel_size, mod.stride, mod.padding, mod.dilation, bias=(mod.bias is not None))
            original_params, low_rank_params = conv_param_counts(mod, rank)
            _set_module_by_name(model, name, low_rank)
            replacements.append(
                LowRankReplacement(
                    name=name,
                    kind="conv2d",
                    in_features=mod.in_channels,
                    out_features=mod.out_channels,
                    rank=rank,
                    original_params=original_params,
                    low_rank_params=low_rank_params,
                )
            )

    return summarize_replacements(replacements)


def summarize_replacements(replacements: List[LowRankReplacement]) -> Dict[str, object]:
    return {
        "num_replaced": len(replacements),
        "original_params_total": int(sum(item.original_params for item in replacements)),
        "low_rank_params_total": int(sum(item.low_rank_params for item in replacements)),
        "param_reduction_total": int(sum(item.original_params - item.low_rank_params for item in replacements)),
        "layers": [
            {
                "name": item.name,
                "kind": item.kind,
                "in_features": item.in_features,
                "out_features": item.out_features,
                "rank": item.rank,
                "original_params": item.original_params,
                "low_rank_params": item.low_rank_params,
            }
            for item in replacements
        ],
    }
