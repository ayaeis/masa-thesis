import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


@dataclass
class LowRankReplacement:
    name: str
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


def compute_low_rank_rank(mod: nn.Linear, rank_ratio: float) -> int:
    rank_ratio = float(rank_ratio)
    if not (0.0 < rank_ratio < 1.0):
        raise ValueError(f"low-rank ratio must be in (0, 1), got {rank_ratio}")
    rank = max(1, int(math.ceil(min(mod.in_features, mod.out_features) * rank_ratio)))
    return min(rank, min(mod.in_features, mod.out_features) - 1)


def would_compress(mod: nn.Linear, rank: int) -> bool:
    original_params = mod.in_features * mod.out_features + (mod.out_features if mod.bias is not None else 0)
    low_rank_params = mod.in_features * rank + rank * mod.out_features + (mod.out_features if mod.bias is not None else 0)
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
        if not isinstance(mod, nn.Linear):
            continue
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
                in_features=mod.in_features,
                out_features=mod.out_features,
                rank=rank,
                original_params=mod.in_features * mod.out_features + (mod.out_features if mod.bias is not None else 0),
                low_rank_params=mod.in_features * rank + rank * mod.out_features + (mod.out_features if mod.bias is not None else 0),
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
        if not isinstance(mod, nn.Linear):
            continue
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
                in_features=mod.in_features,
                out_features=mod.out_features,
                rank=rank,
                original_params=mod.in_features * mod.out_features + (mod.out_features if mod.bias is not None else 0),
                low_rank_params=mod.in_features * rank + rank * mod.out_features + (mod.out_features if mod.bias is not None else 0),
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
                "in_features": item.in_features,
                "out_features": item.out_features,
                "rank": item.rank,
                "original_params": item.original_params,
                "low_rank_params": item.low_rank_params,
            }
            for item in replacements
        ],
    }
