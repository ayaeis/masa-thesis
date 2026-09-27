#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/masa-thesis}"
ENV="/workspace/masa_env_rebuilt"
DATA_ROOT="/workspace/MSASL/native_masa_ready/MSASL"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/masa-thesis/fall_results}"
BASELINE_CKPT="$RESULTS_ROOT/msasl100/baseline/best.pth.tar"
RUN="$ROOT/run_fall_msasl100_overnight.sh"
export BASELINE_CKPT

[[ -d "$DATA_ROOT" ]] || { echo "[FAIL] Missing MSASL data: $DATA_ROOT"; exit 1; }
[[ -f "$BASELINE_CKPT" ]] || { echo "[FAIL] Missing MSASL100 baseline: $BASELINE_CKPT"; exit 1; }
[[ -f "$RUN" ]] || { echo "[FAIL] Missing runner: $RUN"; exit 1; }
bash -n "$RUN"
echo "[OK] Runner shell syntax"

cd "$ROOT"
PYTHONPATH="$ROOT" "$ENV/bin/python" - <<'PY'
import os

import torch

from data_prep.msasl.msasl200_archive_loader import MSASLArchive

root = "/workspace/MSASL/native_masa_ready/MSASL"
train = MSASLArchive(root, data_split="train", class_num=100, use_cache=False)
test = MSASLArchive(root, data_split="test", class_num=100, use_cache=False)

assert train.flag > 0 and len(train) > train.flag
assert len(test) > 0
print(f"[OK] MSASL100 train={train.flag}, val={len(train) - train.flag}, test={len(test)}")

checkpoint = torch.load(
    os.environ["BASELINE_CKPT"],
    map_location="cpu",
)
state_dict = checkpoint["state_dict"]
weight = state_dict["encoder_q.proj.fc.weight"]
assert weight.shape[0] == 100, weight.shape
print("[OK] Baseline classifier: 100 classes")
PY

for script in \
  training/msasl/finetune.py \
  training/msasl/finetune_ghost_kd.py \
  training/msasl/finetune_lowrank.py \
  analysis/msasl_checkpoint_metrics.py \
  analysis/msasl_quantization_report.py; do
  PYTHONPATH="$ROOT" "$ENV/bin/python" "$script" --help >/dev/null
done
echo "[OK] All MSASL training and evaluation scripts import and start"
echo "[PASS] MSASL100 overnight preflight complete."
