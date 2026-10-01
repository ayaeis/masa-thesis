#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/masa-thesis}"
ENV="/workspace/masa_env_rebuilt"
DATA_ROOT="/workspace/MSASL/native_masa_ready/MSASL"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-/workspace/checkpoints/pretrained_model.pth.tar}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/masa-thesis/fall_results}"
BASELINE_CKPT="$RESULTS_ROOT/msasl100/baseline/best.pth.tar"
RUN="$ROOT/run_msasl100_architecture_first.sh"

[[ -d "$DATA_ROOT" ]] || { echo "[FAIL] Missing MSASL data: $DATA_ROOT"; exit 1; }
[[ -f "$PRETRAIN_CKPT" ]] || { echo "[FAIL] Missing MASA pretraining checkpoint: $PRETRAIN_CKPT"; exit 1; }
[[ -f "$BASELINE_CKPT" ]] || { echo "[FAIL] Missing MSASL100 baseline: $BASELINE_CKPT"; exit 1; }
[[ -f "$RUN" ]] || { echo "[FAIL] Missing runner: $RUN"; exit 1; }

bash -n "$RUN"
echo "[OK] Runner shell syntax"

cd "$ROOT"
PYTHONPATH="$ROOT" "$ENV/bin/python" - <<'PY'
import torch

from data_prep.msasl.msasl200_archive_loader import MSASLArchive

root = "/workspace/MSASL/native_masa_ready/MSASL"
train = MSASLArchive(root, data_split="train", class_num=100, use_cache=False)
test = MSASLArchive(root, data_split="test", class_num=100, use_cache=False)
assert train.flag == 3788 and len(train) - train.flag == 1190 and len(test) == 757
assert {int(row[1]) for row in train.video_list} == set(range(100))
assert {int(row[1]) for row in test.video_list} == set(range(100))
print("[OK] MSASL100 loader split: train=3788, val=1190, test=757, classes=100")

checkpoint = torch.load("/workspace/checkpoints/pretrained_model.pth.tar", map_location="cpu")
state_dict = checkpoint.get("state_dict", checkpoint)
assert any(key.startswith("encoder_q.") or key.startswith("module.encoder_q.") for key in state_dict)
print("[OK] MASA pretraining checkpoint contains encoder_q weights")
PY

for script in \
  training/msasl/finetune.py \
  training/msasl/finetune_lowrank_architecture_first.py \
  analysis/msasl_checkpoint_metrics.py; do
  PYTHONPATH="$ROOT" "$ENV/bin/python" "$script" --help >/dev/null
done
echo "[OK] Ghost, low-rank, and evaluation scripts import and start"
echo "[PASS] MSASL100 architecture-first preflight complete."
