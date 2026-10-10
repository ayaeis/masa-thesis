#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/masa-thesis}"
ENV="/workspace/masa_env_rebuilt"
DATA_ROOT="/workspace/MSASL/native_masa_ready/MSASL"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/masa-thesis/fall_results}"
BASELINE_CKPT="$RESULTS_ROOT/msasl100/baseline/best.pth.tar"
LOWRANK_ROOT="$RESULTS_ROOT/msasl100/overnight_final_run/lowrank"
RUN="$ROOT/run_msasl100_lowrank_kd.sh"

[[ -d "$DATA_ROOT" ]] || { echo "[FAIL] Missing MSASL data: $DATA_ROOT"; exit 1; }
[[ -f "$BASELINE_CKPT" ]] || { echo "[FAIL] Missing dense baseline: $BASELINE_CKPT"; exit 1; }
[[ -f "$RUN" ]] || { echo "[FAIL] Missing runner: $RUN"; exit 1; }

for tag in r0125 r25 r05; do
  ckpt="$LOWRANK_ROOT/all_${tag}/best.pth.tar"
  [[ -f "$ckpt" ]] || { echo "[FAIL] Missing post-training low-rank student: $ckpt"; exit 1; }
done

bash -n "$RUN"
echo "[OK] Runner shell syntax"

cd "$ROOT"
PYTHONPATH="$ROOT" "$ENV/bin/python" - <<'PY'
from data_prep.msasl.msasl200_archive_loader import MSASLArchive

root = "/workspace/MSASL/native_masa_ready/MSASL"
train = MSASLArchive(root, data_split="train", class_num=100, use_cache=False)
test = MSASLArchive(root, data_split="test", class_num=100, use_cache=False)
assert train.flag == 3788 and len(train) - train.flag == 1190 and len(test) == 757
assert {int(row[1]) for row in train.video_list} == set(range(100))
assert {int(row[1]) for row in test.video_list} == set(range(100))
print("[OK] MSASL100 split: train=3788, val=1190, test=757, classes=100")
PY

for script in \
  training/msasl/finetune_lowrank_kd.py \
  analysis/msasl_checkpoint_metrics.py; do
  PYTHONPATH="$ROOT" "$ENV/bin/python" "$script" --help >/dev/null
done
echo "[OK] Low-rank KD trainer and evaluator import and start"
echo "[PASS] MSASL100 low-rank KD preflight complete."
