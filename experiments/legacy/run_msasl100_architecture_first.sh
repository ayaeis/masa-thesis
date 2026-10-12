#!/usr/bin/env bash
set -euo pipefail

# Architecture-first comparison.  This never initializes from the fine-tuned
# MSASL100 baseline; it uses generic MASA pretraining for the first stage.
ROOT="${ROOT:-/workspace/masa-thesis}"
export PYTHONPATH="$ROOT"
PYTHON="/workspace/masa_env_rebuilt/bin/python"
DATA_ROOT="/workspace/MSASL/native_masa_ready/MSASL"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-/workspace/checkpoints/pretrained_model.pth.tar}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/masa-thesis/fall_results}"
BASELINE_CKPT="$RESULTS_ROOT/msasl100/baseline/best.pth.tar"
OUT_ROOT="$RESULTS_ROOT/msasl100/architecture_first"
REPORT_DIR="$OUT_ROOT/reports"

mkdir -p "$OUT_ROOT" "$REPORT_DIR"

ensure_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

run_logged() {
  local tag="$1"
  shift
  echo "[$(date '+%F %T')] START $tag"
  "$@"
  echo "[$(date '+%F %T')] DONE  $tag"
}

cleanup_temp_state_dicts() {
  find "$OUT_ROOT" -type f \( -name '*_state_dict.pth' -o -name '*_baseline_state_dict.pth' \) -delete 2>/dev/null || true
}

ensure_file "$PRETRAIN_CKPT"
ensure_file "$BASELINE_CKPT"

COMMON_TRAIN_ARGS=(
  --data-root "$DATA_ROOT"
  --num-class 100
  --epochs 60
  --batch-size 64
  --workers 8
  --optim sgd
  --momentum 0.9
  --lr 0.01
  --weight-decay 0.0
  --scheduler multistep
  --milestones 20 40
  --lr-gamma 0.1
  --target-t 32
  --temporal-sampling index
  --freeze-epochs 0
  --head-lr-mult 1.0
  --warmup-epochs 0
  --train-temporal-crop-min 1.0
  --patience 999
  --min-delta 0.0
  --seed 123
)

COMMON_EVAL_ARGS=(
  --data-root "$DATA_ROOT"
  --num-class 100
  --target-t 32
  --batch-size 32
  --workers 8
  --dropout 0.0
  --warmup-steps 5
  --temporal-sampling index
)

# Ghost Conv: generic pretraining -> 60 epochs -> another 60-epoch stage.
GHOST_60_DIR="$OUT_ROOT/ghost_allk/epoch60"
GHOST_120_DIR="$OUT_ROOT/ghost_allk/epoch120"
GHOST_60_CKPT="$GHOST_60_DIR/best.pth.tar"
GHOST_120_CKPT="$GHOST_120_DIR/best.pth.tar"

if [[ ! -f "$GHOST_60_CKPT" ]]; then
  run_logged "01_train_ghost_allk_epoch60" \
    "$PYTHON" "$ROOT/training/msasl/finetune.py" \
      "${COMMON_TRAIN_ARGS[@]}" \
      --pretrained "$PRETRAIN_CKPT" \
      --use-ghost-conv --ghost-ratio 2 --ghost-mode all \
      --out-dir "$GHOST_60_DIR"
fi

if [[ ! -f "$REPORT_DIR/ghost_allk_epoch60_vs_baseline.json" ]]; then
  run_logged "02_eval_ghost_allk_epoch60" \
    "$PYTHON" "$ROOT/analysis/msasl_checkpoint_metrics.py" \
      --ckpt "$GHOST_60_CKPT" --baseline-ckpt "$BASELINE_CKPT" \
      --out "$REPORT_DIR/ghost_allk_epoch60_vs_baseline.json" \
      "${COMMON_EVAL_ARGS[@]}" \
      --use-ghost-conv --ghost-ratio 2 --ghost-mode all
  cleanup_temp_state_dicts
fi

if [[ ! -f "$GHOST_120_CKPT" ]]; then
  run_logged "03_train_ghost_allk_epoch120" \
    "$PYTHON" "$ROOT/training/msasl/finetune.py" \
      "${COMMON_TRAIN_ARGS[@]}" \
      --pretrained "$GHOST_60_CKPT" \
      --use-ghost-conv --ghost-ratio 2 --ghost-mode all \
      --out-dir "$GHOST_120_DIR"
fi

if [[ ! -f "$REPORT_DIR/ghost_allk_epoch120_vs_baseline.json" ]]; then
  run_logged "04_eval_ghost_allk_epoch120" \
    "$PYTHON" "$ROOT/analysis/msasl_checkpoint_metrics.py" \
      --ckpt "$GHOST_120_CKPT" --baseline-ckpt "$BASELINE_CKPT" \
      --out "$REPORT_DIR/ghost_allk_epoch120_vs_baseline.json" \
      "${COMMON_EVAL_ARGS[@]}" \
      --use-ghost-conv --ghost-ratio 2 --ghost-mode all
  cleanup_temp_state_dicts
fi

# Low rank: factorize pretrained-initialized weights -> 60 epochs -> resume the
# same factorized architecture for a second 60-epoch recovery stage.
LOWRANK_60_DIR="$OUT_ROOT/lowrank_all_r0125/epoch60"
LOWRANK_120_DIR="$OUT_ROOT/lowrank_all_r0125/epoch120"
LOWRANK_60_CKPT="$LOWRANK_60_DIR/best.pth.tar"
LOWRANK_120_CKPT="$LOWRANK_120_DIR/best.pth.tar"
LOWRANK_TRAIN_ARGS=(
  --rank-ratio 0.125
  --low-rank-targets transformer,conv,project_head
  --low-rank-min-features 64
)
LOWRANK_EVAL_ARGS=(
  --use-low-rank
  --low-rank-ratio 0.125
  --low-rank-targets transformer,conv,project_head
  --low-rank-min-features 64
)

if [[ ! -f "$LOWRANK_60_CKPT" ]]; then
  run_logged "05_train_lowrank_all_r0125_epoch60" \
    "$PYTHON" "$ROOT/training/msasl/finetune_lowrank_architecture_first.py" \
      "${COMMON_TRAIN_ARGS[@]}" \
      --init-ckpt "$PRETRAIN_CKPT" \
      --init-mode pretrained_dense \
      "${LOWRANK_TRAIN_ARGS[@]}" \
      --out-dir "$LOWRANK_60_DIR"
fi

if [[ ! -f "$REPORT_DIR/lowrank_all_r0125_epoch60_vs_baseline.json" ]]; then
  run_logged "06_eval_lowrank_all_r0125_epoch60" \
    "$PYTHON" "$ROOT/analysis/msasl_checkpoint_metrics.py" \
      --ckpt "$LOWRANK_60_CKPT" --baseline-ckpt "$BASELINE_CKPT" \
      --out "$REPORT_DIR/lowrank_all_r0125_epoch60_vs_baseline.json" \
      "${COMMON_EVAL_ARGS[@]}" "${LOWRANK_EVAL_ARGS[@]}"
  cleanup_temp_state_dicts
fi

if [[ ! -f "$LOWRANK_120_CKPT" ]]; then
  run_logged "07_train_lowrank_all_r0125_epoch120" \
    "$PYTHON" "$ROOT/training/msasl/finetune_lowrank_architecture_first.py" \
      "${COMMON_TRAIN_ARGS[@]}" \
      --init-ckpt "$LOWRANK_60_CKPT" \
      --init-mode resume_low_rank \
      "${LOWRANK_TRAIN_ARGS[@]}" \
      --out-dir "$LOWRANK_120_DIR"
fi

if [[ ! -f "$REPORT_DIR/lowrank_all_r0125_epoch120_vs_baseline.json" ]]; then
  run_logged "08_eval_lowrank_all_r0125_epoch120" \
    "$PYTHON" "$ROOT/analysis/msasl_checkpoint_metrics.py" \
      --ckpt "$LOWRANK_120_CKPT" --baseline-ckpt "$BASELINE_CKPT" \
      --out "$REPORT_DIR/lowrank_all_r0125_epoch120_vs_baseline.json" \
      "${COMMON_EVAL_ARGS[@]}" "${LOWRANK_EVAL_ARGS[@]}"
  cleanup_temp_state_dicts
fi

echo
echo "Architecture-first MSASL100 comparison completed."
echo "Results root: $OUT_ROOT"
