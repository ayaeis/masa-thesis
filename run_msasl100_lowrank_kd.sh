#!/usr/bin/env bash
set -euo pipefail

# Targeted KD-recovery ablation: dense MSASL100 baseline teacher and the
# completed post-training all-layer low-rank checkpoints as students.
ROOT="${ROOT:-/workspace/masa-thesis}"
export PYTHONPATH="$ROOT"
PYTHON="/workspace/masa_env_rebuilt/bin/python"
DATA_ROOT="/workspace/MSASL/native_masa_ready/MSASL"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/masa-thesis/fall_results}"
BASELINE_CKPT="$RESULTS_ROOT/msasl100/baseline/best.pth.tar"
LOWRANK_ROOT="$RESULTS_ROOT/msasl100/overnight_final_run/lowrank"
OUT_ROOT="$RESULTS_ROOT/msasl100/lowrank_kd_recovery"
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

# Three compression severities. Each KD model starts from the existing
# post-training student and is compared with both baseline and that student.
LOWRANK_CONFIGS=(
  "0.125|r0125"
  "0.25|r25"
  "0.50|r05"
)

for config in "${LOWRANK_CONFIGS[@]}"; do
  IFS='|' read -r rank_ratio rank_tag <<< "$config"
  target_spec="transformer,conv,project_head"
  student_ckpt="$LOWRANK_ROOT/all_${rank_tag}/best.pth.tar"
  kd_dir="$OUT_ROOT/all_${rank_tag}"
  kd_ckpt="$kd_dir/best.pth.tar"
  report_baseline="$REPORT_DIR/all_${rank_tag}_vs_baseline.json"
  report_student="$REPORT_DIR/all_${rank_tag}_vs_post_training_student.json"

  ensure_file "$student_ckpt"

  if [[ ! -f "$kd_ckpt" ]]; then
    run_logged "01_train_lowrank_kd_all_${rank_tag}" \
      "$PYTHON" "$ROOT/training/msasl/finetune_lowrank_kd.py" \
        "${COMMON_TRAIN_ARGS[@]}" \
        --teacher-ckpt "$BASELINE_CKPT" \
        --student-ckpt "$student_ckpt" \
        --student-low-rank-ratio "$rank_ratio" \
        --student-low-rank-targets "$target_spec" \
        --student-low-rank-min-features 64 \
        --dropout 0.0 \
        --kd-alpha 0.5 \
        --kd-temp 4.0 \
        --out-dir "$kd_dir"
  fi

  if [[ ! -f "$report_baseline" ]]; then
    run_logged "02_eval_lowrank_kd_all_${rank_tag}_vs_baseline" \
      "$PYTHON" "$ROOT/analysis/msasl_checkpoint_metrics.py" \
        --ckpt "$kd_ckpt" \
        --baseline-ckpt "$BASELINE_CKPT" \
        --out "$report_baseline" \
        "${COMMON_EVAL_ARGS[@]}" \
        --use-low-rank \
        --low-rank-ratio "$rank_ratio" \
        --low-rank-targets "$target_spec" \
        --low-rank-min-features 64
    cleanup_temp_state_dicts
  fi

  if [[ ! -f "$report_student" ]]; then
    run_logged "03_eval_lowrank_kd_all_${rank_tag}_vs_post_training_student" \
      "$PYTHON" "$ROOT/analysis/msasl_checkpoint_metrics.py" \
        --ckpt "$kd_ckpt" \
        --baseline-ckpt "$student_ckpt" \
        --out "$report_student" \
        "${COMMON_EVAL_ARGS[@]}" \
        --use-low-rank \
        --low-rank-ratio "$rank_ratio" \
        --low-rank-targets "$target_spec" \
        --low-rank-min-features 64 \
        --baseline-use-low-rank \
        --baseline-low-rank-ratio "$rank_ratio" \
        --baseline-low-rank-targets "$target_spec" \
        --baseline-low-rank-min-features 64
    cleanup_temp_state_dicts
  fi
done

echo
echo "MSASL100 low-rank KD recovery ablation completed."
echo "Results root: $OUT_ROOT"
