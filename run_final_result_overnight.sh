#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/masa-thesis"
ENV_ACTIVATE="/workspace/masa_env/bin/activate"
DATA_ROOT="/workspace/kaggle_wlasl100/masa_ready/WLASL"
GHOST_INIT_CKPT="$ROOT/baseline_ckpt/best.pth.tar"
BASELINE_CKPT="$ROOT/baseline_ckpt/best.pth.tar"
OUT_ROOT="$ROOT/final_result"
REPORT_DIR="$OUT_ROOT/reports"
LOG_DIR="$OUT_ROOT/logs"

mkdir -p "$OUT_ROOT" "$REPORT_DIR" "$LOG_DIR"

if [[ -f "$ENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_ACTIVATE"
fi

cd "$ROOT"

run_logged() {
  local name="$1"
  shift
  local log_path="$LOG_DIR/${name}.log"
  echo
  echo "[$(date '+%F %T')] START $name"
  "$@" 2>&1 | tee "$log_path"
  local cmd_status=${PIPESTATUS[0]}
  cleanup_temp_state_dicts
  if [[ $cmd_status -ne 0 ]]; then
    return "$cmd_status"
  fi
  echo "[$(date '+%F %T')] DONE  $name"
}

ensure_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[error] missing required file: $path" >&2
    exit 1
  fi
}

cleanup_temp_state_dicts() {
  find "$REPORT_DIR" -maxdepth 1 -type f \( -name '*_state_dict.pth' -o -name '*_baseline_state_dict.pth' \) -delete 2>/dev/null || true
}

ensure_file "$BASELINE_CKPT"
ensure_file "$GHOST_INIT_CKPT"

COMMON_TRAIN_ARGS=(
  --data-root "$DATA_ROOT"
  --pretrained "$GHOST_INIT_CKPT"
  --num-class 100
  --subset-num 100
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
  --subset-num 100
  --target-t 32
  --batch-size 32
  --workers 8
  --num-class 100
  --dropout 0.0
  --warmup-steps 5
  --temporal-sampling index
)

COMMON_QUANT_ARGS=(
  --data-root "$DATA_ROOT"
  --subset-num 100
  --target-t 32
  --batch-size 32
  --workers 8
  --num-class 100
  --dropout 0.0
  --warmup-steps 5
  --temporal-sampling index
  --baseline-ckpt "$BASELINE_CKPT"
)

COMMON_KD_ARGS=(
  --data-root "$DATA_ROOT"
  --teacher-ckpt "$BASELINE_CKPT"
  --subset-num 100
  --num-class 100
  --epochs 60
  --batch-size 64
  --workers 8
  --lr 0.01
  --momentum 0.9
  --weight-decay 0.0
  --optim sgd
  --scheduler multistep
  --milestones 20 40
  --lr-gamma 0.1
  --freeze-epochs 0
  --head-lr-mult 1.0
  --warmup-epochs 0
  --train-temporal-crop-min 1.0
  --patience 999
  --min-delta 0.0
  --target-t 32
  --seed 123
  --dropout 0.0
  --temporal-sampling index
  --kd-alpha 0.5
  --kd-temp 4.0
)

declare -A GHOST_MODE_MAP=(
  [allk]="all"
  [k1]="kernel1"
  [gt1]="gt1"
)

baseline_report="$REPORT_DIR/baseline_report.json"
if [[ ! -f "$baseline_report" ]]; then
  run_logged "01_baseline_report" \
    python "$ROOT/report_checkpoint_metrics.py" \
      --ckpt "$BASELINE_CKPT" \
      --out "$baseline_report" \
      "${COMMON_EVAL_ARGS[@]}"
fi

if [[ ! -f "$OUT_ROOT/quant_baseline/summary.json" ]]; then
  run_logged "02_quant_baseline" \
    python "$ROOT/quantize_finetuned_int8_fp16_report.py" \
      --finetuned-ckpt "$BASELINE_CKPT" \
      --out-dir "$OUT_ROOT/quant_baseline" \
      "${COMMON_QUANT_ARGS[@]}"
fi

for tag in allk k1 gt1; do
  ghost_mode="${GHOST_MODE_MAP[$tag]}"
  ghost_dir="$OUT_ROOT/ghost_${tag}"
  ghost_ckpt="$ghost_dir/best.pth.tar"
  ghost_report="$REPORT_DIR/ghost_${tag}_vs_baseline.json"
  quant_ghost_dir="$OUT_ROOT/quant_ghost_${tag}"
  kd_dir="$OUT_ROOT/kd_ghost_${tag}"
  kd_ckpt="$kd_dir/best.pth.tar"
  kd_report_baseline="$REPORT_DIR/kd_ghost_${tag}_vs_baseline.json"
  kd_report_ghost="$REPORT_DIR/kd_ghost_${tag}_vs_ghost.json"
  quant_kd_dir="$OUT_ROOT/quant_kd_ghost_${tag}"

  if [[ ! -f "$ghost_ckpt" ]]; then
    run_logged "03_train_ghost_${tag}" \
      python "$ROOT/finetune_wlasl100.py" \
        "${COMMON_TRAIN_ARGS[@]}" \
        --use-ghost-conv \
        --ghost-ratio 2 \
        --ghost-mode "$ghost_mode" \
        --out-dir "$ghost_dir"
  fi

  if [[ ! -f "$ghost_report" ]]; then
    run_logged "04_eval_ghost_${tag}" \
      python "$ROOT/report_checkpoint_metrics.py" \
        --ckpt "$ghost_ckpt" \
        --baseline-ckpt "$BASELINE_CKPT" \
        --out "$ghost_report" \
        "${COMMON_EVAL_ARGS[@]}" \
        --use-ghost-conv \
        --ghost-ratio 2 \
        --ghost-mode "$ghost_mode"
  fi

  if [[ ! -f "$quant_ghost_dir/summary.json" ]]; then
    run_logged "05_quant_ghost_${tag}" \
      python "$ROOT/quantize_finetuned_int8_fp16_report.py" \
        --finetuned-ckpt "$ghost_ckpt" \
        --out-dir "$quant_ghost_dir" \
        "${COMMON_QUANT_ARGS[@]}" \
        --use-ghost-conv \
        --ghost-ratio 2 \
        --ghost-mode "$ghost_mode"
  fi

  if [[ ! -f "$kd_ckpt" ]]; then
    run_logged "06_train_kd_ghost_${tag}" \
      python "$ROOT/finetune_wlasl100_kd.py" \
        "${COMMON_KD_ARGS[@]}" \
        --student-ckpt "$ghost_ckpt" \
        --student-use-ghost-conv \
        --student-ghost-ratio 2 \
        --student-ghost-mode "$ghost_mode" \
        --out-dir "$kd_dir"
  fi

  if [[ ! -f "$kd_report_baseline" ]]; then
    run_logged "07_eval_kd_ghost_${tag}_vs_baseline" \
      python "$ROOT/report_checkpoint_metrics.py" \
        --ckpt "$kd_ckpt" \
        --baseline-ckpt "$BASELINE_CKPT" \
        --out "$kd_report_baseline" \
        "${COMMON_EVAL_ARGS[@]}" \
        --use-ghost-conv \
        --ghost-ratio 2 \
        --ghost-mode "$ghost_mode"
  fi

  if [[ ! -f "$kd_report_ghost" ]]; then
    run_logged "08_eval_kd_ghost_${tag}_vs_ghost" \
      python "$ROOT/report_checkpoint_metrics.py" \
        --ckpt "$kd_ckpt" \
        --baseline-ckpt "$ghost_ckpt" \
        --out "$kd_report_ghost" \
        "${COMMON_EVAL_ARGS[@]}" \
        --use-ghost-conv \
        --ghost-ratio 2 \
        --ghost-mode "$ghost_mode" \
        --baseline-use-ghost-conv \
        --baseline-ghost-ratio 2 \
        --baseline-ghost-mode "$ghost_mode"
  fi

  if [[ ! -f "$quant_kd_dir/summary.json" ]]; then
    run_logged "09_quant_kd_ghost_${tag}" \
      python "$ROOT/quantize_finetuned_int8_fp16_report.py" \
        --finetuned-ckpt "$kd_ckpt" \
        --out-dir "$quant_kd_dir" \
        "${COMMON_QUANT_ARGS[@]}" \
        --use-ghost-conv \
        --ghost-ratio 2 \
        --ghost-mode "$ghost_mode"
  fi
done

echo
echo "All overnight experiments completed."
echo "Results root: $OUT_ROOT"
