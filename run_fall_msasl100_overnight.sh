#!/usr/bin/env bash
set -euo pipefail

# A second pod can clone the code to its container disk while sharing /workspace data.
ROOT="${ROOT:-/workspace/masa-thesis}"
export PYTHONPATH="$ROOT"
PYTHON="/workspace/masa_env_rebuilt/bin/python"
DATA_ROOT="/workspace/MSASL/native_masa_ready/MSASL"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/masa-thesis/fall_results}"
BASELINE_CKPT="$RESULTS_ROOT/msasl100/baseline/best.pth.tar"
OUT_ROOT="$RESULTS_ROOT/msasl100/overnight_final_run"
REPORT_DIR="$OUT_ROOT/reports"

mkdir -p "$OUT_ROOT" "$REPORT_DIR"

ensure_file() {
  local path="$1"
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
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

COMMON_QUANT_ARGS=(
  "${COMMON_EVAL_ARGS[@]}"
  --baseline-ckpt "$BASELINE_CKPT"
)

COMMON_KD_ARGS=(
  "${COMMON_TRAIN_ARGS[@]}"
  --teacher-ckpt "$BASELINE_CKPT"
  --dropout 0.0
  --kd-alpha 0.5
  --kd-temp 4.0
)

baseline_report="$REPORT_DIR/baseline_report.json"
if [[ ! -f "$baseline_report" ]]; then
  run_logged "01_baseline_report" \
    "$PYTHON" "$ROOT/analysis/msasl_checkpoint_metrics.py" \
      --ckpt "$BASELINE_CKPT" \
      --out "$baseline_report" \
      "${COMMON_EVAL_ARGS[@]}"
  cleanup_temp_state_dicts
fi

if [[ ! -f "$OUT_ROOT/quant_baseline/summary.json" ]]; then
  run_logged "02_quant_baseline" \
    "$PYTHON" "$ROOT/analysis/msasl_quantization_report.py" \
      --finetuned-ckpt "$BASELINE_CKPT" \
      --out-dir "$OUT_ROOT/quant_baseline" \
      "${COMMON_QUANT_ARGS[@]}"
fi

declare -A GHOST_MODE_MAP=([allk]=all [k1]=kernel1 [gt1]=gt1)

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
      "$PYTHON" "$ROOT/training/msasl/finetune.py" \
        "${COMMON_TRAIN_ARGS[@]}" \
        --pretrained "$BASELINE_CKPT" \
        --use-ghost-conv --ghost-ratio 2 --ghost-mode "$ghost_mode" \
        --out-dir "$ghost_dir"
  fi

  if [[ ! -f "$ghost_report" ]]; then
    run_logged "04_eval_ghost_${tag}" \
      "$PYTHON" "$ROOT/analysis/msasl_checkpoint_metrics.py" \
        --ckpt "$ghost_ckpt" --baseline-ckpt "$BASELINE_CKPT" \
        --out "$ghost_report" "${COMMON_EVAL_ARGS[@]}" \
        --use-ghost-conv --ghost-ratio 2 --ghost-mode "$ghost_mode"
    cleanup_temp_state_dicts
  fi

  if [[ ! -f "$quant_ghost_dir/summary.json" ]]; then
    run_logged "05_quant_ghost_${tag}" \
      "$PYTHON" "$ROOT/analysis/msasl_quantization_report.py" \
        --finetuned-ckpt "$ghost_ckpt" --out-dir "$quant_ghost_dir" \
        "${COMMON_QUANT_ARGS[@]}" \
        --use-ghost-conv --ghost-ratio 2 --ghost-mode "$ghost_mode"
  fi

  if [[ ! -f "$kd_ckpt" ]]; then
    run_logged "06_train_kd_ghost_${tag}" \
      "$PYTHON" "$ROOT/training/msasl/finetune_ghost_kd.py" \
        "${COMMON_KD_ARGS[@]}" \
        --student-ckpt "$ghost_ckpt" \
        --student-use-ghost-conv --student-ghost-ratio 2 --student-ghost-mode "$ghost_mode" \
        --out-dir "$kd_dir"
  fi

  if [[ ! -f "$kd_report_baseline" ]]; then
    run_logged "07_eval_kd_ghost_${tag}_vs_baseline" \
      "$PYTHON" "$ROOT/analysis/msasl_checkpoint_metrics.py" \
        --ckpt "$kd_ckpt" --baseline-ckpt "$BASELINE_CKPT" \
        --out "$kd_report_baseline" "${COMMON_EVAL_ARGS[@]}" \
        --use-ghost-conv --ghost-ratio 2 --ghost-mode "$ghost_mode"
    cleanup_temp_state_dicts
  fi

  if [[ ! -f "$kd_report_ghost" ]]; then
    run_logged "08_eval_kd_ghost_${tag}_vs_ghost" \
      "$PYTHON" "$ROOT/analysis/msasl_checkpoint_metrics.py" \
        --ckpt "$kd_ckpt" --baseline-ckpt "$ghost_ckpt" \
        --out "$kd_report_ghost" "${COMMON_EVAL_ARGS[@]}" \
        --use-ghost-conv --ghost-ratio 2 --ghost-mode "$ghost_mode" \
        --baseline-use-ghost-conv --baseline-ghost-ratio 2 --baseline-ghost-mode "$ghost_mode"
    cleanup_temp_state_dicts
  fi

  if [[ ! -f "$quant_kd_dir/summary.json" ]]; then
    run_logged "09_quant_kd_ghost_${tag}" \
      "$PYTHON" "$ROOT/analysis/msasl_quantization_report.py" \
        --finetuned-ckpt "$kd_ckpt" --out-dir "$quant_kd_dir" \
        "${COMMON_QUANT_ARGS[@]}" \
        --use-ghost-conv --ghost-ratio 2 --ghost-mode "$ghost_mode"
  fi
done

LOWRANK_CONFIGS=(
  "transformer|transformer|0.125|r0125"
  "transformer|transformer|0.25|r25"
  "transformer|transformer|0.50|r05"
  "transformer|transformer|0.75|r075"
  "all|transformer,conv,project_head|0.125|r0125"
  "all|transformer,conv,project_head|0.25|r25"
  "all|transformer,conv,project_head|0.50|r05"
  "all|transformer,conv,project_head|0.75|r075"
)

for config in "${LOWRANK_CONFIGS[@]}"; do
  IFS='|' read -r target_tag target_spec rank_ratio rank_tag <<< "$config"
  lowrank_dir="$OUT_ROOT/lowrank/${target_tag}_${rank_tag}"
  lowrank_ckpt="$lowrank_dir/best.pth.tar"
  lowrank_report="$lowrank_dir/report_vs_baseline.json"
  lowrank_quant_dir="$lowrank_dir/quantized"

  if [[ ! -f "$lowrank_ckpt" ]]; then
    run_logged "10_train_lowrank_${target_tag}_${rank_tag}" \
      "$PYTHON" "$ROOT/training/msasl/finetune_lowrank.py" \
        "${COMMON_TRAIN_ARGS[@]}" \
        --dense-ckpt "$BASELINE_CKPT" \
        --rank-ratio "$rank_ratio" --low-rank-targets "$target_spec" \
        --low-rank-min-features 64 --out-dir "$lowrank_dir"
  fi

  if [[ ! -f "$lowrank_report" ]]; then
    run_logged "11_eval_lowrank_${target_tag}_${rank_tag}" \
      "$PYTHON" "$ROOT/analysis/msasl_checkpoint_metrics.py" \
        --ckpt "$lowrank_ckpt" --baseline-ckpt "$BASELINE_CKPT" \
        --out "$lowrank_report" "${COMMON_EVAL_ARGS[@]}" \
        --use-low-rank --low-rank-ratio "$rank_ratio" \
        --low-rank-targets "$target_spec" --low-rank-min-features 64
    cleanup_temp_state_dicts
  fi

  if [[ ! -f "$lowrank_quant_dir/summary.json" ]]; then
    run_logged "12_quant_lowrank_${target_tag}_${rank_tag}" \
      "$PYTHON" "$ROOT/analysis/msasl_quantization_report.py" \
        --finetuned-ckpt "$lowrank_ckpt" --out-dir "$lowrank_quant_dir" \
        "${COMMON_QUANT_ARGS[@]}" \
        --use-low-rank --low-rank-ratio "$rank_ratio" \
        --low-rank-targets "$target_spec" --low-rank-min-features 64
  fi
done

echo
echo "All MSASL100 overnight experiments completed."
echo "Results root: $OUT_ROOT"
