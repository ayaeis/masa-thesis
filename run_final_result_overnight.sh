#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/masa-thesis"
ENV_ACTIVATE="/workspace/masa_env/bin/activate"
PRETRAINED="/workspace/checkpoints/pretrained_model.pth.tar"

OLD_DATA_ROOT="/workspace/datasets/preprocessed_wlasl100/WLASL"
OLD_BASELINE_CKPT="$ROOT/test_oldroot_baseline_rebuilt/best.pth.tar"

NEW_DATA_ROOT="/workspace/kaggle_wlasl100/masa_ready/WLASL"
NEW_BASELINE_CKPT="$ROOT/baseline_ckpt/best.pth.tar"

MASTER_OUT_ROOT="$ROOT/final_result_dual_protocol"

if [[ -f "$ENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_ACTIVATE"
fi

cd "$ROOT"

run_logged() {
  local log_dir="$1"
  local name="$2"
  shift 2
  local log_path="$log_dir/${name}.log"
  echo
  echo "[$(date '+%F %T')] START $name"
  "$@" 2>&1 | tee "$log_path"
  echo "[$(date '+%F %T')] DONE  $name"
}

ensure_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[error] missing required file: $path" >&2
    exit 1
  fi
}

ensure_file "$PRETRAINED"
ensure_file "$NEW_BASELINE_CKPT"

declare -A GHOST_MODE_MAP=(
  [allk]="all"
  [k1]="kernel1"
  [gt1]="gt1"
)

run_protocol() {
  local protocol_name="$1"
  local data_root="$2"
  local baseline_ckpt="$3"
  local ghost_init_ckpt="$4"

  local out_root="$MASTER_OUT_ROOT/$protocol_name"
  local report_dir="$out_root/reports"
  local log_dir="$out_root/logs"

  mkdir -p "$out_root" "$report_dir" "$log_dir"

  local common_train_args=(
    --data-root "$data_root"
    --pretrained "$ghost_init_ckpt"
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

  local common_eval_args=(
    --data-root "$data_root"
    --subset-num 100
    --target-t 32
    --batch-size 32
    --workers 8
    --num-class 100
    --dropout 0.0
    --warmup-steps 5
    --temporal-sampling index
  )

  local common_quant_args=(
    --data-root "$data_root"
    --subset-num 100
    --target-t 32
    --batch-size 32
    --workers 8
    --num-class 100
    --dropout 0.0
    --warmup-steps 5
    --temporal-sampling index
    --baseline-ckpt "$baseline_ckpt"
  )

  local common_kd_args=(
    --data-root "$data_root"
    --teacher-ckpt "$baseline_ckpt"
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

  echo
  echo "============================================================"
  echo "Protocol: $protocol_name"
  echo "Data root: $data_root"
  echo "Baseline ckpt: $baseline_ckpt"
  echo "Ghost init ckpt: $ghost_init_ckpt"
  echo "Results root: $out_root"
  echo "============================================================"

  local baseline_report="$report_dir/baseline_report.json"
  if [[ ! -f "$baseline_report" ]]; then
    run_logged "$log_dir" "01_baseline_report" \
      python "$ROOT/report_checkpoint_metrics.py" \
        --ckpt "$baseline_ckpt" \
        --out "$baseline_report" \
        "${common_eval_args[@]}"
  fi

  if [[ ! -f "$out_root/quant_baseline/summary.json" ]]; then
    run_logged "$log_dir" "02_quant_baseline" \
      python "$ROOT/quantize_finetuned_int8_fp16_report.py" \
        --finetuned-ckpt "$baseline_ckpt" \
        --out-dir "$out_root/quant_baseline" \
        "${common_quant_args[@]}"
  fi

  for tag in allk k1 gt1; do
    local ghost_mode="${GHOST_MODE_MAP[$tag]}"
    local ghost_dir="$out_root/ghost_${tag}"
    local ghost_ckpt="$ghost_dir/best.pth.tar"
    local ghost_report="$report_dir/ghost_${tag}_vs_baseline.json"
    local quant_ghost_dir="$out_root/quant_ghost_${tag}"
    local kd_dir="$out_root/kd_ghost_${tag}"
    local kd_ckpt="$kd_dir/best.pth.tar"
    local kd_report_baseline="$report_dir/kd_ghost_${tag}_vs_baseline.json"
    local kd_report_ghost="$report_dir/kd_ghost_${tag}_vs_ghost.json"
    local quant_kd_dir="$out_root/quant_kd_ghost_${tag}"

    if [[ ! -f "$ghost_ckpt" ]]; then
      run_logged "$log_dir" "03_train_ghost_${tag}" \
        python "$ROOT/finetune_wlasl100.py" \
          "${common_train_args[@]}" \
          --use-ghost-conv \
          --ghost-ratio 2 \
          --ghost-mode "$ghost_mode" \
          --out-dir "$ghost_dir"
    fi

    if [[ ! -f "$ghost_report" ]]; then
      run_logged "$log_dir" "04_eval_ghost_${tag}" \
        python "$ROOT/report_checkpoint_metrics.py" \
          --ckpt "$ghost_ckpt" \
          --baseline-ckpt "$baseline_ckpt" \
          --out "$ghost_report" \
          "${common_eval_args[@]}" \
          --use-ghost-conv \
          --ghost-ratio 2 \
          --ghost-mode "$ghost_mode"
    fi

    if [[ ! -f "$quant_ghost_dir/summary.json" ]]; then
      run_logged "$log_dir" "05_quant_ghost_${tag}" \
        python "$ROOT/quantize_finetuned_int8_fp16_report.py" \
          --finetuned-ckpt "$ghost_ckpt" \
          --out-dir "$quant_ghost_dir" \
          "${common_quant_args[@]}" \
          --use-ghost-conv \
          --ghost-ratio 2 \
          --ghost-mode "$ghost_mode"
    fi

    if [[ ! -f "$kd_ckpt" ]]; then
      run_logged "$log_dir" "06_train_kd_ghost_${tag}" \
        python "$ROOT/finetune_wlasl100_kd.py" \
          "${common_kd_args[@]}" \
          --student-ckpt "$ghost_ckpt" \
          --student-use-ghost-conv \
          --student-ghost-ratio 2 \
          --student-ghost-mode "$ghost_mode" \
          --out-dir "$kd_dir"
    fi

    if [[ ! -f "$kd_report_baseline" ]]; then
      run_logged "$log_dir" "07_eval_kd_ghost_${tag}_vs_baseline" \
        python "$ROOT/report_checkpoint_metrics.py" \
          --ckpt "$kd_ckpt" \
          --baseline-ckpt "$baseline_ckpt" \
          --out "$kd_report_baseline" \
          "${common_eval_args[@]}" \
          --use-ghost-conv \
          --ghost-ratio 2 \
          --ghost-mode "$ghost_mode"
    fi

    if [[ ! -f "$kd_report_ghost" ]]; then
      run_logged "$log_dir" "08_eval_kd_ghost_${tag}_vs_ghost" \
        python "$ROOT/report_checkpoint_metrics.py" \
          --ckpt "$kd_ckpt" \
          --baseline-ckpt "$ghost_ckpt" \
          --out "$kd_report_ghost" \
          "${common_eval_args[@]}" \
          --use-ghost-conv \
          --ghost-ratio 2 \
          --ghost-mode "$ghost_mode" \
          --baseline-use-ghost-conv \
          --baseline-ghost-ratio 2 \
          --baseline-ghost-mode "$ghost_mode"
    fi

    if [[ ! -f "$quant_kd_dir/summary.json" ]]; then
      run_logged "$log_dir" "09_quant_kd_ghost_${tag}" \
        python "$ROOT/quantize_finetuned_int8_fp16_report.py" \
          --finetuned-ckpt "$kd_ckpt" \
          --out-dir "$quant_kd_dir" \
          "${common_quant_args[@]}" \
          --use-ghost-conv \
          --ghost-ratio 2 \
          --ghost-mode "$ghost_mode"
    fi
  done

  echo
  echo "[done] protocol completed: $protocol_name"
  echo "results: $out_root"
}

rebuild_old_baseline_if_missing() {
  if [[ -f "$OLD_BASELINE_CKPT" ]]; then
    echo "[ok] old baseline checkpoint exists: $OLD_BASELINE_CKPT"
    return 0
  fi

  local bootstrap_root="$MASTER_OUT_ROOT/bootstrap_old_baseline"
  local bootstrap_log_dir="$bootstrap_root/logs"
  mkdir -p "$bootstrap_log_dir"

  echo
  echo "[bootstrap] old baseline checkpoint missing"
  echo "[bootstrap] rebuilding old baseline at: $OLD_BASELINE_CKPT"

  run_logged "$bootstrap_log_dir" "00_rebuild_old_baseline" \
    python "$ROOT/finetune_wlasl100.py" \
      --data-root "$OLD_DATA_ROOT" \
      --pretrained "$PRETRAINED" \
      --num-class 100 \
      --subset-num 100 \
      --epochs 60 \
      --batch-size 64 \
      --workers 8 \
      --optim sgd \
      --momentum 0.9 \
      --lr 0.01 \
      --weight-decay 0.0 \
      --scheduler multistep \
      --milestones 20 40 \
      --lr-gamma 0.1 \
      --target-t 32 \
      --temporal-sampling index \
      --freeze-epochs 0 \
      --head-lr-mult 1.0 \
      --warmup-epochs 0 \
      --train-temporal-crop-min 1.0 \
      --patience 999 \
      --min-delta 0.0 \
      --seed 123 \
      --out-dir "$ROOT/test_oldroot_baseline_rebuilt"

  ensure_file "$OLD_BASELINE_CKPT"
}

rebuild_old_baseline_if_missing

run_protocol \
  "oldroot_pretrained" \
  "$OLD_DATA_ROOT" \
  "$OLD_BASELINE_CKPT" \
  "$PRETRAINED"

run_protocol \
  "newroot_newbaseline" \
  "$NEW_DATA_ROOT" \
  "$NEW_BASELINE_CKPT" \
  "$NEW_BASELINE_CKPT"

echo
echo "All dual-protocol overnight experiments completed."
echo "Results root: $MASTER_OUT_ROOT"
