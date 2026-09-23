#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
config=current_h32_br-both_fu-sum_sk2_z0_D1_A-shared_n-bn_a-relu_head64_tok4_state16
fp32_root="${FP32_ROOT:-./results/SPATIAL_SPLIT_3WAY_DENSE}"
fp32_tag="${FP32_TAG:-all_samples_sqrt_inverse_clip3_2000_nobias}"
extra=()
[[ "${DRY_RUN:-0}" == 1 ]] && extra+=(--dry-run)
[[ "${ALLOW_DISPLAY_PROCESSES:-0}" == 1 ]] && extra+=(--allow-display-processes)
[[ -n "${QAT_RUN_DIR:-}" ]] && extra+=(--qat-run-dir "$QAT_RUN_DIR")
"${PYTHON:-python}" -u benchmark_up_gpu_power.py \
  --fp32-dir "${FP32_UP:-$fp32_root/UP_$fp32_tag/$config}" \
  --data-path "${DATA_ROOT:-./data}" --device "${DEVICE:-cuda:0}" --seed "${SEED:-0}" \
  --model-kind "${MODEL_KIND:-fp32}" --batch-sizes "${BATCH_SIZES:-1,2,4,8,16,32,64}" \
  --seconds "${MEASURE_SECONDS:-30}" --warmup-seconds "${WARMUP_SECONDS:-10}" \
  --idle-seconds "${IDLE_SECONDS:-5}" --repeats "${REPEATS:-3}" --sample-ms "${SAMPLE_MS:-100}" \
  --output-dir "${OUTPUT_ROOT:-./gpu_power_UP_fp32}" "${extra[@]}" "$@"
