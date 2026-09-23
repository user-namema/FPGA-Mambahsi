#!/usr/bin/env bash
set -euo pipefail
# Keep this directory intact; do not overwrite the old server QAT modules.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
qat_run="${1:-/home/music/mzz/MambaHSI/results/SPATIAL_SPLIT_3WAY_DENSE_QAT/UP_D1_sharedU_D8_dtin9_dtout8/current_h32_br-both_fu-sum_sk2_z0_D1_A-shared_n-bn_a-relu_head64_tok4_state16/run_seed0}"
data_dir="${2:-/home/music/mzz/MambaHSI/data}"
output_dir="${3:-/home/music/mzz/MambaHSI/sim_nonlinear_D1_run1}"
python -u "${script_dir}/both_FPGA_nonlinear_D1.py" \
  --qat-run-dir "${qat_run}" \
  --dataset UP \
  --data-path "${data_dir}" \
  --device cuda \
  --nonlinear-backend all \
  --trace-tiles 5 \
  --output-dir "${output_dir}"
