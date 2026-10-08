#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# S0: exact current FPGA-oriented model.
run_current_ablation "$@"
