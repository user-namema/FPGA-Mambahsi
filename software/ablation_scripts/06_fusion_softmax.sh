#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# S3b: replace direct branch sum with learned two-scalar softmax fusion.
run_current_ablation \
    --fusion_mode softmax \
    "$@"
