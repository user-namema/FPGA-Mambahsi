#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# E4b: increase SpeMamba spectral tokens from 4 to 8.
run_current_ablation \
    --token_num 8 \
    "$@"
