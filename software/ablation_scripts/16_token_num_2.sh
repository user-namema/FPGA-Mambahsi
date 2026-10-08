#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# E4a: reduce SpeMamba spectral tokens from 4 to 2.
run_current_ablation \
    --token_num 2 \
    "$@"
