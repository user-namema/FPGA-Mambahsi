#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Sref-full: batch-safe original architecture at hidden64/head128.
run_original_full_reference "$@"
