#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
python -m gate_ad.cli.run_defaults --config "$ROOT_DIR/configs/defaults/visa.yaml" "$@"
