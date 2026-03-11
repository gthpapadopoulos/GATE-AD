#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "$ROOT_DIR/scripts/run_mvtec_default.sh"
bash "$ROOT_DIR/scripts/run_visa_default.sh"
bash "$ROOT_DIR/scripts/run_mpdd_default.sh"
