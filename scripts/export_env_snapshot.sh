#!/usr/bin/env bash
set -euo pipefail

# Export an environment snapshot into requirements.lock (pip freeze)
# Usage:
#   cd /mnt/data/bench
#   bash scripts/export_env_snapshot.sh
#
# This is a helper for Step 3 single-env strategy.
# The run-time snapshot is additionally written per-run into run_dir by bench utils.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_FILE="${ROOT_DIR}/requirements.lock"

echo "# requirements.lock (snapshot)" > "${OUT_FILE}"
echo "# generated_at: $(date -Iseconds)" >> "${OUT_FILE}"
echo "# python: $(python -c 'import sys; print(sys.version.replace(\"\\n\",\" \"))')" >> "${OUT_FILE}"
echo "" >> "${OUT_FILE}"

python -m pip freeze >> "${OUT_FILE}"

echo "Wrote ${OUT_FILE}"

