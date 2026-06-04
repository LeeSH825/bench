#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export BENCH_DATA_CACHE="${BENCH_DATA_CACHE:-/home/dss-pc-05/bench/bench_data_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

PYTHON="${PYTHON:-/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python}"
SUITE_YAML="bench/configs/gpu_basilisk_structured_corruption_full.yaml"
SUITE_NAME="gpu_basilisk_structured_corruption_full"
TASK_ID="Basilisk_ADCS_structured_corruption_v0"
LOG_DIR="logs/structured_corruption_full"
mkdir -p "${LOG_DIR}"

chunk_complete() {
  local model="$1"
  local seed="$2"
  "${PYTHON}" - "$SUITE_NAME" "$TASK_ID" "$model" "$seed" <<'PY'
import json
import math
import sys
from pathlib import Path

suite, task, model, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
root = Path("runs") / suite / task / model / "frozen" / f"seed_{seed}"
metrics_paths = sorted(root.glob("scenario_*/metrics.json"))
if len(metrics_paths) != 4:
    raise SystemExit(1)
for metrics_path in metrics_paths:
    run_dir = metrics_path.parent
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        raise SystemExit(1)
    mse = float(metrics.get("mse", metrics.get("accuracy", {}).get("mse", "nan")))
    mse_db = float(metrics.get("mse_db", metrics.get("accuracy", {}).get("mse_db", "nan")))
    expected = 10.0 * math.log10(max(mse, 1.0e-300))
    if run_dir.joinpath("failure.json").exists():
        raise SystemExit(1)
    if str(metrics.get("device_resolved", "")) != "cuda":
        raise SystemExit(1)
    if not math.isfinite(mse_db) or abs(mse_db - expected) >= 1.0e-8:
        raise SystemExit(1)
    if not run_dir.joinpath("diagnostics", "stats.json").exists():
        raise SystemExit(1)
raise SystemExit(0)
PY
}

run_chunk() {
  local model="$1"
  local seed="$2"
  local log_path="${LOG_DIR}/${model}_seed${seed}.log"

  if chunk_complete "${model}" "${seed}"; then
    echo "[$(date -Is)] skip complete chunk model=${model} seed=${seed}" | tee -a "${log_path}"
    return 0
  fi

  echo "[$(date -Is)] start chunk model=${model} seed=${seed}" | tee -a "${log_path}"
  "${PYTHON}" -m bench.runners.run_suite \
    --suite-yaml "${SUITE_YAML}" \
    --models "${model}" \
    --seeds "${seed}" \
    --plans trained:frozen \
    --device cuda \
    --log-level DEBUG \
    --log-to-file \
    --debug-every 10 2>&1 | tee -a "${log_path}"
  echo "[$(date -Is)] finished chunk model=${model} seed=${seed}" | tee -a "${log_path}"
}

for model in split_knet me_split_knet_v0; do
  for seed in 0 1 2; do
    run_chunk "${model}" "${seed}"
  done
done

"${PYTHON}" scripts/audit_structured_corruption_runs.py \
  --suite-name "${SUITE_NAME}" \
  --profiles clean_gaussian mild moderate severe \
  --seeds 0 1 2 \
  --expected-device cuda \
  --train-updates 500 \
  --prefix me_split_structured_corruption_full
