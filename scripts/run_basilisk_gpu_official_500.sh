#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/dss-pc-05/bench"
PYTHON="/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python"
SUITE_YAML="bench/configs/gpu_basilisk_adcs_official.yaml"
TASK_ID="Basilisk_ADCS_sensor_noise_sweep_v0"
PLAN="trained:frozen"

export BENCH_DATA_CACHE="/home/dss-pc-05/bench/bench_data_cache"
export MPLCONFIGDIR="/tmp/matplotlib"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

cd "${ROOT_DIR}"
mkdir -p logs

SCENARIO_IDS=(
  "37bd751afdc0"  # sensor_noise_scale_db=-10
  "4c70131fb32a"  # sensor_noise_scale_db=0
  "b2da60a0830d"  # sensor_noise_scale_db=10
  "4d7e2a5202c0"  # sensor_noise_scale_db=20
  "ef19946c6066"  # sensor_noise_scale_db=30
)
MODELS=("kalmannet_tsp" "split_knet")
SEEDS=(0 1 2)

is_complete() {
  local model="$1"
  local seed="$2"
  local scenario_id="$3"
  local run_dir="runs/gpu_basilisk_adcs_official/${TASK_ID}/${model}/frozen/seed_${seed}/scenario_${scenario_id}"
  set +e
  "${PYTHON}" - "${run_dir}" <<'PY'
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
metrics_path = run_dir / "metrics.json"
failure_path = run_dir / "failure.json"
stats_path = run_dir / "diagnostics" / "stats.json"
if failure_path.exists() or not metrics_path.exists() or not stats_path.exists():
    raise SystemExit(1)

metrics = json.loads(metrics_path.read_text())
accuracy = metrics.get("accuracy", {})
budgets = metrics.get("budgets", {})
plan = metrics.get("run_plan", {})
mse = float(accuracy["mse"])
mse_db = float(accuracy["mse_db"])
if abs(mse_db - 10.0 * math.log10(mse)) >= 1e-8:
    raise SystemExit(1)
if plan.get("device_resolved") != "cuda":
    raise SystemExit(1)
if int(budgets.get("train_max_updates", -1)) != 500:
    raise SystemExit(1)
if int(budgets.get("adapt_updates_used", 0)) != 0:
    raise SystemExit(1)

stats = json.loads(stats_path.read_text())
residual = stats.get("residual_stats") or {}
if not residual.get("finite", False):
    raise SystemExit(1)
if int(residual.get("nan_count") or 0) != 0 or int(residual.get("inf_count") or 0) != 0:
    raise SystemExit(1)
PY
  local status=$?
  set -e
  return "${status}"
}

run_one_scenario() {
  local model="$1"
  local seed="$2"
  local scenario_id="$3"
  local chunk_log="logs/basilisk_gpu_official_500_${model}_seed${seed}.log"

  if [[ "${FORCE}" -eq 0 ]] && is_complete "${model}" "${seed}" "${scenario_id}"; then
    echo "[$(date -Is)] SKIP complete model=${model} seed=${seed} scenario=${scenario_id}" | tee -a "${chunk_log}"
    return 0
  fi

  echo "[$(date -Is)] RUN model=${model} seed=${seed} scenario=${scenario_id}" | tee -a "${chunk_log}"
  "${PYTHON}" -m bench.runners.run_suite \
    --suite-yaml "${SUITE_YAML}" \
    --tasks "${TASK_ID}" \
    --scenario-ids "${scenario_id}" \
    --models "${model}" \
    --seeds "${seed}" \
    --plans "${PLAN}" \
    --device cuda \
    --log-level DEBUG \
    --log-to-file \
    --debug-every 10 2>&1 | tee -a "${chunk_log}"
}

echo "[$(date -Is)] Basilisk GPU official budget=500 start force=${FORCE}"
echo "[$(date -Is)] chunks=6 scenario_runs=30 suite=${SUITE_YAML} task=${TASK_ID}"

for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "[$(date -Is)] CHUNK start model=${model} seed=${seed}" | tee -a "logs/basilisk_gpu_official_500_${model}_seed${seed}.log"
    for scenario_id in "${SCENARIO_IDS[@]}"; do
      run_one_scenario "${model}" "${seed}" "${scenario_id}"
    done
    echo "[$(date -Is)] CHUNK done model=${model} seed=${seed}" | tee -a "logs/basilisk_gpu_official_500_${model}_seed${seed}.log"
  done
done

echo "[$(date -Is)] Basilisk GPU official budget=500 done"
