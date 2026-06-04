#!/usr/bin/env bash
set -euo pipefail

export BENCH_DATA_CACHE="${BENCH_DATA_CACHE:-/home/dss-pc-05/bench/bench_data_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

PYTHON="${PYTHON:-/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python}"
SUITE="bench/configs/gpu_basilisk_me_split_full.yaml"
TASK="Basilisk_ADCS_sensor_noise_sweep_v0"

mkdir -p logs

for model in split_knet me_split_knet_v0; do
  for seed in 0 1 2; do
    log_path="logs/me_split_knet_full_${model}_seed${seed}.log"
    echo "[start] model=${model} seed=${seed} log=${log_path}"
    "${PYTHON}" -m bench.runners.run_suite \
      --suite-yaml "${SUITE}" \
      --tasks "${TASK}" \
      --models "${model}" \
      --seeds "${seed}" \
      --plans trained:frozen \
      --device cuda \
      --log-level DEBUG \
      --log-to-file \
      --debug-every 10 2>&1 | tee "${log_path}"
    echo "[done] model=${model} seed=${seed}"
  done
done
