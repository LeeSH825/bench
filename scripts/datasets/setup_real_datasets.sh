#!/usr/bin/env bash
set -euo pipefail

# Convenience setup for TG7 external datasets.
# Downloads a practical minimal subset from official sources, then prepares
# loader-ready NPZ files under external_data/.

NCLT_ROOT="external_data/nclt"
UZH_ROOT="external_data/uzh_fpv"
INSECURE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nclt-root)
      NCLT_ROOT="${2:?missing value for --nclt-root}"
      shift 2
      ;;
    --uzh-root)
      UZH_ROOT="${2:?missing value for --uzh-root}"
      shift 2
      ;;
    --insecure)
      INSECURE=1
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  setup_real_datasets.sh [--nclt-root DIR] [--uzh-root DIR] [--insecure]

Defaults:
  --nclt-root external_data/nclt
  --uzh-root  external_data/uzh_fpv
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

extra=()
if [[ "${INSECURE}" -eq 1 ]]; then
  extra+=(--insecure)
fi

scripts/datasets/download_nclt_groundtruth.sh --out-root "${NCLT_ROOT}" "${extra[@]}"
python3 scripts/datasets/prepare_nclt_groundtruth_npz.py --raw-root "${NCLT_ROOT}/raw/ground_truth" --out-root "${NCLT_ROOT}"

scripts/datasets/download_uzh_fpv_raw.sh --out-root "${UZH_ROOT}" --sequence indoor_forward_6 "${extra[@]}"
python3 scripts/datasets/prepare_uzh_fpv_leica_npz.py --zip-path "${UZH_ROOT}/raw/indoor_forward_6.zip" --out-root "${UZH_ROOT}"

cat <<EOF

Setup complete.
Use:
  export NCLT_ROOT="$(cd "${NCLT_ROOT}" && pwd)"
  export UZH_FPV_ROOT="$(cd "${UZH_ROOT}" && pwd)"

Then validate:
  .venv/bin/python -m bench.tasks.smoke_data --suite-yaml bench/configs/suite_tg7_datasets_smoke.yaml --task TG7_nclt_loader_smoke_v0 --seed 0
  .venv/bin/python -m bench.tasks.smoke_data --suite-yaml bench/configs/suite_tg7_datasets_smoke.yaml --task TG7_uzh_fpv_loader_smoke_v0 --seed 0
EOF
