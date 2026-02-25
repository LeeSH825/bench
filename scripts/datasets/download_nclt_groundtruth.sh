#!/usr/bin/env bash
set -euo pipefail

# Download minimal NCLT assets used by TG7 prep:
# - groundtruth_<session>.csv
#
# Example:
#   scripts/datasets/download_nclt_groundtruth.sh --out-root external_data/nclt --insecure

OUT_ROOT="external_data/nclt"
INSECURE=0
SESSIONS=("2012-01-22" "2012-04-29")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-root)
      OUT_ROOT="${2:?missing value for --out-root}"
      shift 2
      ;;
    --session)
      SESSIONS+=("${2:?missing value for --session}")
      shift 2
      ;;
    --insecure)
      INSECURE=1
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  download_nclt_groundtruth.sh [--out-root DIR] [--session YYYY-MM-DD] [--insecure]

Defaults:
  --out-root external_data/nclt
  --session  2012-01-22
  --session  2012-04-29

Downloads:
  ${OUT_ROOT}/raw/ground_truth/groundtruth_<session>.csv
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

RAW_DIR="${OUT_ROOT}/raw/ground_truth"
mkdir -p "${RAW_DIR}"

curl_flags=(-fL --retry 3 --retry-delay 2 --connect-timeout 20)
if [[ "${INSECURE}" -eq 1 ]]; then
  curl_flags=(-k "${curl_flags[@]}")
fi

for session in "${SESSIONS[@]}"; do
  url="https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/ground_truth/groundtruth_${session}.csv"
  out="${RAW_DIR}/groundtruth_${session}.csv"
  tmp="${out}.part"
  if [[ -f "${out}" ]]; then
    echo "[skip] ${out} already exists"
    continue
  fi
  echo "[download] ${url}"
  curl "${curl_flags[@]}" -C - -o "${tmp}" "${url}"
  mv -f "${tmp}" "${out}"
  echo "[ok] ${out}"
done

cat <<EOF

Done.
Next:
  python3 scripts/datasets/prepare_nclt_groundtruth_npz.py --raw-root "${OUT_ROOT}/raw/ground_truth" --out-root "${OUT_ROOT}"

Then export:
  export NCLT_ROOT="$(cd "${OUT_ROOT}" && pwd)"
EOF
