#!/usr/bin/env bash
set -euo pipefail

# Download minimal UZH-FPV raw package used by TG7 prep script.
#
# Example:
#   scripts/datasets/download_uzh_fpv_raw.sh --out-root external_data/uzh_fpv --sequence indoor_forward_6

OUT_ROOT="external_data/uzh_fpv"
SEQUENCE="indoor_forward_6"
INSECURE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-root)
      OUT_ROOT="${2:?missing value for --out-root}"
      shift 2
      ;;
    --sequence)
      SEQUENCE="${2:?missing value for --sequence}"
      shift 2
      ;;
    --insecure)
      INSECURE=1
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  download_uzh_fpv_raw.sh [--out-root DIR] [--sequence NAME] [--insecure]

Defaults:
  --out-root external_data/uzh_fpv
  --sequence indoor_forward_6

Downloads:
  ${OUT_ROOT}/raw/<sequence>.zip
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

RAW_DIR="${OUT_ROOT}/raw"
mkdir -p "${RAW_DIR}"

url="http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/raw/${SEQUENCE}.zip"
out="${RAW_DIR}/${SEQUENCE}.zip"
tmp="${out}.part"

if [[ -f "${out}" ]]; then
  echo "[skip] ${out} already exists"
else
  curl_flags=(-fL --retry 3 --retry-delay 2 --connect-timeout 20)
  if [[ "${INSECURE}" -eq 1 ]]; then
    curl_flags=(-k "${curl_flags[@]}")
  fi
  echo "[download] ${url}"
  curl "${curl_flags[@]}" -C - -o "${tmp}" "${url}"
  mv -f "${tmp}" "${out}"
  echo "[ok] ${out}"
fi

cat <<EOF

Done.
Next:
  python3 scripts/datasets/prepare_uzh_fpv_leica_npz.py --zip-path "${out}" --out-root "${OUT_ROOT}"

Then export:
  export UZH_FPV_ROOT="$(cd "${OUT_ROOT}" && pwd)"
EOF
