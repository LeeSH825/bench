# TG7 Dataset Loader Scaffolding

This benchmark provides external-dataset loader scaffolding for:

- `bench.tasks.generator.datasets.nclt` (NCLT)
- `bench.tasks.generator.datasets.uzh_fpv` (UZH-FPV)

The loaders keep raw datasets external and convert prepared session arrays into canonical D15 artifacts (`train.npz`, `val.npz`, `test.npz`) with `x/y` in NTD (`[N,T,D]`).

## Environment Variables

- `NCLT_ROOT`: root directory containing prepared NCLT session NPZ files.
- `UZH_FPV_ROOT`: root directory containing prepared UZH-FPV session NPZ files.

If an env var is missing or invalid, loaders raise `DatasetMissingError` with:

- `error_code = "io_error"`
- dataset name
- env var name
- expected directory layout hints

This keeps CI stable when datasets are absent.

## Expected Prepared Layout (Examples)

NCLT:

- `$NCLT_ROOT/prepared/2012-01-22.npz`
- `$NCLT_ROOT/prepared/2012-04-29.npz`
- or `$NCLT_ROOT/sessions/<session>/xy.npz`

UZH-FPV:

- `$UZH_FPV_ROOT/prepared/6th_indoor_forward_facing.npz`
- or `$UZH_FPV_ROOT/sequences/6th_indoor_forward_facing/xy.npz`

Each NPZ should include `x` and `y` arrays (or accepted aliases such as `states`/`observations`), either as:

- rank-2 continuous arrays `[T,D]`
- or rank-3 pre-windowed arrays `[N,T,D]`

## Paper-Aligned Split Metadata

NCLT encodes protocol metadata in `meta.splits`:

- Split-KalmanNet 2023 protocol (default):
  - train: session `2012-01-22`, `L=80`, `T=50`
  - val: session `2012-01-22`, `L=5`, `T=200`
  - test: session `2012-04-29`, `L=1`, `T=2000`
- KalmanNet 2022 protocol (optional):
  - train `L=23`, `T=200`; val `L=2`, `T=200`; test `L=1`, `T=277`

UZH-FPV encodes MAML-KalmanNet 2025 protocol metadata:

- session `6th indoor forward-facing`
- total steps `3020` at `100 Hz`
- train `L=25`, `T=80`
- test `L=1`, `T=1020`

## Local Cache Build Commands

### 1) Download + prepare official minimal subsets

You can bootstrap both datasets with:

```bash
scripts/datasets/setup_real_datasets.sh --insecure
```

Or run per dataset:

```bash
scripts/datasets/download_nclt_groundtruth.sh --out-root external_data/nclt --insecure
python3 scripts/datasets/prepare_nclt_groundtruth_npz.py --raw-root external_data/nclt/raw/ground_truth --out-root external_data/nclt
```

```bash
scripts/datasets/download_uzh_fpv_raw.sh --out-root external_data/uzh_fpv --sequence indoor_forward_6 --insecure
python3 scripts/datasets/prepare_uzh_fpv_leica_npz.py --zip-path external_data/uzh_fpv/raw/indoor_forward_6.zip --out-root external_data/uzh_fpv
```

`--insecure` is optional and only needed on hosts with TLS certificate-chain issues.

### 2) Export dataset roots

```bash
export NCLT_ROOT=$PWD/external_data/nclt
export UZH_FPV_ROOT=$PWD/external_data/uzh_fpv
```

### 3) Run TG7 smoke conversions

NCLT:

```bash
NCLT_ROOT=/path/to/nclt \
.venv/bin/python -m bench.tasks.smoke_data \
  --suite-yaml bench/configs/suite_tg7_datasets_smoke.yaml \
  --task TG7_nclt_loader_smoke_v0 \
  --seed 0
```

UZH-FPV:

```bash
UZH_FPV_ROOT=/path/to/uzh_fpv \
.venv/bin/python -m bench.tasks.smoke_data \
  --suite-yaml bench/configs/suite_tg7_datasets_smoke.yaml \
  --task TG7_uzh_fpv_loader_smoke_v0 \
  --seed 0
```
