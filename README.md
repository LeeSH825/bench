# AI-ADCS Bench

AI-ADCS Bench is a reproducible benchmark and control toolkit for KalmanNet
family estimators, classical KF/MEKF baselines, and spacecraft-attitude data.
It provides deterministic data generation, model adapters, suite execution,
reports, checkpoint-aware run control, and post-hoc visualization.

The repository feature map and portability boundary are documented in
[`docs/FEATURE_INVENTORY.md`](docs/FEATURE_INVENTORY.md).

## Install

The supported Python floor is 3.10. Clone the pinned upstream model
submodules when you need KalmanNet-family adapters.

The current Git history remains large even though generated products are no
longer tracked. A shallow clone avoids transferring the old payload history:

```bash
git clone --depth 1 --recurse-submodules --shallow-submodules \
  <repository-url> bench
cd bench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Optional surfaces are installed explicitly:

```bash
python -m pip install -e '.[control]'   # FastAPI, Dash, telemetry
python -m pip install -e '.[viz]'       # Streamlit Run Inspector
python -m pip install -e '.[research]'  # research schema validation
python -m pip install -e '.[basilisk]'  # Basilisk-backed generators
```

Install the appropriate PyTorch build for the target CPU/CUDA platform when
the default PyPI resolution is not suitable.

## Smallest benchmark path

All paths below are repository-relative; no external `/mnt/data` tree is
required.

```bash
export BENCH_DATA_CACHE="${PWD}/bench_data_cache"

bench-smoke-data \
  --suite-yaml bench/configs/suite_kf_baseline_smoke.yaml \
  --task A_linear_kf_baseline_smoke_v0 \
  --seed 0

bench-run-suite \
  --suite-yaml bench/configs/suite_kf_baseline_smoke.yaml \
  --tasks A_linear_kf_baseline_smoke_v0 \
  --models oracle_kf \
  --seeds 0 \
  --plans pretrained:frozen \
  --device cpu

bench-make-report \
  --suite-yaml bench/configs/suite_kf_baseline_smoke.yaml
```

Generated datasets, run records, and reports are written under
`bench_data_cache/`, `runs/`, and `reports/`. They are intentionally ignored
by Git; canonical research evidence remains under `experiments/`.

## Control plane

The versioned control plane includes:

- immutable run allocation, worker supervision, registry, event journal, and
  CPU/GPU telemetry;
- checkpoint catalog inspection and validation;
- persistent graceful-stop requests;
- exact-resume planning and immutable resumed-child launch within the
  explicitly certified execution envelope;
- configuration preset browse, validation, hash preview, and launch;
- FastAPI and Dash observation surfaces.

CLI entry points:

```bash
bench-control --help
bench-control checkpoints --help
bench-control stop --help
bench-control resume --help
bench-control-api --host 127.0.0.1 --port 8765
bench-dashboard --host 127.0.0.1 --port 8766 \
  --api http://127.0.0.1:8765
```

The API and dashboard are read-only by default. To register the guarded HTTP
launch/stop/resume routes and enable the config-driven New Run page, both
processes must be started with explicit local write mode:

```bash
export BENCH_CONTROL_ENABLE_WRITES=1
bench-control-api --host 127.0.0.1 --port 8765
bench-dashboard --host 127.0.0.1 --port 8766 \
  --api http://127.0.0.1:8765
```

Write mode refuses non-loopback binds. The control plane has no
authentication, so do not expose it directly to a network. The exact-resume
capability endpoint reports the certified and uncertified envelopes; absence
from the certification matrix means unsupported, not best-effort support.

## Offline visualization

```bash
python -m streamlit run viz/app/main.py
```

The Run Inspector reads completed artifacts and does not replace the control
dashboard. ADCS replay and Vizard contract/tooling modules are included in the
portable wheel. Historical Phase labels, mocks, and identity baselines establish
structural support only; real KalmanNet package replay, Basilisk/native Vizard
conversion, and manual frame/sign review remain explicit environment gates.

## Packaging verification

Editable installs can accidentally import untracked local files. The release
check builds only `HEAD` from a Git archive, installs the wheel outside the
repository, verifies the full benchmark/control/config/viz surface, and runs
every public CLI help path:

The release verifier requires the standard-library `venv` module (the
`python3-venv` OS package on Debian/Ubuntu) and the development plus control
extras in the invoking environment. It reuses those installed dependencies,
but requires `bench` and `viz` themselves to import from the clean wheel:

```bash
python -m pip install -e '.[dev,control]'
```

```bash
python scripts/verify_clean_wheel.py
```

For an already-built wheel:

```bash
python scripts/verify_portable_wheel.py dist/bench-*.whl
```

The wheel contains `bench`, its shipped suite YAML files, and `viz`. It does
not contain run output, report output, data caches, or research evidence.

## Research boundaries

`DECISIONS.md`, `FAIRNESS.md`, and `METRICS.md` govern the benchmark contract.
In a governed research checkout, the checkout's `AGENTS.md` and active
control/state files also apply. Installing the package or passing a smoke test
does not authorize a research stage, open sealed evaluation, or turn an
execution result into scientific evidence.
