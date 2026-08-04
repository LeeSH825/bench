# Implementation Baseline — Phase 0/1/3 Foundation

Captured before any implementation change in this tranche.

- **Capture date:** 2026-07-30
- **Repository root:** `/home/dss-pc-05/bench`
  (the task brief assumed `/home/sung-lee/bench`; the actual checkout lives under
  `/home/dss-pc-05/bench` — all paths in this document are relative to the real root)
- **Audit reference commit:** `d1cc4b035597bb029e0bce95a546b29b4664b5c6`
- **Reference audit document:** `docs/benchmark_gui_current_state_audit.md`

---

## 1. Git state

| Item | Value |
|---|---|
| Branch | `main` (tracking `origin/main`) |
| HEAD | `d1cc4b035597bb029e0bce95a546b29b4664b5c6` |
| HEAD subject | `Add in-app user guide and contextual help to Run Inspector` |
| HEAD date | 2026-07-30 11:51:56 +0900 |
| HEAD == audit commit | **yes** |

### 1.1 Working tree (dirty, preserved as-is)

`git status --porcelain` reported **385 entries** at capture time:

| Status | Count | Notes |
|---:|---:|---|
| `M` (modified) | 116 | Mostly `reports/**` artifacts and `bench/**/__pycache__/*.pyc`; also `DECISIONS.md`, `USERGUIDE.md`, `bench/models/{adaptive_knet,maml_knet,registry}.py`, `bench/tasks/bench_generated.py`, `bench/tasks/generator/basilisk_imu_adcs.py`, `bench/tests/{run_all,test_kf_baseline_smoke_plan,test_runner_smoke}.py` |
| `D` (deleted) | 185 | Tracked run artifacts under `runs/**` deleted from the working tree (config snapshots, `model.pt`, `metrics.json`, `metrics_step.csv`, `run_plan.json`, `env.json`, …) |
| `??` (untracked) | 84 | Research/planning markdown at repo root, many `bench/configs/suite_basilisk_*.yaml`, and the design-doc bundle under `docs/benchmark_visualization/` |

Tracked-diff summary excluding `*.pyc`: **263 files changed, 1942 insertions(+), 4672 deletions(-)**.

> **This dirty state is pre-existing and was left untouched.** In particular the 185
> deleted run artifacts were **not** restored and **not** committed. Nothing under
> `runs/`, `bench_data_cache/`, or `reports/` was modified, moved, or deleted by this
> tranche.

### 1.2 Submodules

| Submodule | Revision | Dirty |
|---|---|---|
| `third_party/Adaptive-KNet-ICASSP24` | `acff2a65f6627139f53331ac4c5ce6741fae4c90` (heads/main) | no |
| `third_party/KalmanNet_TSP` | `828a2cf529bc84f43b37d543d916fe5858054457` (heads/main) | **yes** |
| `third_party/MAML_KalmanNet` | `01834cd3a03a31e0e5446c373d7efbd81308ce60` (heads/master) | **yes** |
| `third_party/Split_KalmanNet` | `0d6265668e58e6f934a09212b465a82666e544a6` (heads/main) | no |

Dirty content in both cases is **untracked `__pycache__` byte-code only** — no source
modification:

```
third_party/KalmanNet_TSP:     ?? KNet/__pycache__/  ?? Simulations/__pycache__/
third_party/MAML_KalmanNet:    ?? MAML-KalmanNet/__pycache__/filter.cpython-310.pyc
                               ?? MAML-KalmanNet/__pycache__/state_dict_learner.cpython-310.pyc
```

This closes audit risk **R-18** (third-party dirty revision unrecorded) for this
snapshot: the revisions are pinned above and the dirt is byte-code, not source.
No third-party file was modified by this tranche, so no `TP-xxx` exception record
is required.

---

## 2. Environment

### 2.1 Interpreter (important correction to the audit)

The repository contains a `.venv/` directory, but it is **stale and empty**:

| Interpreter | Version | Has project deps |
|---|---|---|
| `/home/dss-pc-05/bench/.venv/bin/python` | 3.13.12 | **no** — `torch`, `numpy`, `yaml`, … all missing |
| `/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python` | 3.10.13 | **yes** — this is the real working interpreter |

Evidence: every `__pycache__` artifact in the tree is `cpython-310`. The audit's
statement "pytest is not installed in `.venv`" is accurate but misleading — `.venv`
has *nothing* installed; the environment actually in use is the pyenv 3.10.13
site-packages.

**All commands in this tranche use `/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python`.**

### 2.2 Runtime

| Item | Value |
|---|---|
| Python | 3.10.13 (pyenv) |
| PyTorch | 2.9.1+cu128 |
| `torch.version.cuda` | 12.8 |
| `torch.cuda.is_available()` | `True` |
| GPU 0 | NVIDIA GeForce RTX 4060, 8188 MiB, UUID `GPU-c6b24f76-344f-2d98-2516-7c2f2287eb19` |
| NVIDIA driver | 610.74 |
| Host | Linux 5.15.167.4-microsoft-standard-WSL2 |
| CPU / RAM | 16 logical cores / 30 GiB total, 28 GiB available |
| Disk (repo fs) | 1007 G total, 90 G used, 867 G free (10 %) |

Single GPU only — consistent with the "no multi-GPU queue" scope exclusion.

### 2.3 Dependency source

`pyproject.toml` (setuptools backend) is the declared source; `uv.lock` (511 KB) and
`requirements.lock` also exist, and `uv 0.10.11` is installed. However the live
environment is **not** a `uv`-synced venv — packages are installed directly into pyenv
3.10.13 site-packages, and installed versions do not match `pyproject.toml` pins
(`torch` is 2.9.1+cu128 while `[[tool.uv.index]]` pins the cu121 index). The lockfiles
are therefore **already stale relative to the live environment before this tranche**.

### 2.4 Packages present before this tranche

`torch 2.9.1+cu128`, `numpy 2.2.6`, `pandas 2.3.3`, `scipy 1.15.3`,
`matplotlib 3.10.7`, `pyyaml 6.0.3`, `streamlit 1.60.0`, `plotly 6.9.0`,
`uvicorn 0.51.0`.

Absent: `pytest`, `psutil`, `fastapi`, `dash`, `pydantic`, `httpx`, `pynvml`.

### 2.5 Dependency changes made by this tranche

Two installs, both **purely additive** — verified with `pip install --dry-run` first,
and no already-installed package was upgraded or downgraded:

1. **`pytest>=7.4`** — already declared in `pyproject.toml` `[project.optional-dependencies].dev`
   but not installed. Installing it is required by acceptance gate **B-01**.
   Added: `pytest 9.1.1`, `pluggy 1.6.0`, `iniconfig 2.3.0`, `tomli 2.4.1`, `Pygments 2.20.0`.

2. **Control-plane runtime deps** (new, required by this tranche's scope):
   `fastapi 0.141.1`, `dash 3.4.0`, `psutil 7.2.2`, `pydantic 2.13.4`,
   `httpx 0.28.1`, `nvidia-ml-py 13.610.43`.
   Transitively added: `Flask 3.1.3`, `Werkzeug 3.1.8`, `pydantic-core 2.46.4`,
   `httpcore 1.0.9`, `annotated-types`, `annotated-doc`, `typing-inspection`,
   `importlib_metadata`, `nest-asyncio`, `retrying`, `zipp`.
   `uvicorn` and `plotly` were already satisfied.

#### Lockfile impact — reported, not silently applied

`uv.lock` and `requirements.lock` were **not** regenerated. Doing so would require a
full `uv sync`, which would rewrite the live environment (including downgrading
`torch 2.9.1+cu128` toward the pinned cu121 index) and is far outside this tranche's
mandate.

Required follow-up, to be decided by the repository owner:

- add the new runtime deps to `pyproject.toml` as a dedicated
  `[project.optional-dependencies].control` extra (done in this tranche — declaration only),
- regenerate `uv.lock` in a separate, reviewed change,
- reconcile the pre-existing drift between `pyproject.toml`'s cu121 torch pin and the
  installed cu128 torch.

---

## 3. Test baseline

Runner: `python -m pytest` under pyenv 3.10.13, from repo root.

### 3.1 Full collection (B-01)

```
python -m pytest --collect-only -q
→ 427 tests collected, 4 errors in 9.60s
```

The 4 collection errors are **pre-existing baseline failures**:

| Module | Error | Ours? |
|---|---|---|
| `bench/tests/test_report_schema_guardrails.py` | `ImportError: attempted relative import with no known parent package` | **yes — real repo bug** |
| `third_party/Adaptive-KNet-ICASSP24/filters/KalmanFilter_test.py` | `ModuleNotFoundError: No module named 'filters'` | no — upstream |
| `third_party/KalmanNet_TSP/Filters/EKF_test.py` | `ModuleNotFoundError: No module named 'Filters'` | no — upstream |
| `third_party/KalmanNet_TSP/Filters/KalmanFilter_test.py` | `ModuleNotFoundError: No module named 'Filters'` | no — upstream |

**Baseline bug B-BUG-01** — `bench/tests/test_report_schema_guardrails.py:11` does
`from .test_plan_matrix_minimal import run_plan_matrix_minimal`, but `bench/tests/`
has **no `__init__.py`** (the repo consistently uses `init.py` without dunders:
`bench/tests/init.py`, `bench/models/init.py`, …). The relative import can therefore
never resolve under pytest. This module is uncollectable at baseline and remains so;
it was **not** fixed in this tranche because doing so means either adding
`bench/tests/__init__.py` (changes package semantics repo-wide) or rewriting the
import (changes an existing test module) — both outside this tranche's mandate and
neither is needed by the control plane.

**Baseline observation B-OBS-01** — bare `pytest` from the repo root descends into
`third_party/`, which is why 3 upstream helper scripts named `*_test.py` are picked up.
This tranche adds a `[tool.pytest.ini_options]` block to `pyproject.toml` with
`norecursedirs` for `third_party` and friends, so that plain `pytest` collects only
repository-owned tests. This changes no library behaviour.

### 3.2 Baseline pass/fail (repository-owned suites)

```
python -m pytest bench/tests tests -q --ignore=bench/tests/test_report_schema_guardrails.py
→ 425 passed, 2 skipped, 30 subtests passed in 74.04s
```

**Baseline is green.** Any new failure in these 425 is a regression introduced by this
tranche.

### 3.3 28-variant identity regression (re-run)

```
python -m pytest tests/test_viz_init_provenance_comparison.py -q   → 28 passed in 8.78s
python -m unittest tests.test_viz_init_provenance_comparison       → Ran 28 tests ... OK
```

Confirms the audit's §11 finding: the init-provenance suite is green under both
runners. This is the regression gate that must stay green.

---

## 4. Representative artifacts

| Item | Location / count |
|---|---|
| Run tree root | `runs/` — 110 top-level suite directories |
| Leaf run directories (`run_plan.json`) | 1655 |
| Runs with `metrics.json` | 1210 |
| Runs with viz `meta.json` (Inspector-loadable) | 50 |
| Example viz run | `runs/viz3a_synthetic/large_T10000_N32/` (`meta.json` + `series/`) |
| Example checkpoint | `runs/_model_cache/gpu_basilisk_me_split_pilot/split_knet/5f666d4dd0faff11c3fdda19/model.pt` |
| Legacy run dir layout | `runs/<suite>/<task_id>/<model_id>/<track_id>/seed_<n>/scenario_<hash>/` (from `run_suite.py` `output_dir_template`), optionally `/init_<init_id>` under `--plan-isolation` |
| Legacy artifact set | `config_snapshot.yaml`, `run_plan.json`, `metrics.json`, `metrics_step.csv`, `budget_ledger.json`, `env.json`, `env.txt`, `git_versions.txt`, `pip_freeze.txt`, `requirements.lock`, `stdout.log`, `timing.csv`, `checkpoints/model.pt`, `checkpoints/train_state.json`, `artifacts/preds_test.npz` |
| Data cache | `bench_data_cache/` — 88 entries |
| Reports | `reports/` — 38 entries |

**None of these were written to by this tranche.** All new control-plane state lives
under a separate root (`control/` by default, overridable), and every test uses an
isolated `tmp_path` root.

---

## 5. Pre-existing findings that shape the design

1. **`bench/runners/orchestrate.py` is a dead scaffold.** `run_plan()` and
   `run_in_docker()` raise `NotImplementedError`; only `plan_runs()` and
   `run_in_subprocess()` are live, and `run_in_subprocess` uses
   `subprocess.Popen(..., stdout=PIPE, stderr=PIPE)` with `communicate()` — no process
   group, no session, no detachment, no heartbeat. It is **not** reusable as the worker
   manager (it blocks the caller and dies with it), so `bench/control/process/` is a new
   implementation. `plan_runs()` is left untouched.

2. **The real runner is `bench/runners/run_suite.py`** (2974 lines); `run_one()`
   at line 1572 owns the entire lifecycle and writes to a **deterministic path** derived
   from `output_dir_template` (line ~1621). Re-running the same config overwrites the
   same directory — this is exactly risk **R-01/DND-004**, and is why the new allocator
   is a separate, additive path rather than a change to `run_one`.

3. **Identity today is presentation-only.** `viz/app/components/model_toggle_picker.py:35`
   `variant_label(meta)` builds a display string from `model_id` + init provenance; the
   Inspector uses it as a de-facto key, and elsewhere keys on `str(run.run_dir)`. There
   is no canonical persistent ID. Confirms audit §11.

4. **The Streamlit Inspector already supports run deep links.**
   `viz/app/components/overlay_picker.py:147` calls
   `_run_from_query(runs, runs_root, "run")`, which matches a `?run=` query value
   against either the absolute run dir or its path relative to the runs root. **No change
   to the Streamlit app is needed** for the Dash → Inspector deep link; the Dash side
   only has to emit `?run=<path>`. This is the lowest-risk possible integration.

5. **Model registry has 20 `model_id` keys** (`bench/models/registry.py`) mapping onto
   10 adapter classes, several `model_id`s sharing one class
   (`me_split_knet_v0*` → `MESplitKNetV0Adapter`, `oracle_kf`/`nominal_kf`/`mb_kf_*` →
   `ModelBasedKFAdapter`). `model_id` alone therefore does **not** determine the
   implementation — reinforcing the need for a separate `implementation_id`.

6. **No browser is installed** (`google-chrome`, `chromium`, `firefox`,
   `chromedriver`, `geckodriver` all absent). Real headless-browser tests
   (`dash[testing]`/Selenium/Playwright) cannot run here. Dash page smoke tests are
   therefore driven server-side through the Flask/WSGI test client and a live HTTP
   server, which exercises routing, layout rendering, and callback dispatch but **not**
   client-side JavaScript. Recorded as a known limitation rather than papered over.

---

## 6. Baseline command reference

```bash
PY=/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python

# collection gate
$PY -m pytest --collect-only -q

# baseline suites
$PY -m pytest bench/tests tests -q --ignore=bench/tests/test_report_schema_guardrails.py

# 28-variant identity regression
$PY -m pytest tests/test_viz_init_provenance_comparison.py -q
```
