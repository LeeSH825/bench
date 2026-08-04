# Reproducible Release Baseline

This document exists because of finding **V-001**: the benchmark execution
visualization control plane was verified while it lived only in a dirty working
tree. Anyone who checked out the verified commit `d1cc4b0` got a repository with
no `bench/control/`, no `bench/ui/`, and no control tests. The verification was
real; the artifact under review was not reachable.

This is the procedure that makes the implementation reachable and reproducible
from Git alone.

## 1. Commits

| Commit | Purpose |
|---|---|
| `d1cc4b0` | Prior baseline. Contains **no** control plane. |
| `355b6ca` | `feat(benchmark-viz)`: `bench/control/**`, `bench/ui/**`, adapter/runner observer instrumentation, `control` extra + pytest collection config |
| `179a42d` | `test(benchmark-viz)`: collection repair (V-002) and real-adapter parity test (V-003) |
| `ee862a2` | `fix(benchmark-viz)`: track the viz run fixtures three tracked tests need |

Branch: `benchmark-viz/stabilize-release-baseline`. Local only; nothing pushed.

## 2. What is deliberately *not* in these commits

The working tree this was cut from also contains a large body of unrelated,
in-progress user work. It was left untouched, unstaged, and uncommitted:

- `bench/models/registry.py` + `spike_ra_knet.py`, `spike_split_knet.py`,
  `g1_snn_split_knet.py` (spike/SNN adapters)
- `bench/models/adaptive_knet.py`, `bench/models/maml_knet.py` — these remove
  `preds_test.npz` writing, which is a behaviour change unrelated to
  observability and would weaken the legacy artifact contract
- `bench/tasks/bench_generated.py`, `bench/tasks/generator/basilisk_imu_adcs.py`
  (ADCS replay task families)
- `bench/tests/run_all.py`, `test_runner_smoke.py`, `test_kf_baseline_smoke_plan.py`
- 27 untracked test files and their `bench/visualization/` modules covering
  ADCS, Phase 5c–7 replay/checkpoint, and Vizard
- `runs/`, `reports/`, `DECISIONS.md`, `USERGUIDE.md` edits
- every `__pycache__` / `.pyc`, and every verification artifact directory

This is why a clean checkout of the baseline collects **449** tests while the
working tree collects **590**. The 141-test difference is exactly those 27
untracked files. No control-plane test is missing from the commits, and no test
present in the clean checkout is absent from the working tree.

## 3. Reproducing from scratch

```bash
git worktree add --detach /tmp/bench-viz-check ee862a2
cd /tmp/bench-viz-check
git submodule update --init --recursive

PY=~/.pyenv/versions/3.10.13/bin/python     # see §5

$PY -c "import bench.control, bench.ui, bench.control.api.app, bench.ui.dash_app"
$PY -m bench.control.cli --help

$PY -m pytest --collect-only -q    # 449 collected, 0 errors
$PY -m pytest -q                   # 448 passed, 1 skipped
$PY -m pytest -q tests/test_viz_init_provenance_comparison.py          # 28 passed
$PY -m pytest -q tests/test_control_*.py                               # 163 passed
```

`git status --short` in that worktree must be empty apart from ignored generated
files.

## 4. Verified results at `ee862a2`

| Gate | Result |
|---|---|
| `pytest --collect-only -q` | 449 collected, **0 collection errors** |
| `pytest -q` | **448 passed, 1 skipped, 0 failed** |
| Identity regression | 28 passed |
| Control targeted | 163 passed |
| Real-adapter parity (KNet + Split, A/B/C/D) | bitwise equal |
| Lifecycle E2E (normal / restart / failure / orphan) | PASS |
| Playwright DOM smoke | PASS |

## 5. Interpreter

The mandated commands are `python -m pytest`. On this machine `pyenv global` is
`system`, so bare `python` does not resolve and `python3` resolves to 3.12.3,
which has no `pytest` and no `fastapi`. The project environment is **3.10.13**
(this is also what the original verification used — its warnings cite
`.pyenv/versions/3.10.13/...`).

Select it explicitly, by any of:

```bash
PYENV_VERSION=3.10.13 pyenv exec python -m pytest -q
# or
~/.pyenv/versions/3.10.13/bin/python -m pytest -q
# or, to make bare `python` work inside the repo:
pyenv local 3.10.13
```

`pyenv local` writes a `.python-version` file. That file is **not** committed
here: it is a machine-level environment choice, and pinning it in the repository
would change interpreter resolution for every existing user of this checkout
without their asking. It is recorded here instead so the next verifier does not
lose time on a missing-`pytest` error that has nothing to do with the code.

## 6. Standing risk

`runs/` is in `.gitignore`, yet 17,540 files under it are force-tracked because
tests depend on them. `ee862a2` follows that existing practice for three more
fixture trees. It is a pre-existing pattern worth revisiting — test fixtures
would be better under an explicit `tests/fixtures/` path than under an ignored
output directory — but changing it is a migration, not a stabilization step.
