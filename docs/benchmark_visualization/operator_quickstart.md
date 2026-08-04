# Operator Quickstart

How to launch a supervised run and watch it. Ten minutes, no GPU required.

**One thing to internalize first:** by default the dashboard is a **viewer**. It
cannot start, stop, or change a run. You launch from the CLI; the run executes in
its own detached process; the dashboard reads what that process records. Closing
the dashboard, or the API, does not affect a running job.

Write control is opt-in. With `BENCH_CONTROL_ENABLE_WRITES=1` and a loopback
bind, the UI additionally offers Stop safely / Resume training
(`write_control_operator_guide.md`) and a New run wizard
(`config_gui_operator_guide.md`). Everything below works the same either way.

---

## 0. Environment

```bash
PY=/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python   # the real interpreter
cd /home/dss-pc-05/bench
```

Do **not** activate `.venv/` — it is an empty Python 3.13 environment with none
of the project dependencies installed.

Install the control-plane dependencies once (additive; nothing is upgraded):

```bash
$PY -m pip install -e '.[control]'      # or: pip install fastapi uvicorn dash plotly pydantic psutil nvidia-ml-py
$PY -m pip install pytest               # already declared in the dev extra
```

Choose where control-plane state lives. Default is `<repo>/control`:

```bash
export BENCH_CONTROL_ROOT=/home/dss-pc-05/bench/control
```

Nothing under `runs/`, `reports/`, or `bench_data_cache/` is written by the
control plane; every new run lives under `$BENCH_CONTROL_ROOT/runs/`.

---

## 1. Launch a synthetic run (30 seconds, no dataset, no torch)

```bash
$PY -m bench.control.cli launch-synthetic --updates 60 --telemetry-interval 1.0
```

Output includes the `run_id`, `pid`, `process_group_id`, and run directory. The
CLI exits immediately — the worker keeps running.

```bash
$PY -m bench.control.cli list
```

```
RUN ID                                 STATE      MODEL                     STEP  LEGACY TASK
019fb340-7e3f-7cf5-8b55-6887ee935824   COMPLETED  kalmannet_tsp               60  no     synthetic_task
```

---

## 2. Import your existing runs (read-only)

```bash
$PY -m bench.control.cli import-legacy --limit 200
```

This **reads** `runs/` and writes registry rows. It never modifies, moves, or
deletes anything in that tree. Imported rows are marked `legacy` and carry a
status confidence, because a legacy directory has no recorded state machine —
its outcome is inferred from its artifacts.

Re-running the command is a no-op for already-imported directories.

---

## 3. Start the dashboard (two processes)

```bash
# terminal 1 — read-only API
$PY -m bench.control.api.app --host 127.0.0.1 --port 8765

# terminal 2 — Dash UI
$PY -m bench.ui.dash_app --host 127.0.0.1 --port 8766 --api http://127.0.0.1:8765
```

Open <http://127.0.0.1:8766/runs>.

Both refuse a non-loopback bind. There is no authentication, and the API exposes
local filesystem paths, so if you genuinely need remote access put it behind a
trusted reverse proxy and set `BENCH_CONTROL_ALLOW_PUBLIC_BIND=1`.

### Pages

| Page | Shows |
|---|---|
| `/runs` | Run table keyed by `run_id`, with state, model + variant, task, step, source. Filter by state and by control-plane vs legacy. |
| `/runs/<run_id>` | Identity and capability badges, progress and worker liveness, outcome, live loss/eval charts, resource chart, bounded stdout/stderr tails, checkpoints, artifacts, provenance and the Inspector deep link, full state-transition history. |
| `/system` | Per-subsystem health, GPU inventory, orphan candidates, worker table, and the full model capability matrix. |

Pages refresh every 3 seconds by polling. Deep links work in a fresh session:
`/runs/<run_id>` is a real route.

---

## 4. Watch a run live

Launch something slow enough to observe, then open its detail page:

```bash
BENCH_CONTROL_STEP_SLEEP=0.2 $PY -m bench.control.cli launch-synthetic \
    --updates 200 --telemetry-interval 1.0
```

While it runs you should see, **before** it reaches a terminal state: the state
badge on `RUNNING`, `global_step` advancing, a fresh heartbeat, the training-loss
curve extending, CPU/RSS traces, and the stdout tail growing.

---

## 5. Launch a real benchmark run

```bash
$PY -m bench.control.cli launch \
    --suite bench/configs/gpu_figure_pack_smoke.yaml \
    --task F5aP_gpu_m2n2_T50_invR2db_0 \
    --model kalmannet_tsp \
    --init trained \
    --device cpu
```

This calls the existing, unmodified `bench.runners.run_suite.run_one`, with its
output directory and model cache redirected **inside** the new run directory —
so it cannot overwrite anything under `runs/`.

`--init` follows the runner's own semantics: only `trained` triggers training;
`untrained` / `pretrained` / `loaded` are evaluation plans.

Use `--dry-run` to allocate and register a run without starting a worker.

Only `kalmannet_tsp`, `split_knet`, and the model-based KF baselines stream
per-update metrics. Other adapters emit phase boundaries and final metrics only —
the run detail page shows which, as a badge.

**One GPU on this host.** Nothing acquires a GPU lease automatically, so do not
launch two CUDA runs concurrently; the second will likely OOM. This is equally
true of runs started from the browser — there is no queue.

### The same run, from the browser

With write mode enabled, <http://127.0.0.1:8766/new-run> starts the same run
from the same tracked preset: choose preset → configure → validate → review →
launch. It resolves through the *same* code as the CLI, and the resolved spec,
hashes, `variant_id` and training path are identical field-for-field (pinned by
`tests/test_control_gui_cli_parity.py`). Details in
`config_gui_operator_guide.md`.

---

## 6. When something goes wrong

### A run failed

`FAILED` with an exit code and an error summary means the workload raised and the
worker recorded it. Look at:

```bash
$PY -m bench.control.cli show <run_id>
less $BENCH_CONTROL_ROOT/runs/<experiment_id>/<run_id>/failure.json
less $BENCH_CONTROL_ROOT/runs/<experiment_id>/<run_id>/artifacts/traceback.txt
less $BENCH_CONTROL_ROOT/runs/<experiment_id>/<run_id>/stderr.log
```

Exit codes: `0` completed · `20` cancelled before execution · `30` validation/
incompatibility · `40` training or evaluation raised · `50` checkpoint write
failure · `60` worker protocol failure · `70` external termination. A **negative**
code means a signal killed the process.

### A run is stuck on `RUNNING` but nothing is happening

Its worker may be gone. Nothing is ever auto-killed on a stale heartbeat alone —
PID identity is verified first.

```bash
$PY -m bench.control.cli reconcile --dry-run     # report only
$PY -m bench.control.cli reconcile               # mark verified-dead workers ORPHANED
```

The System page shows the same candidates with the reason for each.

`ORPHANED` means **unknown outcome**, not failure. Decide by looking at the
artifacts and the last recorded step; a live-but-hung worker is a different
problem from a vanished one, and `reconcile` will not touch the former.

### The dashboard shows "unreachable"

The API is not running, or is on a different port. Start it, or pass the right
`--api`. The dashboard degrades to an error panel rather than breaking.

### After a machine restart

Runs that were executing are gone but their registry rows still say `RUNNING`.
Run `reconcile` once; their workers' PIDs are absent, so they are classified
`ORPHANED` with `status_confidence = unknown`.

---

## 7. Visualizing results in the existing Run Inspector

The Streamlit Run Inspector is unchanged and remains the tool for trajectory and
diagnostic analysis. The run detail page's **Provenance and deep links** section
gives you the query parameter to use:

```bash
streamlit run viz/app/main.py
# then append the shown ?run=<path> to the Streamlit URL
```

The link appears only when the run actually has a `meta.json`; the Inspector
indexes nothing else, so an unconditional link would just lead to "no valid runs
found".

---

## 8. Safety rules worth remembering

1. Never edit anything under `$BENCH_CONTROL_ROOT/runs/<...>/` by hand. Run
   directories are immutable; re-run instead.
2. Never treat a `model.pt` as resumable. It has no optimizer, RNG, or cursor
   state — loading it is a warm start.
3. Never quote `paper_fidelity_status: unverified` as "verified". For
   `split_knet` specifically, the adapter is **not** the paper's alternating
   optimization.
4. Do not aggregate legacy runs of different status confidence without saying so.
5. Back up `registry.sqlite3` before upgrading the control plane; migrations are
   forward-only and take an automatic backup, but the file is small — copy it.
6. Copy `registry.sqlite3`, `registry.sqlite3-wal`, and `registry.sqlite3-shm`
   together, or use `sqlite3 registry.sqlite3 ".backup out.sqlite3"`.

---

## 9. Command reference

```bash
$PY -m bench.control.cli --help
$PY -m bench.control.cli launch --help
$PY -m bench.control.cli launch-synthetic --updates N [--fail-at-step K] [--device cpu]
$PY -m bench.control.cli list [--limit N] [--active] [--no-include-legacy] [--json]
$PY -m bench.control.cli show <run_id>
$PY -m bench.control.cli import-legacy [--root PATH] [--limit N]
$PY -m bench.control.cli reconcile [--heartbeat-timeout S] [--dry-run]

$PY -m bench.control.api.app --host 127.0.0.1 --port 8765 [--control-root PATH]
$PY -m bench.ui.dash_app     --host 127.0.0.1 --port 8766 --api http://127.0.0.1:8765 [--poll-ms 3000]
```

Useful environment variables:

| Variable | Effect |
|---|---|
| `BENCH_CONTROL_ROOT` | Control-plane root (default `<repo>/control`) |
| `BENCH_CONTROL_API` | Default API base URL for the dashboard |
| `BENCH_CONTROL_HEARTBEAT_INTERVAL` | Worker heartbeat period, seconds (default 10) |
| `BENCH_CONTROL_STEP_SLEEP` | Synthetic executor: seconds per step (demo pacing) |
| `BENCH_CONTROL_FAIL_AT_STEP` | Synthetic executor: inject a failure at this step |
| `BENCH_CONTROL_ALLOW_PUBLIC_BIND` | Set to `1` to permit a non-loopback bind |

API endpoints (all `GET`; interactive docs at `/docs`):

```
/api/v1/system/health           /api/v1/system/gpus
/api/v1/system/workers          /api/v1/system/state-machine
/api/v1/capabilities            /api/v1/orphan-candidates
/api/v1/runs                    /api/v1/runs/{run_id}
/api/v1/runs/{run_id}/events    ?after_event_id=&limit=&event_type=
/api/v1/runs/{run_id}/metrics   /api/v1/runs/{run_id}/resources
/api/v1/runs/{run_id}/artifacts /api/v1/runs/{run_id}/logs?stream=stdout|stderr
```

---

## 10. Running the tests

```bash
$PY -m pytest --collect-only -q   # → 0 collection errors
$PY -m pytest -q                  # no --ignore needed

$PY -m pytest tests/test_viz_init_provenance_comparison.py -q   # 28-variant identity gate
$PY -m pytest tests/test_control_*.py -q                        # control plane only
$PY -m pytest tests/test_control_real_adapter_numerical_parity.py -q  # observer/telemetry inertness
```

The `--ignore` workaround is gone: `test_report_schema_guardrails.py` used a
relative import that only resolved when it was imported as part of the
`bench.tests` package, and now uses the absolute import the rest of the package
already uses. Collection is scoped to `bench/tests` and `tests` by
`[tool.pytest.ini_options]` so that bare `pytest` does not descend into
`third_party/` and try to import vendored upstream `*_test.py` helpers.

Counts depend on which working-tree work is present. On a clean checkout of the
stabilization baseline the suite is **449 collected / 448 passed / 1 skipped**.
A working tree that also carries the in-progress ADCS / Phase 5c–7 replay /
Vizard / spike-model work collects ~590. Both must report **0 collection
errors** and **0 failures**.

---

## 11. Scale characterization

Measured on a temporary synthetic registry (never the production one), 12
repetitions per endpoint, on this workstation:

| Measurement | 1,000 runs | 10,000 runs |
|---|---|---|
| Registry migration | 0.011 s | 0.010 s |
| Bulk insert | 0.14 s | 1.65 s |
| DB size | 1.08 MB | 9.72 MB |
| `GET /runs?limit=50` p50 / p95 | 2.85 / 3.36 ms | 4.76 / 5.04 ms |
| Deep page (`offset=500`) p50 | 3.55 ms | 8.30 ms |
| `GET /runs/{id}` p50 | 1.52 ms | 1.50 ms |
| State filter p50 | 2.44 ms | 3.33 ms |
| Events cursor p50 | 1.24 ms | 1.20 ms |
| Dash Runs page load p50 | — | 1.08 s |
| API process peak RSS | 36 MB | 40 MB |

What this establishes:

- **Listing is bounded.** `limit` is capped server-side at 1000; `limit=100000`
  and `limit=5000` are rejected with HTTP 422 rather than silently served.
- **The UI does not ship the whole table.** At 10,000 runs the Runs page renders
  `Showing 50 of 10000 runs` — 50 rows, ~6 KB of body text.
- **Run detail does not degrade with registry size** (1.5 ms at both scales).

One endpoint does scale with load: `GET /api/v1/system/health` measured 13 ms at
1,000 runs and 110 ms at 10,000. The cost is `find_orphan_candidates()`, which
inspects every **non-terminal** run and probes its PID. The synthetic fixture
made 25% of runs `RUNNING` (2,500 live workers), which is not a realistic
operating state. Re-measured with 10,000 runs that are all terminal, health
returns in **4.06 ms p50**. So the cost tracks concurrent live runs, not
registry size, and is not a blocker. If you ever do run thousands of concurrent
workers, poll `/system/health` less often than the run list.

These numbers are a characterization, not an SLA.
