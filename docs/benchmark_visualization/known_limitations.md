# Known Limitations

What this build does **not** do, and what could mislead you if you assume otherwise.
Read alongside `implementation_status_phase0_phase1_phase3.md` §10.

---

## 1. Research-integrity limitations (read these first)

### 1.1 No exact resume exists anywhere

`supports_exact_resume` is `false` for every model, and a config with
`resume.mode != "none"` is rejected at validation. A `model.pt` file — legacy or
new — contains weights and a couple of counters. It has **no** optimizer state,
scheduler, AMP scaler, RNG streams, sampler position, epoch/batch cursor, or
early-stopping counter. Loading it is a **warm start**. Calling it a resume, in a
paper or a log, would be wrong.

### 1.2 `paper_fidelity_status` is not a claim of correctness

`unverified` means exactly that: nobody has tested fidelity in either direction.
It is not a soft "probably fine". The one status that carries real information is
`split_knet` = `partial`, which records a specific known deviation: the adapter
uses **one combined network with a single Adam optimizer**, not the paper's
alternating optimization of the two split heads. Any comparison against published
Split-KalmanNet numbers must state this.

### 1.3 Legacy run states are inferred, not recorded

Imported runs never had a state machine. `COMPLETED` on a legacy row means "the
artifacts look like a finished run", at the stated confidence — not "a worker
recorded a terminal transition". A `low`/`unknown` confidence row is a run whose
outcome is genuinely not known. The dashboard labels these; do not aggregate
across confidence levels without saying so.

### 1.4 `ORPHANED` is not `FAILED`

`ORPHANED` means the worker vanished without recording an outcome. The run may
have been nearly finished, or may have died at step 1. Nothing in this build
converts it to `FAILED` automatically, and nothing should: adjudicating it
requires a human looking at the heartbeat, the PID, and the artifacts.

---

## 2. Testing limitations

### 2.1 No real browser test

This host has no Chrome, Chromium, Firefox, `chromedriver`, or `geckodriver`, so
`dash[testing]`, Selenium, and Playwright cannot run. Dash is tested
**server-side**: routes over HTTP, and callbacks dispatched through Dash's own
`_dash-update-component` endpoint — the same code path a browser triggers.

What that covers: routing, layout construction, data binding, callback outputs,
error handling when the API is down.
What it does **not** cover: client-side JavaScript, actual rendering, CSS layout,
responsive behaviour, keyboard navigation in a real browser, and screen-reader
output. Accessibility is addressed by construction (state is always a text label,
never colour alone; every badge carries a `title`) but is **unverified by an
assistive-technology test**.

### 2.2 Performance is unmeasured at scale

Design doc 06 §12 asks for 1k/10k/100k-run registry timings, large-journal tail
latency, and 24-hour telemetry projections. None of that was measured. The
registry has the required indexes and every list/log/event API is bounded, but
the numbers are **unknown**. Largest tested: ~75 imported runs and journals of a
few hundred events.

### 2.3 Failure injection is partial

Covered: ordinary adapter exception, SIGKILL, stale heartbeat, PID reuse,
unreachable API, corrupt/truncated journal, broken telemetry collector,
uncollectable legacy directory.
**Not** covered: CUDA OOM, NaN loss propagation through the control plane,
checkpoint disk-full, SQLite `SQLITE_BUSY` under sustained heavy load, manager
crash mid-launch, event-write failure on a full disk.

### 2.4 Telemetry overhead is not quantified

Acceptance T-04 asks for a telemetry on/off comparison. Not measured. The
sampler is a daemon thread doing a `psutil` read and one NVML call per interval
(default 2 s), which is not expected to be material against a training step, but
that is reasoning, not a measurement.

---

## 3. Functional limitations

### 3.1 Read-only dashboard, single-node, single-user

No launch, stop, terminate, resume, or warm-start control exists in the UI or the
API. The API has **no** `POST`/`PUT`/`PATCH`/`DELETE` routes at all (enforced by
a test). Launching is CLI-only. There is no authentication, no authorization, and
no multi-user support; both servers refuse a non-loopback bind unless
`BENCH_CONTROL_ALLOW_PUBLIC_BIND=1`.

### 3.2 No scheduler and no GPU arbitration in practice

`gpu_leases` and its exclusivity index exist and work, but **nothing acquires a
lease automatically**. Launching two GPU runs concurrently will oversubscribe the
device. On this host there is exactly one GPU (RTX 4060, 8 GiB), so a second
concurrent CUDA run is likely to OOM. Serialize GPU runs manually.

### 3.3 Live updates are polling, not push

`dcc.Interval` at 3 s drives HTTP polling. There is no WebSocket. The event API
already provides the cursor and gap-fill semantics a push transport needs, so
adding one later is additive — but a run producing events much faster than 3 s
will show them in batches, and the resource chart's resolution is bounded by the
telemetry interval.

### 3.4 Step-level instrumentation covers three adapters

`kalmannet_tsp`, `split_knet`, and `mb_kf` emit per-update metrics. `adaptive_knet`,
`maml_knet`, `me_split_knet_v0`, and `basilisk_mrp_ekf` emit **phase boundaries
and final metrics only** — their charts will show a final point and no curve.
That is a declared coverage gap (visible as a badge in the UI), not a bug, and it
is deliberately **not** patched by parsing stdout. Reasons per adapter are in
`implementation_status_phase0_phase1_phase3.md` §7.

### 3.5 The typed config models the execution contract, not every adapter knob

Interpreted keys are listed in `bench/control/config/compatibility.py`
(`TASK_SUPPORTED_KEYS`, `MODEL_SUPPORTED_KEYS`, `RUNNER_SUPPORTED_KEYS`).
Everything else is preserved verbatim in `model_config_extra` / `task_config_extra`,
listed in `unsupported_fields`, and **included in the structural hash** — an
uninterpreted key still changes results. Nothing is dropped, but the control
plane cannot validate what it does not model, so a typo inside an unmodelled key
will not be caught.

### 3.6 The suite executor contains outputs, with one dataset exception

`SuiteExecutor` rewrites `reporting.output_dir_template` and
`runner.model_cache_dir` to absolute paths inside the run directory, so the
existing runner writes nothing into the shared `runs/` tree. Two consequences:

* the **train cache is not shared** across control-plane runs, so a cache hit
  that the CLI would get, a control-plane run will not — it retrains;
* the **dataset cache** under `bench_data_cache/` is still shared. A run whose
  scenario has not been generated yet will write a new dataset entry there. That
  adds files; it does not modify or delete existing ones.

### 3.7 Checkpoint catalog is read-only

The `checkpoints` table and its reader exist; nothing writes to them. Weight
files produced by adapters appear under Artifacts, explicitly marked as not
resume-certified. This is Phase 2 work.

### 3.8 Legacy import is a snapshot

Import is idempotent by path, but a legacy run whose files change **after**
import is not re-read; the registry keeps the first reading. Re-importing an
already-present path is a no-op, not a refresh. There is no un-import.

### 3.9 Metric name/step-axis discipline is by convention

`step_type` is carried on every metric event, and the canonical names in
`bench/control/events/schema.py` are the intended vocabulary — but nothing
*enforces* that an adapter uses them. An adapter emitting `train_loss` instead of
`loss/train_total` will simply not appear on the standard chart.

---

## 4. Environment limitations

### 4.1 Lockfiles are stale, and were already stale

`uv.lock` and `requirements.lock` do not describe the live environment, and did
not before this tranche (`pyproject.toml` pins the cu121 torch index; the
installed torch is 2.9.1+cu128). This tranche added a `control` extra to
`pyproject.toml` but **did not regenerate the lockfiles** — doing so needs a
`uv sync`, which would rewrite the working environment. Reconciling this is a
separate, reviewed change.

### 4.2 `.venv/` is a decoy

The repository's `.venv/` is an empty Python 3.13 environment with none of the
project dependencies. The real interpreter is
`/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python`. Activating `.venv` will
appear to break everything.

### 4.3 psutil and NVML are optional and their absence is silent-by-design

Without `psutil`, CPU/RAM telemetry reports itself unavailable and the sample
simply lacks those fields. Without NVML *and* `nvidia-smi`, the GPU section is
`null`. `null` means "not measured" — it is never rendered as `0`, and the API
says so explicitly, but a reader who assumes a missing series means "idle" will
be wrong.

### 4.4 POSIX only

Process groups, `os.setsid`, `killpg`, `/proc` fallbacks, and directory `fsync`
are POSIX. The worker supervision layer will not work on native Windows. Tested
on Linux under WSL2.

---

## 5. Pre-existing repository issues (not introduced here)

* **B-BUG-01** — `bench/tests/test_report_schema_guardrails.py` cannot be
  collected: it uses a relative import, but `bench/tests/` has no `__init__.py`
  (the repo uses `init.py` without dunders). Uncollectable at baseline, still
  uncollectable.
* The working tree is dirty with **185 deleted** tracked run artifacts and 116
  modified files. This tranche preserved that state exactly and restored nothing.
* Two submodules (`KalmanNet_TSP`, `MAML_KalmanNet`) are dirty with untracked
  `__pycache__` only — no source modification.
