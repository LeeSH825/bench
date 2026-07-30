# Implementation Status — Phase 0, Phase 1, Phase 3

Scope: frontend-agnostic backend foundation + read-only Dash dashboard.
Baseline: `docs/benchmark_visualization/implementation_baseline.md`.
Audit reference commit: `d1cc4b035597bb029e0bce95a546b29b4664b5c6`.

**This document separates what was built from what was planned.** Anything under
"Not implemented" is not present in the code, is not exposed by the API, and has
no button in the UI.

---

## 1. Versions

| Contract | Version | Where defined |
|---|---|---|
| Control plane | `0.1.0` | `bench/control/__init__.py` `CONTROL_PLANE_VERSION` |
| Run config / `ResolvedRunSpec` | `1` | `bench/control/config/schema.py` `CONFIG_SCHEMA_VERSION` |
| Registry (SQLite) | `1` | `bench/control/registry/schema.py` `REGISTRY_SCHEMA_VERSION`, `PRAGMA user_version` |
| Event journal | `1` | `bench/control/events/schema.py` `EVENT_SCHEMA_VERSION` |
| Variant identity document | `1` | `bench/control/identity.py` `VARIANT_IDENTITY_SCHEMA_VERSION` |
| Adapter capability document | `1` | `bench/control/capabilities.py` `CAPABILITY_SCHEMA_VERSION` |
| HTTP API | `v1` | `/api/v1/...` |

Reader policy: a document whose `schema_version` is **newer** than the running
build is rejected with an explicit error, never guessed at. The event reader
accepts one version back.

---

## 2. Modules and public interfaces

### 2.1 `bench/control` — core (no UI framework imports; enforced by test)

| Module | Public surface | Purpose |
|---|---|---|
| `canonical.py` | `canonical_json`, `canonical_bytes`, `content_hash`, `text_hash`, `short_hash` | Deterministic JSON + SHA-256. Rejects NaN/Inf and sets. |
| `identity.py` | `ExperimentId`, `RunId`, `ModelId`, `ImplementationId`, `InitId`, `VariantId`, `CheckpointId`, `ArtifactId`, `WorkerInstanceId`, `uuid7`, `compute_variant_id`, `variant_label`, `describe_identity` | Allocated (UUIDv7) vs derived (SHA-256) identity. Python `hash()` is never used. |
| `paths.py` | `control_root`, `registry_path`, `runs_root`, `legacy_runs_root`, `safe_relative_path`, `UnsafePathError` | Control-root resolution and path allowlisting. |
| `provenance.py` | `repository_provenance`, `git_commit`, `git_dirty`, `submodule_revisions`, `environment_fingerprint` | Per-run code/env provenance including submodule dirty flags (closes R-18). |
| `capabilities.py` | `AdapterCapabilities`, `capabilities_for`, `implementation_id_for`, `all_capabilities`, `capability_index` | Declared model capabilities. `supports_exact_resume` is `False` everywhere. |
| `allocation.py` | `RunLocation`, `allocate_run_directory`, `write_run_spec`, `atomic_write_text`, `AllocationError` | Immutable `runs/<experiment_id>/<run_id>/` allocation. |
| `config/schema.py` | `RunSpecDraft`, `ResolvedRunSpec`, all section types, `ValidationIssue`, `ConfigValidationError`, `UnknownKeyPolicy`, `structural_config_hash`, `operational_config_hash` | Typed config (stdlib dataclasses, no Pydantic in core). |
| `config/resolver.py` | `validate_draft`, `resolve_run_spec`, `draft_from_dict`, `resolved_from_dict`, `resolved_from_json` | Validation + identity/hash resolution + round-trip. |
| `config/compatibility.py` | `draft_from_suite`, `drafts_from_suite`, `TASK_SUPPORTED_KEYS`, `MODEL_SUPPORTED_KEYS`, `RUNNER_SUPPORTED_KEYS` | Existing suite YAML → typed draft. Suite format unchanged. |
| `registry/schema.py` | `RunState`, `ALLOWED_TRANSITIONS`, `TERMINAL_STATES`, `ACTIVE_STATES_THIS_TRANCHE`, `validate_transition`, `RunRecord`, `WorkerRecord`, `ExperimentRecord`, `ArtifactRecord` | State machine and record types. |
| `registry/sqlite.py` | `SqliteRegistry`, `open_registry`, `ConcurrencyError`, `RegistryError`, `SchemaVersionError`, `backup_database` | WAL registry with optimistic concurrency. |
| `registry/migrations/` | `MIGRATIONS`, `latest_version`, `pending` | Forward-only numbered migrations. |
| `events/schema.py` | `Event`, `EventType`, `EVENT_SCHEMA_VERSION`, canonical metric-name constants | Event document. |
| `events/writer.py` | `EventWriter` (+ `.status/.metric/.log/.warning/.failure/.resource/.artifact/.checkpoint`) | Append-only JSONL writer with a split flush/fsync policy. |
| `events/reader.py` | `EventReader`, `EventPage`, `RecoveryWarning` | Bounded cursor reads, partial-tail recovery, metric/resource projections. |
| `events/observer.py` | `RunObserver` (Protocol), `NullObserver`, `JournalObserver`, `active_observer`, `set_active_observer` | The adapter-facing contract. |
| `telemetry/base.py` | `ResourceSample`, `GpuSample`, `Collector`, `TelemetrySampler` | Sampling loop; collector failure never fails a run. |
| `telemetry/cpu.py` | `CpuCollector`, `psutil_available`, `process_start_time`, `process_alive` | Process-tree CPU/RSS/disk; psutil optional. |
| `telemetry/nvidia.py` | `NvidiaCollector`, `gpu_inventory` | NVML with `nvidia-smi` fallback; explicit attribution quality. |
| `process/signals.py` | `ExitCode`, `describe_exit_code`, `SignalHandler`, `start_new_session`, `signal_process_group` | Exit-code contract and process-group helpers. |
| `process/manager.py` | `WorkerManager`, `LaunchResult`, `OrphanCandidate` | Detached launch, reaping, orphan detection/adjudication. |
| `process/executors.py` | `Executor`, `SyntheticExecutor`, `SuiteExecutor`, `build_executor` | What a worker runs. |
| `process/worker_cli.py` | `main`, `HeartbeatThread` | Worker entry point. |
| `legacy/importer.py` | `discover_legacy_runs`, `inspect_legacy_run`, `import_legacy_runs`, `legacy_run_id`, `legacy_path_hash`, `LegacyRunCandidate`, `LegacyImportReport` | Read-only import of `runs/`. |
| `api/app.py` | `create_app`, `resolve_bind_host` | FastAPI factory; GET-only. |
| `api/deps.py` | `configure`, `get_registry`, `get_manager`, `active_root` | Service singletons. |
| `cli.py` | `main`, subcommands `launch`, `launch-synthetic`, `list`, `show`, `import-legacy`, `reconcile` | Operator CLI. |

### 2.2 `bench/ui` — Dash client (HTTP only; never opens the registry)

`api_client.ApiClient`, `components` (badges/panels/sections),
`pages.runs`, `pages.run_detail`, `pages.system`, `dash_app.create_dash_app`.

### 2.3 Changes to existing files (deliberately minimal)

| File | Change | Risk |
|---|---|---|
| `bench/runners/run_suite.py` | Guarded `active_observer` import; phase `PHASE_START`/`PHASE_END` events wrapped around `_try_call_setup/_try_call_train/_try_call_eval` in `try/finally` | None under the CLI — the default observer is a no-op. Return values and exceptions propagate unchanged. |
| `bench/models/kalmannet_tsp.py` | Guarded import; per-update train loss, per-eval validation loss, phase start/end, checkpoint artifact event | Same. |
| `bench/models/split_knet.py` | Same instrumentation | Same. |
| `bench/models/mb_kf.py` | Guarded import; `PHASE_SKIPPED` train event | Same. |
| `pyproject.toml` | Added `control` optional extra; added `[tool.pytest.ini_options]` (`testpaths`, `norecursedirs`) | No library behaviour change. |

**No third-party (`third_party/**`) file was modified.** No `TP-xxx` exception
record is required.

---

## 3. Run directory layout produced

```
<control_root>/                       default <repo>/control, override BENCH_CONTROL_ROOT
  registry.sqlite3                    WAL registry (+ -wal/-shm)
  runs/<experiment_id>/<run_id>/
    resolved_run_spec.json            immutable execution contract
    original_config.json              untouched source suite document
    events.jsonl                      append-only journal
    stdout.log  stderr.log            worker stdio, redirected by the child
    failure.json                      written only on an ordinary failure
    artifacts/                        metrics.json, traceback.txt, runner_result.json …
    checkpoints/                      (empty in this tranche — checkpoint v1 is Phase 2)
    provenance/
    tmp/                              partial writes only; never advertised by the API
    legacy/                           SuiteExecutor: the existing runner's own output tree
    model_cache/                      SuiteExecutor: redirected train cache
```

`legacy/` and `model_cache/` exist because `SuiteExecutor` rewrites the suite's
`reporting.output_dir_template` and `runner.model_cache_dir` to absolute paths
**inside this run directory** before calling `run_suite.run_one`. That is what
keeps a control-plane run from writing into the shared deterministic `runs/`
tree (R-01 / DND-004). The repository's `runs/`, `reports/`, and
`bench_data_cache/` trees are never written to by the control plane, with one
documented exception: the **dataset** cache under `bench_data_cache/` is still
read and, for a not-yet-generated scenario, written by the existing generator —
that is dataset material, not run material, and adding entries does not modify
existing ones.

---

## 4. Commands

### Launch (CLI only — the dashboard cannot launch)

```bash
export BENCH_CONTROL_ROOT=/path/to/control          # optional; default <repo>/control
PY=/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python

# tiny synthetic run — no dataset, no torch, seconds
$PY -m bench.control.cli launch-synthetic --updates 60 --telemetry-interval 1.0

# a real suite entry through the existing runner
$PY -m bench.control.cli launch \
    --suite bench/configs/gpu_figure_pack_smoke.yaml \
    --task F5aP_gpu_m2n2_T50_invR2db_0 \
    --model kalmannet_tsp --init trained --device cpu

$PY -m bench.control.cli list
$PY -m bench.control.cli show <run_id>
$PY -m bench.control.cli import-legacy --limit 200
$PY -m bench.control.cli reconcile --dry-run
```

### Worker (normally started by the manager, not by hand)

```bash
python -m bench.control.process.worker_cli \
    --run-id <run_id> \
    --registry <control_root>/registry.sqlite3 \
    --run-spec <control_root>/runs/<experiment_id>/<run_id>/resolved_run_spec.json \
    [--executor suite|synthetic] [--heartbeat-interval 10] \
    [--fail-at-step N] [--step-sleep S]
```

### Dashboard (two processes)

```bash
python -m bench.control.api.app  --host 127.0.0.1 --port 8765     # API
python -m bench.ui.dash_app      --host 127.0.0.1 --port 8766 --api http://127.0.0.1:8765
# then open http://127.0.0.1:8766/runs
```

Both refuse a non-loopback bind unless `BENCH_CONTROL_ALLOW_PUBLIC_BIND=1`.

---

## 5. Test results

Interpreter: `/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python` (3.10.13).

### 5.1 Full suite

```
pytest -q --ignore=bench/tests/test_report_schema_guardrails.py
→ 585 passed, 2 skipped, 30 subtests passed in 92.86s
```

Baseline was **425 passed, 2 skipped**. **160 new tests, zero regressions.**

Collection: `pytest --collect-only -q` → **587 collected, 1 error**. That one
error is the pre-existing `bench/tests/test_report_schema_guardrails.py`
(B-BUG-01, see §8) and is unchanged from baseline. The three baseline
`third_party` collection errors are gone because `norecursedirs` now excludes
vendored code from collection — no vendored file was touched.

### 5.2 Regression gate

```
pytest tests/test_viz_init_provenance_comparison.py -q        → 28 passed
python -m unittest tests.test_viz_init_provenance_comparison  → Ran 28, OK
```

The 28-variant identity regression is green under both runners, unchanged.

### 5.3 New test files

| File | Tests | Covers |
|---|---:|---|
| `tests/test_control_identity_config.py` | 39 | C-01 … C-07: canonical JSON, UUIDv7, variant collisions, cross-process stability under `PYTHONHASHSEED`, field-level validation, structural vs operational hashes, round-trip, suite compatibility, capability honesty, path safety |
| `tests/test_control_registry_events.py` | 36 | R-01 … R-06: migrations, WAL/FK pragmas, transition validity, terminal finality, ORPHANED semantics, optimistic concurrency, threaded SQLite, GPU lease exclusivity, concurrent immutable allocation, event monotonicity, partial-tail recovery, mid-file corruption, cursor pagination, payload bounds, observer failure isolation |
| `tests/test_control_process_telemetry.py` | 24 | P-01 … P-06, T-01 … T-03: real subprocess lifecycle, process group, launcher-death independence, heartbeat, stdout/stderr capture, ordinary failure + traceback artifact, SIGKILL → ORPHANED, PID-reuse defence, live events before terminal, CPU-only telemetry, NVIDIA telemetry |
| `tests/test_control_api_dashboard.py` | 39 | A-01 … A-06, U-01 … U-10: health/GPU/capability endpoints, run list/detail/filters/bounds, cursor pagination, bounded logs, no write routes, bind safety, Dash routing, runs table, run detail, system page, button gating, offline-API degradation, layering isolation |
| `tests/test_control_legacy_import.py` | 22 | Read-only guarantee (byte-for-byte tree fingerprint), deterministic UUIDv5 ids, idempotency, status confidence levels, unknown-field reporting, legacy checkpoints not resume-certified, real `runs/` slice import |

Verified against a **real** KalmanNet suite run through the control plane
(`gpu_figure_pack_smoke`, `F5aP_gpu_m2n2_T50_invR2db_0`, CPU, 10 updates):
state `COMPLETED`, `global_step=10`, and the journal contained
`loss/train_total` ×10, `loss/validation_total` ×2, `metric/test_mse`,
`metric/test_mse_db`, `metric/test_rmse`, `latency/update_ms`, phase events for
`setup`/`train`/`test`/`report`, and resource samples.

### 5.4 Environment caveat on browser tests

No Chrome/Chromium/Firefox and no webdriver exist on this host, so
`dash[testing]`, Selenium, and Playwright cannot run. Dash is exercised
**server-side**: page routes over HTTP, and callbacks dispatched through Dash's
own `_dash-update-component` endpoint (the same code path a browser triggers).
This validates routing, layout construction, data binding, and callback output.
It does **not** validate client-side JavaScript or visual rendering. No
duplicate browser-testing dependency was added.

---

## 6. Compatibility with the existing CLI

* `bench/runners/run_suite.py` still parses the same suite YAML, still writes the
  same artifacts, and its `run_one` signature is unchanged.
* The suite YAML format is unchanged. The compatibility layer is a one-way
  projection; nothing rewrites a config file.
* Adapter instrumentation is additive and inert without a control-plane worker:
  `active_observer()` returns a `NullObserver`, whose methods do nothing. Each
  adapter's import is `try/except`-guarded, so `bench.control` is not a hard
  dependency of the training code.
* The Streamlit Run Inspector was **not modified**. It already accepted a
  `?run=` query parameter (`viz/app/components/overlay_picker.py:147`,
  `_run_from_query`), matching an absolute run directory or one relative to its
  runs root; the dashboard simply emits that value. This was the lowest-risk
  possible integration and required no change to `viz/`.
* All 425 baseline tests still pass, including the 28-variant identity gate.

---

## 7. Instrumented model coverage

`event_instrumentation` is declared per adapter in `bench/control/capabilities.py`
and rendered as a badge in the dashboard.

| model_id | implementation_id | Instrumentation | What is emitted |
|---|---|---|---|
| `kalmannet_tsp` | `bench_kalmannet_tsp_adapter_v1` | **step** | per-update `loss/train_total`, per-eval `loss/validation_total`, train phase start/end, checkpoint artifact |
| `split_knet` | `bench_split_adapter_v1` | **step** | same as above |
| `mb_kf` (`oracle_kf`, `nominal_kf`, `oracle_shift_kf`, `mb_kf_oracle`, `mb_kf_nominal`) | `bench_mb_kf_adapter_v1` | **step** (complete: no training loop exists) | `PHASE_SKIPPED` for train, plus runner phase events and final metrics |
| `adaptive_knet` | `bench_adaptive_knet_adapter_v1` | **phase** | runner phase boundaries + final metrics only |
| `maml_knet` | `bench_maml_knet_adapter_v1` | **phase** | runner phase boundaries + final metrics only |
| `me_split_knet_v0` (+6 aliases) | `bench_me_split_adapter_v0` | **phase** | runner phase boundaries + final metrics only |
| `basilisk_mrp_ekf` | `bench_basilisk_mrp_ekf_adapter_v1` | **phase** | runner phase boundaries + final metrics only |
| `spike_split_knet`, `g1_snn_split_knet`, `spike_ra_knet` | aliased to split/kalmannet implementations | **step** (inherited) | inherited from the aliased adapter |

Every model gets **at least** phase-level coverage, because the runner-level
hooks in `_try_call_setup/_try_call_train/_try_call_eval` are adapter-agnostic.
Final metrics for suite runs are lifted from `run_one`'s **return mapping** — a
structured object — never from parsed stdout (DND-006).

Not instrumented at step level, and why:

* **`adaptive_knet`** — the adapt phase constructs its own optimizer inside the
  adapter; exposing its per-update losses needs an adapt-phase step vocabulary
  that this tranche does not define.
* **`maml_knet`** — inner/outer loops need a two-level step axis
  (`outer_step` × `inner_step`). Emitting both against a single `global_step`
  would produce charts that are quietly wrong.
* **`me_split_knet_v0`** — two sequential training phases with separate
  optimizers need a per-phase step namespace.

## 8. Paper fidelity and resume certification

| model_id | `paper_fidelity_status` | `supports_exact_resume` |
|---|---|---|
| `kalmannet_tsp` | `unverified` | **false** |
| `split_knet` | `partial` — single Adam; the paper's alternating optimization of the two split heads is **not** implemented | **false** |
| `adaptive_knet` | `unverified` | **false** |
| `maml_knet` | `unverified` | **false** |
| `me_split_knet_v0` | `not_applicable` — project extension, not a reproduction | **false** |
| `mb_kf`, `basilisk_mrp_ekf` | `not_applicable` — no learned procedure | **false** |

No adapter is marked `verified` merely because it executes (DND-013), and a test
(`test_no_implementation_claims_exact_resume`) fails the build if any
implementation ever claims exact resume without the corresponding parity test.

---

## 9. Legacy import coverage

* Discovery: any directory under `runs/` containing `run_plan.json`,
  `metrics.json`, `meta.json`, or `failure.json`; each directory yielded once.
  `_model_cache` and `_quarantine*` trees are skipped.
* Verified against the repository's real tree: **75 runs imported from a 150-directory
  slice** in ~3 s; a 20-run import completed with zero errors.
* Status inference and confidence:

| Evidence | State | Confidence |
|---|---|---|
| `failure.json` present | `FAILED` | high |
| `metrics.json` with `status` in {ok, pass, success, completed} | `COMPLETED` | high |
| `metrics.json` with another status | `FAILED` | medium |
| `metrics.json` with no status field | `COMPLETED` | medium |
| viz `meta.json` only | `COMPLETED` | medium |
| `run_plan.json` only | `ORPHANED` | low |
| nothing recognizable | `ORPHANED` | unknown |

* Records carry `legacy = 1`; missing fields are recorded as `unknown` in
  `unknown_fields` rather than invented.
* Checkpoints found on disk are **not** registered in the checkpoint catalog and
  are never marked resume-certified.
* Idempotent: the synthetic `run_id` is UUIDv5 over the absolute path.
* Read-only, enforced by a test that fingerprints every file in the tree before
  and after an import and asserts byte equality.

---

## 10. Not implemented (deliberately)

Present in the schema so a later tranche needs no migration, but **not** a
feature of this build, and not exposed anywhere in the UI or API:

| Capability | Schema present | Code | UI |
|---|---|---|---|
| Exact resume | `ResumeSection`, `checkpoints.exact_resume_certified`, `resumed_from_*` columns | none — `resume.mode != "none"` is a validation error | absent |
| Graceful stop | `STOP_REQUESTED`, `CHECKPOINTING` states; `run_actions` table; `STOP_GRACEFUL` vocabulary | none | absent |
| Force terminate / kill | `signal_process_group` helper exists | not wired to any endpoint | absent |
| Warm start API | `InitializationSection.checkpoint_uri` | config-level only | absent |
| Checkpoint v1 (atomic write, manifest, retention) | `checkpoints` table + reader | no writer | catalog shows "none registered" |
| Config GUI / launch from UI | — | CLI only | absent |
| Multi-GPU queue / scheduler | `gpu_leases` table + exclusivity index | lease API exists, no scheduler | absent |
| Shared GPU execution | — | none | absent |
| Multi-user / auth | — | none | `authentication: false` declared |
| WebSocket streaming | event cursor semantics are complete | none — bounded polling only | absent |
| React frontend | FastAPI boundary preserves the option | none | — |

`/api/v1/capabilities` reports every one of these as `false`, and the System page
renders that list. **No disabled buttons exist**: a control that cannot work is
not drawn at all.

---

## 11. Bugs found and fixed during implementation

Four real defects, all found by running the code rather than by reading it:

1. **`HeartbeatThread._stop` shadowed `threading.Thread._stop()`**
   (`process/worker_cli.py`). `Thread.join()` calls the private `_stop()`
   internally, so every join raised `TypeError: 'Event' object is not callable`.
   Effect: the worker recorded FAILED correctly but then crashed in its `finally`
   block and exited **1 instead of 40**, breaking the exit-code contract.
   Fixed by renaming to `_stop_event`.

2. **Zombie processes were counted as alive** (`telemetry/cpu.py`).
   `psutil.pid_exists` and `kill(pid, 0)` both report a zombie as existing. A
   SIGKILLed worker becomes a zombie child of the manager until reaped, so
   `find_orphan_candidates` saw it as alive and a dead run would have stayed
   `RUNNING` forever, never becoming `ORPHANED`. Fixed by treating
   `STATUS_ZOMBIE` as dead and adding `WorkerManager.reap()`, called at the start
   of orphan detection.

3. **`original_config` was dropped by `ResolvedRunSpec.as_dict()`**
   (`config/schema.py`). The worker reads only `resolved_run_spec.json`, so the
   `SuiteExecutor` could not reconstruct its task/model entries and **every real
   suite run failed immediately**. Fixed by serializing it (with a
   `json.dumps(default=str)` coercion for YAML dates/tuples). Covered by
   `test_original_config_survives_the_spec_round_trip`.

4. **Deadlock in the API dependency layer** (`api/deps.py`). `get_manager()`
   held a non-reentrant `threading.Lock` and then called `get_registry()`, which
   tried to take the same lock. It triggers only when both singletons are unset —
   i.e. on the **first request after start-up** if that request depends on the
   manager before the registry, which `/api/v1/orphan-candidates` does. The API
   server would hang forever. Fixed with `threading.RLock`.

Pre-existing baseline issues (not introduced here, not fixed here):

* **B-BUG-01** — `bench/tests/test_report_schema_guardrails.py:11` uses
  `from .test_plan_matrix_minimal import ...`, but `bench/tests/` has no
  `__init__.py` (the repo uses `init.py` without dunders throughout). The module
  is uncollectable under pytest and was uncollectable at baseline. Fixing it
  means either adding `bench/tests/__init__.py` (changes package semantics
  repo-wide) or editing an existing test — both outside this tranche's mandate.

---

## 12. Acceptance criteria

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Two runs of the same config get different `run_id` and path | ✅ | `test_identical_config_gets_distinct_directories`, `test_concurrent_allocation_of_identical_configs` (20 threads → 20 dirs) |
| 2 | Runs execute in a separate process | ✅ | `test_worker_runs_in_its_own_process_group` (pid == pgid ≠ manager pgid) |
| 3 | Killing FastAPI/Dash does not kill the worker | ✅ | `test_worker_survives_the_death_of_its_launcher`; also the API/Dash are read-only and never the worker's parent |
| 4 | State and events recoverable after a server restart | ✅ | `test_state_survives_a_fresh_registry_connection`; all state is in SQLite + JSONL, none in process memory |
| 5 | Metric/log/resource visible before terminal | ✅ | `test_metrics_logs_and_resources_appear_before_the_terminal_state` asserts state is still `RUNNING` when all three event kinds are present |
| 6 | Worker exception → FAILED + traceback | ✅ | `test_ordinary_failure_is_recorded_with_traceback` (exit 40, `failure.json`, `artifacts/traceback.txt`, `failure` event) |
| 7 | Abrupt death not mistaken for completion | ✅ | `test_abrupt_death_is_never_reported_as_completed` (SIGKILL → stays `RUNNING`, never `COMPLETED`, then `ORPHANED` with `status_confidence=unknown`) |
| 8 | Existing Streamlit Inspector unbroken | ✅ | `viz/` unmodified; 28-variant regression green; full suite green |
| 9 | Exact-resume / Stop buttons not wrongly enabled | ✅ | `test_no_stop_resume_or_launch_controls_exist_anywhere`, `test_no_write_endpoints_are_exposed`, `test_no_implementation_claims_exact_resume` |
| 10 | New tests + existing regressions reported | ✅ | §5 |

---

## 13. Recommended next tranche, in order

1. **Checkpoint v1** — atomic write (`tmp → fsync → hash → rename → fsync dir →
   registry row → event`), `CheckpointManifest`, and the compatibility payload.
   Everything else on the resume path depends on this, and the ordering rule
   (payload before catalog row, DND-011) must be established before any writer
   exists. Registry table and reader are already in place.
2. **Warm start / resume API separation** — expose warm start (weights only,
   optimizer/RNG/cursor reset) as an explicitly *different* operation from
   resume, with `parent_run_id` lineage. Do this before any resume UI so the two
   can never be confused (DND-003).
3. **KNet + Split exact-resume certification (E-01)** — the continuous-vs-resumed
   parity harness, with declared device/precision/tolerance bounds, gated on
   `optimizer_update` boundaries. Only after it passes may
   `supports_exact_resume` be flipped for those two implementation ids — and
   only for those two (DND-008).

Deferred beyond that: graceful stop state transitions, config GUI/launch,
checkpoint/resume control UI, and (only if a multi-user need actually appears) a
React frontend.
