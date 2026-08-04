# Config GUI and Benchmark Launch — Tranche Report

Verdict: **`READY_FOR_GPU_QUEUE_AND_EXECUTION_POLICY_TRANCHE`**

Contracts: `config_gui_contract.md`, `launch_api_contract.md`.
Operator guide: `config_gui_operator_guide.md`.

## 1. What was built

| Area | File |
|---|---|
| Tracked-allowlist preset catalog | `bench/control/config/presets.py` |
| GUI schema descriptor over the existing dataclasses | `bench/control/config/descriptor.py` |
| Single validation / preview / diff path | `bench/control/config/gui_service.py` |
| Read-only config API | `bench/control/api/routers/config.py` |
| Durable `LAUNCH_RUN` coordinator | `bench/control/launch_coordinator.py` |
| `POST /api/v1/runs/launch` | `bench/control/api/routers/actions.py` |
| Registry migration 4 (nullable `run_actions.run_id`) | `bench/control/registry/migrations/__init__.py` |
| New run wizard | `bench/ui/pages/new_run.py`, `bench/ui/dash_app.py` |

## 2. Preset catalog

62 tracked presets, 50 launchable. `suite_adapt_smoke` is correctly reported
`MODEL_NOT_LAUNCHABLE` rather than hidden.

A `preset_id` is `<stem>.<12 hex of content digest>` — opaque, never a path — so
`../../etc/passwd`, `/etc/passwd` and friends resolve to nothing rather than
being defended against. An untracked file dropped into `bench/configs` is not
listed and not addressable. Symlink targets are checked **after** resolution.

Parser bounds: 512 KiB, depth 30, 20 000 nodes after alias expansion; `safe_load`
only, so `!!python/object/apply` is a parse error. A syntax error returns
`valid: false` with line and column, not a 500.

## 3. One config source of truth

The descriptor describes the dataclasses the resolver already uses; overrides
outside the descriptor are **refused**, not written. Form and raw YAML resolve
through the same `draft_from_suite` → `resolve_run_spec` path and produce the
same `structural_config_hash` and `variant_id`.

## 4. CLI / GUI parity

`tests/test_control_gui_cli_parity.py` allocates through
`bench.control.cli launch --dry-run` and through the GUI service and compares
resolved specs field by field.

| Preset / model | Fields compared | Differing |
|---|---|---|
| `suite_train_smoke` / kalmannet_tsp | 135 | 3 |
| `suite_split_train_smoke` / split_knet | 139 | 4 |
| `suite_all_simple_tiny` / mb_kf_oracle | 139 | 4 |

Every differing field is `experiment.experiment_id`, `identity.run_id`,
`hashes.resolved_spec_hash` (derived from those two), or
`provenance.environment_fingerprint`. Structural hash, operational hash,
`variant_id`, `implementation_id`, `training_path_id`, `certification_id`,
dataset identity, training budget, optimizer, runtime, system and
initialization all match exactly.

Numerical parity (`BENCH_PARITY_E2E=1`): tiny KNet and Split, launched once via
the CLI and once via the launch coordinator, produced **identical** runner
metrics after normalising wall-clock fields and run-local paths.

## 5. Two parity gaps found and fixed

**Executor was unset on the GUI path.** The CLI stamps
`bench_context.executor = "suite"`; the GUI left it `None` and relied on the
worker's default. Same behaviour, but the two stored specs described different
executions of the same config. Now stamped explicitly.

**GUI-launched runs had no repository provenance.** `git_commit`, `git_dirty`,
submodule revisions and the environment fingerprint were all `None`, so a run
started from the browser was not attributable to a repository state. The launch
path now captures provenance fresh at request time. Previews still skip it (it
shells out to Git and a preview may be recomputed on every keystroke), and since
provenance is not an identity input, capturing it late does not move any hash.

## 6. Launch API

Ordering is action → ACKNOWLEDGED → allocation → spawn → COMPLETED, which is
why migration 4 makes `run_actions.run_id` nullable: an action that *creates* a
run cannot name it in advance, and allocating first would leave an
unattributable run behind after a crash between the two writes.

Verified live (`artifacts/…/launch_api_session.json`):

```
same key ×5          → [202, 200, 200, 200, 200], 1 action, 1 run, 1 worker
stale preset digest  → 409, zero side effects
invalid model        → 422, zero side effects
provenance written   → original_preset.yaml, submitted_draft.yaml,
                       config_validation.json, launch_request.json
idempotency key      → absent from every file in the run directory
```

Restart boundaries: an action recorded but never settled is adopted by
`reconcile_open_actions` (one run, one worker; a second reconcile is a no-op);
an allocated-but-unlaunched run is adopted by `settle` rather than re-allocated.
A spawn failure marks the action FAILED and the run CANCELLED with
`exit_code 52` — never left looking live.

## 7. A UI defect found by the browser test

`GET /api/v1/actions/{id}` returned the raw `run_id`, which is NULL for a launch
action — the run it created lives in `result_child_run_id`. The POST response
patched it in, so the link appeared once and then **vanished on the next poll**,
leaving the operator with no handle on the run they had just started. Fixed in
`_action_resource`; pinned by
`test_polling_a_launch_action_still_names_its_run`.

## 8. Browser workflow (Playwright, Chromium)

| Step | Result |
|---|---|
| read-only `/new-run` offers no Launch control | 0 buttons |
| read-only page explains why | yes |
| preset → model → init → edit budget (600) | yes |
| Validate: valid, `control_resumable_v1`, hashes shown | yes |
| Validate side effects | **0 actions, 0 runs** |
| Double-click Launch | **1 action, 1 run, 1 worker** |
| Launch status links to the run | yes |
| Run reaches RUNNING | yes |
| Double-click Stop safely | **1 action** |
| Parent INTERRUPTED, `exit_code 10`, 1 checkpoint | yes (94 / 600 updates) |
| INTERRUPTED visible after refresh | yes |
| Resume training → child link | 1 action |
| Child COMPLETED, `exit_code 0`, 600 updates | yes |
| Force/Warm/Evaluate/Delete/Clone offered | none |
| Console errors | none |

Screenshots: `artifacts/benchmark_config_gui_launch/<ts>/playwright_shots/`.

## 9. Model coverage

| Model | GUI launch | Stop/Resume |
|---|---|---|
| `kalmannet_tsp` | yes (E2E COMPLETED) | yes |
| `split_knet` | yes (E2E COMPLETED, numerical parity) | yes |
| model-based filters (`mb_kf_oracle`, …) | yes (E2E COMPLETED, `not_applicable`) | no — no learning lifecycle |
| Adaptive / MAML / ME-Split | refused, `ADAPTER_NOT_GUI_LAUNCH_CERTIFIED` | — |

## 10. HTTP surface audit

Audited from the OpenAPI schema, not `app.routes`.

```
read-only build   POST /api/v1/config/validate            (preview only)
write mode        + POST /api/v1/runs/{id}/actions/stop
                  + POST /api/v1/checkpoints/{id}/actions/resume
                  + POST /api/v1/runs/launch
```

The read-only invariant is now stated as "**no mutating** routes are
registered", because a side-effect-free preview endpoint is a POST by virtue of
its request body, not by virtue of changing anything. The test says so
explicitly instead of asserting the absence of the verb.

## 11. Regression

```
python -m pytest --collect-only -q   →  639 tests collected
python -m pytest -q                  →  635 passed, 4 skipped, 12 subtests passed
```

No test was deleted, skipped, xfailed or ignored. One existing assertion was
**updated**: `test_schema_version_is_recorded` now expects registry schema
version 4 and migration list `[1, 2, 3, 4]`, with the reason in the comment.

Targeted groups, all passing: write API, graceful stop, worker stop wiring,
resume child, real-process fault/restart, KNet/Split numerical parity, exact
resume certification, checkpoint schema/atomicity, process telemetry, training
path selection (161 tests); 28 init-provenance tests.

One flake observed and not reproduced:
`test_control_process_telemetry.py::FailurePathTests::test_live_worker_is_never_marked_orphaned`
failed once when run in a group immediately after the real-training parity
tests, and passed in isolation, on re-run of the same group, and in both full
suite runs. It spawns a real worker and waits for RUNNING under a 0.001 s
heartbeat timeout, so it is timing-sensitive under CPU contention. It exercises
orphan reconciliation, which this tranche does not touch.

## 12. Third-party

```
git status --porcelain -- third_party   →  two submodules dirty
git diff -- third_party                 →  empty
```

Dirtiness is untracked `__pycache__` inside `KalmanNet_TSP` and
`MAML_KalmanNet` only — no tracked source modified, same as the documented
baseline.

## 13. Known limitations added

`known_limitations.md` §3.10 (GUI launch covers certified models only, no
sweeps/batch/queue), §3.11 (the preset catalog needs Git), §3.12
(`environment_fingerprint` depends on whether torch is imported in the capturing
process — pre-existing, surfaced by the parity test). §2.1 and §3.1 were
annotated where later tranches had made them stale.

## 14. What this makes urgent

A launch button does not create GPU contention, but it removes the friction that
was keeping it rare. Nothing acquires a GPU lease automatically today. The next
tranche should be GPU queueing and execution policy.
