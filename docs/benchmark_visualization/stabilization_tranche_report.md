# Benchmark Execution Visualization — Stabilization Tranche Report

## 1. Executive Verdict

**READY_FOR_NEXT_TRANCHE.**

All three release blockers are closed with executed evidence, not argument:

- **V-001 CLOSED** — the control plane, its tests, and its documents are now in
  commits `355b6ca`, `179a42d`, `ee862a2` (plus this document set). A clean
  worktree cut from that commit installs, imports, serves, and passes.
- **V-002 CLOSED** — `python -m pytest --collect-only -q` and `python -m pytest -q`
  both succeed at the repository root with no exclusion flags. No test was
  deleted, ignored, skipped, or xfailed.
- **V-003 CLOSED** — observer and telemetry are proven numerically inert on the
  real `kalmannet_tsp` and `split_knet` adapters, against the real third-party
  model code, with **bitwise** equality across all four on/off combinations.

The non-blocking findings are also closed: V-004 fixed and re-verified against a
genuinely live worker, V-005 characterized and documented, V-006 cleaned without
touching vendored source.

Nothing outside the tranche scope was implemented. Graceful stop, force
terminate, exact resume, warm-start API, config GUI launch, GPU queue/lease
scheduling, shared-GPU execution, and authentication all remain unimplemented and
declared `false`.

Two things the next tranche should know, neither blocking:

1. A clean checkout collects **449** tests; the working tree collects **590**.
   The 141-test gap is 27 untracked test files belonging to unrelated in-progress
   user work (ADCS, Phase 5c–7 replay/checkpoint, Vizard, spike/SNN models) that
   was correctly left uncommitted. That work is still uncommitted and still at
   risk — see §19.
2. Three tracked tests were failing on **every** clean checkout, including the
   verified baseline `d1cc4b0` itself. This had never been observed because the
   suite had only ever been run in a dirty tree. Fixed in `ee862a2`.

## 2. Authoritative Checkout and Provenance

Confirmed before any modification:

| Check | Value |
|---|---|
| `pwd` / `git rev-parse --show-toplevel` | `/home/dss-pc-05/bench` |
| `git rev-parse HEAD` | `d1cc4b035597bb029e0bce95a546b29b4664b5c6` |
| `git branch --show-current` | `benchmark-viz/stabilize-release-baseline` |
| `bench/control/**` | present (38 modules) |
| `bench/ui/**` | present (8 modules) |
| control tests | present (5 files, untracked) |
| FastAPI app | `bench/control/api/app.py` |
| Dash app | `bench/ui/dash_app.py` |
| `pyproject.toml` `control` extra | present |
| worktrees | single, `/home/dss-pc-05/bench` |
| submodules | 4, all at expected commits |

This is the correct checkout. `WRONG_CHECKOUT_OR_LOST_WORKTREE` does not apply.
The superseded `/home/xynus/bench` verification was not consulted and nothing was
re-implemented on its basis.

## 3. Preflight Safety Snapshot

Written before the first change, to
`artifacts/benchmark_visualization_stabilization/20260730T182502Z/preflight/`:

`git_status.txt` (113 KB), `git_diff.patch` (1.4 MB), `git_diff_cached.patch`,
`untracked_files.txt` (230 entries), `submodule_status.txt`,
`file_inventory.txt` (2.2 MB), and the pre-commit full-suite log.

No `git reset --hard`, `git clean`, `git stash pop`, forced checkout, or bulk
delete was used at any point.

## 4. Working-Tree Change Classification

230 untracked paths and ~857 modified tracked paths were classified. Summary by
category; per-file evidence is in the preflight inventory.

| Path | Git state | Category | Why it belongs | Commit? | Evidence |
|---|---|---|---|---|---|
| `bench/control/**` (38) | untracked | A/B Phase 0+1 | registry, events, process, telemetry, API | **Yes** | control targeted tests import all of it |
| `bench/ui/**` (8) | untracked | C Phase 3 | Dash pages + API client | **Yes** | browser smoke drives it |
| `bench/models/kalmannet_tsp.py`, `split_knet.py`, `mb_kf.py` | modified | A obs. instrumentation | adds `active_observer()` phase/metric/artifact calls only | **Yes** | diff is additive; parity test proves inertness |
| `bench/runners/run_suite.py` | modified | A obs. instrumentation | runner-level setup/train/test phase boundaries | **Yes** | diff is additive |
| `tests/test_control_*.py` (5) | untracked | D | control/API/UI/identity/legacy tests | **Yes** | 163 passed |
| `tests/test_control_real_adapter_numerical_parity.py` | untracked | D | V-003 gate | **Yes** | §9, §10 |
| `bench/tests/test_report_schema_guardrails.py` | modified | D | V-002 relative-import fix | **Yes** | §7 |
| `pyproject.toml` | modified | E | `control` extra + pytest collection config | **Yes** | §7 |
| `runs/viz_v4a_fixtures`, `viz_v4c_cross_models`, `viz_v4c_fixtures` | untracked/ignored | D fixtures | three tracked tests read them | **Yes** | §7.2 |
| `docs/benchmark_visualization/*.md` | untracked | E | architecture/operator/verification docs | **Yes** | this set |
| `bench/models/registry.py` | modified | **H unrelated** | registers spike/SNN adapters whose modules are untracked | **No** | diff adds `spike_*`/`g1_snn_*` only |
| `bench/models/adaptive_knet.py`, `maml_knet.py` | modified | **H unrelated** | *removes* `preds_test.npz` writing — a behaviour regression, not observability; `run_suite.py` does not write it centrally | **No** | diff inspected |
| `bench/tasks/bench_generated.py`, `generator/basilisk_imu_adcs.py` | modified | **H unrelated** | ADCS replay task families | **No** | diff inspected |
| `bench/tests/run_all.py`, `test_runner_smoke.py`, `test_kf_baseline_smoke_plan.py` | modified | **H unrelated** | wires Phase 5c–7 / pred-artifact-meta work | **No** | diff inspected |
| 27 × `bench/tests/test_{adcs,phase5c,phase6*,phase7,vizard,replay,checkpoint,spike}*.py` | untracked | **H unrelated** | separate work streams | **No** | §6 delta analysis |
| `bench/visualization/*` (except `pred_artifact.py`) | untracked | **H unrelated** | modules for the above | **No** | no tracked test imports them |
| `bench/models/spike_*.py`, `g1_snn_split_knet.py` | untracked | **H unrelated** | SNN research | **No** | — |
| `bench/configs/suite_*.yaml` (15) | untracked | **H unrelated** | user experiment configs | **No** | — |
| `DECISIONS.md`, `USERGUIDE.md` | modified | **H unrelated** | user edits | **No** | — |
| `runs/**`, `reports/**` (~850) | modified/deleted | **H unrelated** | user run/report output | **No** | — |
| `*.pyc`, `__pycache__` | modified/untracked | **G generated** | build cache | **No** | — |
| `artifacts/benchmark_visualization_*/**` | untracked | **F verification-only** | raw evidence | **No** | — |
| root `0*_*.md`, `AUDIT_*.md`, `VIZ_*.md` | untracked | **I ambiguous** | user planning notes, not implementation | **No** | left in place |

The dependency boundary was checked mechanically, not assumed: `bench/control`
and `bench/ui` import nothing from `bench/` outside themselves except
`bench.runners.run_suite`, and no tracked test imports any untracked
`bench.visualization` module (only the tracked `pred_artifact`).

No deleted file was staged. No unrelated user change was modified, reverted, or
committed.

## 5. Reproducible Commit Baseline

Four commits on `benchmark-viz/stabilize-release-baseline`, staged by explicit
path only. `git add -A` and `git commit -am` were never used; `git diff --cached
--check` was clean.

| Commit | Content |
|---|---|
| `355b6ca` | `feat`: control plane, UI, observer instrumentation, `control` extra, pytest collection config (56 files) |
| `179a42d` | `test`: guardrails import fix + real-adapter parity test |
| `ee862a2` | `fix`: track 102 viz fixture files three tracked tests require |
| *(this set)* | `docs`: baseline, tranche report, summary JSON, operator update |

`git ls-tree -r HEAD` confirms 46 files under `bench/control/` and `bench/ui/`.
No verification artifact, temp DB, temp run, log, screenshot, `__pycache__`, or
`.pytest_cache` is in any commit.

## 6. Clean-Worktree Installation Result

Worktree `/tmp/bench-viz-stabilization-179a42d`, detached at `ee862a2`,
submodules initialized recursively to the four expected commits.

- `import bench.control, bench.ui, bench.control.api.app, bench.ui.dash_app` → OK
- `python -m bench.control.cli --help` → OK (subcommands `launch`,
  `launch-synthetic`, `list`, `show`, `import-legacy`, `reconcile`)
- FastAPI and Dash both start and serve (§13, §14)
- `git status --short` clean apart from ignored generated files

The environment is `pyenv` 3.10.13; see `reproducible_release_baseline.md` §5 for
why bare `python` does not resolve on this machine and what to do about it. That
is a machine configuration issue, not a repository defect, and is documented
rather than silently patched into the repo.

## 7. Pytest Collection Repair

### 7.1 The reported error

Reproduced first:

```
bench/tests/test_report_schema_guardrails.py
ImportError: attempted relative import with no known parent package
```

`from .test_plan_matrix_minimal import run_plan_matrix_minimal` only resolves when
the module is imported as part of the `bench.tests` package. Under direct
collection it has no parent. Fixed by using the absolute
`from bench.tests.test_plan_matrix_minimal import ...` that the rest of the
package already uses — the smallest correct fix, and one of the explicitly
allowed options.

The other half is `[tool.pytest.ini_options]` in `pyproject.toml`. Bare `pytest`
from the root otherwise descends into `third_party/` and collects three vendored
upstream helpers — `KalmanNet_TSP/Filters/{KalmanFilter,EKF}_test.py` and
`Adaptive-KNet-ICASSP24/filters/KalmanFilter_test.py` — which fail to import
because they expect their own repo root on `sys.path`. Vendored code must not be
modified, so collection is scoped to `bench/tests` and `tests`.

This was verified not to be a discovery-narrowing trick: exactly 3 files are
excluded, all vendored, and **no directory containing our tests is excluded**.
Collected file sets were diffed between configurations; nothing of ours
disappeared.

Nothing was deleted, `--ignore`d, skipped, xfailed, or wrapped in a
swallowing `try/except`.

### 7.2 Self-contained fixture audit (and what it caught)

The audit required by this step found a real defect the previous verification had
missed. Three tests in the tracked `tests/test_viz_release_readiness.py` read run
fixtures that were **not tracked**:

```
runs/viz_v4a_fixtures/failed_train_nan
runs/viz_v4c_cross_models/A_linear_split_train_smoke_v0/...
runs/viz_v4c_fixtures/combined_only
```

They passed only on a machine that happened to have those runs on disk. To
attribute this correctly rather than assume, a worktree was cut at the verified
baseline `d1cc4b0` itself and the file was run there: **the same 3 tests fail
identically**. This is pre-existing and unrelated to the control plane; it was
invisible because the suite had only ever been run in a dirty tree.

Fixed in `ee862a2` by tracking the 102 fixture files (~590 KB of small
`meta.json` / `aggregate.npz` / `traj_*.npz`). `runs/` is in `.gitignore`, but the
repository already force-tracks 17,540 files under it precisely because tests
need them, so this follows existing practice rather than inventing it. Adding a
conditional skip would have been the forbidden fix and was not considered.

Also checked: control tests create their DBs/runs under `tmp_path`; the parity
test builds its own fixtures in-memory; submodule-dependent tests are covered by
the documented `git submodule update --init --recursive` step; no test is skipped
for a platform/GPU reason it does not genuinely need.

### 7.3 Result

| Command | Working tree | Clean worktree @ `ee862a2` |
|---|---|---|
| `pytest --collect-only -q` | 590 collected, **0 errors** | 449 collected, **0 errors** |
| `pytest -q` | **588 passed, 2 skipped** | **448 passed, 1 skipped** |

Against the 585-passed baseline the working tree is +3 (the guardrails file now
collects, plus 2 parity tests) and nothing vanished. Skips went **down** (2 → 1)
in the clean worktree because a fixture became available — no new skip was
introduced.

## 8. Full Regression Results

All in the clean worktree at `ee862a2`:

| Suite | Result |
|---|---|
| `pytest --collect-only -q` | 449 collected, 0 errors |
| `pytest -q` | 448 passed, 1 skipped, 0 failed, 12 subtests passed |
| `tests/test_viz_init_provenance_comparison.py` | **28 passed** |
| `tests/test_control_*.py` | **163 passed** (baseline 160) |
| parity, KNet only | 1 passed |
| parity, Split only | 1 passed |

## 9. KNet Numerical Parity

`tests/test_control_real_adapter_numerical_parity.py`, real
`KalmanNetTSPAdapter`, real `third_party/KalmanNet_TSP/KNet/KalmanNet_nn.py`
(module load confirmed by probing `sys.modules` after the run, not assumed).

CPU / fp32 / seed 17 / `torch.use_deterministic_algorithms(True)` / 1 intra- and
inter-op thread / no DataLoader workers / fixed batch order / own `tmp_path` per
mode. Every mode rebuilds the adapter and re-seeds; no mode inherits another's
state.

| Mode | Observer | Telemetry | vs A |
|---|---|---|---|
| A | off | off | baseline |
| B | **on** | off | identical |
| C | off | **on** | identical |
| D | **on** | **on** | identical |

Compared and equal: dataset fingerprint, initial `state_dict` hash, final
`state_dict` hash, prediction hash, optimizer update count, best step, final train
loss, full validation history. Equality is **bitwise** — SHA-256 over raw tensor
bytes, not `allclose`. No tolerance was introduced or widened.

Allow-listed as excluded: timestamps, wall-clock, PID/PGID, run id, run
directory, heartbeat times, resource samples.

## 10. Split-KNet Numerical Parity

Same harness, real `SplitKNetAdapter` against
`third_party/Split_KalmanNet/GSSFiltering/{dnn,filtering,model}.py` (load
confirmed), seed 23, `Split_KalmanNet_Filter` estimator. A/B/C/D all bitwise
identical on the same field set.

### Proving the test is not vacuous

A parity test that passes because it compares nothing is worse than no test. Two
mutation probes were run against the committed test:

1. **Observer consumes RNG** (`torch.rand(1)` inside the observer callback) →
   still passed. Investigated rather than accepted: after `setup()` the training
   path draws no RNG, so an extra draw genuinely cannot perturb it. The probe was
   inert, not the assertion.
2. **Seed perturbed when the observer is on** → **both adapters FAIL**, on
   `initial_hash` and `final_hash`, with differing digests.

So the equality assertions do bite on real numerical divergence. Probe retained
at `parity/mutation_probe_seed_perturbation.py`.

## 11. Observer and Telemetry Semantic Validation

Verified by reading the committed instrumentation and by the test's assertions:

- **Post-update semantics.** `_observer.metric("loss/train_total", ...)` is called
  *after* `optimizer.step()` and `updates_used += 1`, with `step=updates_used`.
  `global_step` therefore equals the completed-update count, not a pre-update
  index.
- **Validation 1:1.** `loss/validation_total` is emitted inside the
  `should_eval` branch immediately after the actual validation call, once per
  call, and `val_history` and the emitted events agree.
- **Silent when off.** Modes A and C record zero observer metrics and zero
  statuses; B and D record both, including `loss/train_total` and
  `loss/validation_total`.
- **No mode/RNG side effects.** The observer performs no forward or eval pass and
  never touches `.train()`/`.eval()`; the instrumentation diff is purely
  additive. Bitwise parity is the empirical confirmation.
- **No mutable aliasing.** Values are coerced with `float(...)` at the call site,
  so no live tensor reference is retained.
- **Terminal agreement.** `PHASE_END` reports the same `updates_used`/`best_step`
  written to `train_state`, and an artifact event records the checkpoint path.
- **Optional by construction.** `active_observer()` falls back to a null object if
  `bench.control` is unavailable, so the existing CLI keeps working and gains no
  hard dependency.

Telemetry: the sampler runs in its own thread over process/system collectors, never
reads model tensors or RNG; GPU fields report unavailability on a CPU-only host
rather than fabricating zeros; collector failure is recorded as a structured
collector error and does not fail the run. Modes C and D produce samples tagged
with the correct `run_id`, and produce no numerical change.

## 12. Worker-State Freshness Fix

The reported symptom was `run_state=RUNNING` with `worker_state=STARTING` after an
API restart, which invites an operator to kill a healthy worker.

The fix is the smaller, schema-compatible correction: inside the **same
transaction** that moves a run to `RUNNING`, the worker row is advanced from
`STARTING` to `RUNNING` and given a heartbeat timestamp
(`bench/control/registry/sqlite.py`). It is narrowly scoped —
`WHERE run_id = ? AND state = 'STARTING'` — so it is not a blind copy of run
state onto worker state, and it does not promote anything on heartbeat evidence
alone. Nothing auto-kills a stale worker.

Verified against a genuinely live worker, which required care: an earlier attempt
used a run that finished before the API came back, making the check pass
vacuously. Re-run with a 6,000,000-update workload so the run was still
executing:

| Observation | Value |
|---|---|
| `run_state` after restart | `RUNNING` |
| `worker_state` after restart | **`RUNNING`** (not `STARTING`) |
| `worker.pid_alive` | `true` |
| worker survived API shutdown | yes |
| steps progressed while API was down | 248,800 → 461,650 |
| `run_id` stable across restart | yes |
| event cursor recovered | 21,325 → 657,215 |

PID start-time/token defence is retained. **V-004 CLOSED.**

## 13. Lifecycle E2E Revalidation

Clean worktree, each scenario in its own temporary `BENCH_CONTROL_ROOT`. No
production registry, run, checkpoint, or event journal was read or written.

| Scenario | Result | Evidence |
|---|---|---|
| Normal synthetic worker | **PASS** — `COMPLETED`, exit **0**, 27 events incl. `status`/`metric`/`log`/`resource`, artifacts + checkpoints present | `e2e/normal/` |
| API restart recovery | **PASS** — worker survived and progressed; same `run_id`; events recovered; worker not misreported (§12) | `e2e/restart/` |
| Ordinary exception | **PASS** — `FAILED`, exit **40**, traceback event recorded, no false `COMPLETED` | `e2e/failure/` |
| SIGKILL orphan | **PASS** — `RUNNING` → `ORPHANED` via `reconcile`, no false `COMPLETED` | `e2e/orphan/` |

For the SIGKILL case the worker's identity was verified before signalling — the
target PID's `/proc/<pid>/cmdline` was required to contain the run id, and its
`/proc/<pid>/stat` start-time token was recorded — so the test cannot kill an
unrelated process that inherited a recycled PID.

State transitions recorded in the registry for a normal run:
`CREATED → VALIDATING → QUEUED → STARTING → RUNNING → COMPLETED`, with actors
`control-plane`/`manager`/`worker`.

## 14. Browser and Legacy Regression

Real FastAPI + real Dash + Playwright Chromium against a temporary registry seeded
with one `COMPLETED`, one `FAILED`, and one `ORPHANED` run.

- Title `Benchmark Control Plane`; Runs table renders
- All three states visible in the DOM
- `read-only` marker present
- **Zero `<button>` elements on either the Runs or the Run Detail page** — no
  stop/cancel/kill/terminate/resume/launch/delete control is exposed
- Run Detail page loads (2,777 chars)
- Screenshots: `browser/runs.png`, `browser/run_detail.png`

Legacy regression:

- `viz.app.main` (Streamlit Run Inspector entry point) imports cleanly
- `viz.io.loader.load_run` opens a legacy fixture read-only (`VizRun`)
- 28-variant init-provenance regression passes
- `variant_label` remains presentation-only — `bench/control/identity.py` states
  so explicitly and it is not used as a persistent identifier
- All 223 collected `test_viz_*` items pass; the model-based visualization
  numerical non-regression holds

## 15. Local-Only Security Boundary

Already enforced in code, and confirmed:

- Both the API and the Dash app default to `--host 127.0.0.1`.
- `resolve_bind_host()` accepts only `127.0.0.1`, `localhost`, `::1`; any other
  bind is **refused** unless `BENCH_CONTROL_ALLOW_PUBLIC_BIND=1` is set
  explicitly. The refusal message names the reason: no authentication, and local
  filesystem paths in responses.
- Every route is `GET`. There is no POST/PUT/DELETE anywhere in the API, so the
  read-only claim is structural rather than conventional.

Absolute paths (`run_dir`, `registry.path`, `control_root`) **are** returned. Path
redaction was deliberately **not** applied in this tranche: it would change the
response contract that the Dash UI and the legacy deep-link both consume, for no
gain under the only threat model this build supports (single trusted operator on
loopback). The limitation is instead documented in the operator quickstart and is
visible in the health response, which reports `read_only: true` and the active
control root.

Recommended exposure model, unchanged: SSH tunnel, or an authenticated reverse
proxy. No authentication or multi-user support was implemented.

## 16. 1k/10k Scale Characterization

Temporary synthetic registries, never production. Full table in
`operator_quickstart.md` §11; raw data in `scale/scale_results.json`.

Headlines at **10,000 runs**: first page `GET /runs?limit=50` **4.76 ms p50 /
5.04 ms p95**; run detail **1.50 ms** (flat vs 1,000 runs); state filter 3.33 ms;
events cursor 1.20 ms; API peak RSS 40 MB; DB 9.72 MB.

Bounded-response checks:

- `limit` is capped server-side at 1000. `limit=5000` and `limit=100000` are
  **rejected with HTTP 422**, not silently served.
- The Dash Runs page renders `Showing 50 of 10000 runs` — 50 rows, ~6 KB of body
  text, 1.08 s load. The browser is never handed the full table.
- Event and log endpoints are cursor/limit bounded.

One endpoint scales with load rather than size: `/api/v1/system/health` went
13 ms → 110 ms from 1k to 10k. Rather than record that as a mystery, the cause was
isolated: `find_orphan_candidates()` probes every non-terminal run's PID, and the
synthetic fixture made 25% of runs `RUNNING` (2,500 "live" workers), which is not
a realistic operating state. Re-measured with 10,000 **all-terminal** runs, health
returns in **4.06 ms p50**. The cost tracks concurrent live runs, not registry
size. Documented; no code change, and no invented SLA.

## 17. Third-Party Isolation and Cleanup

Before: untracked `__pycache__` in `KalmanNet_TSP` (3 dirs) and `MAML_KalmanNet`
(2 files); `Adaptive-KNet-ICASSP24` and `Split_KalmanNet` clean.

13 generated cache directories were removed with a targeted `find -name
__pycache__ -prune -exec rm -rf`. `git clean` was not used.

**One correction, recorded rather than hidden.** That sweep also deleted
`cpython-39` `.pyc` files inside `MAML_KalmanNet` that upstream **tracks** as
real repository content, which briefly showed 5 files as deleted — a violation of
"do not modify third-party tracked source". It was caught immediately in the
post-cleanup status check and reverted with `git checkout -- .` inside that
submodule.

Final state: **all four submodules report zero dirty entries**, and all four
remain at their expected commits. No vendored tracked file was modified. Cleanup
is separate from the implementation commits and is not itself committed.

## 18. Finding Closure Matrix

| Finding | Previous status | Action | Verification | New status | Release blocking |
|---|---|---|---|---|---|
| **V-001** implementation uncommitted | OPEN (blocker) | 3 commits by explicit path; unrelated user work left untouched | clean worktree at `ee862a2`: import + CLI + API + Dash + 448 passed + full E2E | **CLOSED** | was yes |
| **V-002** bare pytest collection fails | OPEN (blocker) | absolute import + scope collection away from vendored `*_test.py` | 0 collection errors and 0 failures in both trees; no test removed/skipped | **CLOSED** | was yes |
| **V-003** real-adapter parity unverified | OPEN (blocker) | added CPU-deterministic A/B/C/D parity test on real KNet + Split | bitwise equal; mutation probe confirms the test fails on real divergence | **CLOSED** | was yes |
| **V-004** stale worker state after restart | OPEN | transactional worker-row advance on `RUNNING` | live 6M-update run: `run=RUNNING`, `worker=RUNNING`, progressed 248k→461k | **CLOSED** | no |
| **V-005** path exposure / no scale baseline | OPEN | confirmed loopback-only + GET-only; measured 1k/10k; documented | 422 on oversized limits; 50-of-10000 in UI; §15–16 | **DOCUMENTED** | no |
| **V-006** submodule `__pycache__` | OPEN | removed untracked caches only; reverted an over-broad deletion | all 4 submodules clean, commits unchanged | **CLOSED** | no |
| **V-007** *(new)* tracked tests need untracked fixtures | — | tracked 102 fixture files | 3 tests failed on clean `d1cc4b0`, pass at `ee862a2` | **CLOSED** | would have been |

## 19. Remaining Risks

1. **The unrelated user work is still uncommitted.** 27 test files, their
   `bench/visualization` modules, the spike/SNN adapters, ADCS replay task
   families, and edits to `registry.py`, `run_all.py`, `adaptive_knet.py`,
   `maml_knet.py` all remain working-tree-only. They were correctly out of scope
   here, but they are exactly the V-001 failure mode over again and one bad
   `git clean` from gone. They deserve their own tranche.
2. **`adaptive_knet.py` / `maml_knet.py` currently drop `preds_test.npz`.** In the
   working tree these adapters no longer write the prediction artifact and return
   `preds_path: None`. Nothing writes it centrally instead. If that is intended it
   needs a legacy-contract decision; if not, it is a live regression. Uncommitted,
   so the baseline is unaffected.
3. **Fixtures live under an ignored directory.** `runs/` is gitignored while
   17,640 files under it are force-tracked. It works, but it is fragile and
   surprising; `tests/fixtures/` would be better.
4. **Parity covers the tiny-budget CPU path.** Three updates on a tiny fixture is
   enough to prove inertness of the hooks, not to certify long-run or GPU
   training. GPU was deliberately not used as an exact-parity gate.
5. **`/system/health` under many concurrent live runs.** Characterized (§16), not
   optimized.
6. **The user has a Dash app running on port 8802** against API 8801. Verification
   was moved to ports 18901+ to avoid disturbing it; it was left running.

## 20. Explicitly Deferred Features

Not implemented, and still declared `false` / not exposed:

graceful stop · force terminate API · exact resume · checkpoint v1 resume payload
· interrupt checkpoint · resume lineage execution · warm-start launch API · config
form / raw YAML GUI launch · GPU scheduler and lease enforcement · concurrent
shared-GPU runs · remote worker · multi-user / authentication · WebSocket
migration · frontend framework replacement · MAML/AKNet/ME-Split exact-resume
certification.

`supports_exact_resume` is `False` for every adapter entry, and
`supports_graceful_stop` is `False`. The capability schema refuses
`supports_exact_resume=True` without an explicit `resume_boundary`. The UI
exposes no action button of any kind (§14). Training is never executed from an
API or UI callback. SQLite/JSONL/filesystem retain their source-of-truth roles.

## 21. Final Gate Decision

**READY_FOR_NEXT_TRANCHE.**

V-001, V-002, V-003 are all closed on executed evidence. Normal, restart,
failure, and orphan E2E all pass; the Playwright DOM smoke passes; the legacy
28-variant regression and the Streamlit inspector are intact; no premature
control capability is exposed; and the committed baseline reproduces from Git
alone in a clean worktree.

The next tranche (checkpoint/stop design) can proceed. It should start by
committing the unrelated user work in §19.1, which is now the largest
uncommitted risk in the repository.

---

## Appendix A. Committed File Manifest

- `bench/control/**` — 38 files: `identity.py`, `canonical.py`, `provenance.py`,
  `capabilities.py`, `allocation.py`, `paths.py`, `cli.py`;
  `config/{schema,resolver,compatibility}.py`;
  `registry/{schema,sqlite,migrations}`;
  `events/{schema,writer,reader,observer}.py`;
  `process/{manager,executors,signals,worker_cli}.py`;
  `telemetry/{base,cpu,nvidia}.py`; `legacy/importer.py`;
  `api/{app,deps}.py` + `api/routers/{runs,system}.py`
- `bench/ui/**` — 8 files: `dash_app.py`, `api_client.py`, `components.py`,
  `pages/{runs,run_detail,system}.py`
- `bench/models/{kalmannet_tsp,split_knet,mb_kf}.py` — observer instrumentation
- `bench/runners/run_suite.py` — runner phase boundaries
- `bench/tests/test_report_schema_guardrails.py` — import fix
- `tests/test_control_{api_dashboard,identity_config,legacy_import,process_telemetry,registry_events}.py`
- `tests/test_control_real_adapter_numerical_parity.py`
- `pyproject.toml` — `control` extra + pytest collection config
- `runs/viz_v4a_fixtures/**`, `runs/viz_v4c_cross_models/**`,
  `runs/viz_v4c_fixtures/**` — 102 fixture files
- `docs/benchmark_visualization/` — this report, the summary JSON, the
  reproducible baseline, the updated operator quickstart

## Appendix B. Commands

```bash
# preflight
git rev-parse --show-toplevel && git rev-parse HEAD && git status --short --branch
git ls-files --others --exclude-standard && git worktree list && git submodule status

# baseline
git worktree add --detach /tmp/bench-viz-stabilization-<sha> <sha>
git submodule update --init --recursive

# mandated
python -m pytest --collect-only -q
python -m pytest -q
python -m pytest -q tests/test_viz_init_provenance_comparison.py
python -m pytest -q tests/test_control_*.py
python -m pytest -q tests/test_control_real_adapter_numerical_parity.py

# lifecycle (temporary control root)
BENCH_CONTROL_ROOT=<tmp> python -m bench.control.cli launch-synthetic --updates 8
BENCH_CONTROL_ROOT=<tmp> python -m bench.control.cli launch-synthetic --updates 10 --fail-at-step 3
BENCH_CONTROL_ROOT=<tmp> python -m bench.control.cli reconcile
BENCH_CONTROL_ROOT=<tmp> python -m bench.control.api.app --port 18901
BENCH_CONTROL_ROOT=<tmp> python -m bench.ui.dash_app --host 127.0.0.1 --port 18922 --api http://127.0.0.1:18921

# attribution of the pre-existing fixture failure
git worktree add --detach /tmp/bench-baseline-d1cc4b0 d1cc4b0
cd /tmp/bench-baseline-d1cc4b0 && python -m pytest -q tests/test_viz_release_readiness.py
```

## Appendix C. Raw Evidence Index

Root: `artifacts/benchmark_visualization_stabilization/20260730T182502Z/`
(not committed).

| Path | Contents |
|---|---|
| `preflight/` | git status, diffs, untracked list, submodule status, file inventory |
| `git/` | `commits.txt`, `baseline_to_head_stat.txt`, `committed_control_files.txt` |
| `pytest/` | clean collect + full suite, before/after the fixture fix, pre-commit working-tree run |
| `parity/kalmannet_tsp/`, `parity/split_knet/` | per-adapter runs; `mutation_probe_seed_perturbation.py` |
| `e2e/normal/` | `result.json`, `events.jsonl` |
| `e2e/restart/` | before/after JSON, `live_running_restart.json` |
| `e2e/failure/` | `result.json`, `traceback_event.json` |
| `e2e/orphan/` | `result.json` incl. reconcile output and PID start-time token |
| `browser/` | `result.json`, `body_runs.txt`, `runs.png`, `run_detail.png` |
| `scale/` | `scale_results.json`, `dash_scale.json`, `dash_10k.png` |
| `cleanup/` | V-006 before/after and the note on the reverted deletion |
