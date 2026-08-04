# Write-Control Worker Wiring Report

```yaml
git_common_repository: /home/dss-pc-05/bench/.git
user_working_tree: /home/dss-pc-05/bench          # untouched, not used for work
feature_worktree: /tmp/bench-wc-tranche
branch: benchmark-viz/write-control
baseline_commit: 5ea4546
head_commit: 729629437e361d5a177cb81177607b03e27de6f8
control_root: temporary per-run roots under /tmp (never the production root)
verification_layer: production_worker
```

## 1. Executive Verdict

**READY_AFTER_SPECIFIC_FIXES.**

The blocker is closed and the headline gate passes: **KNet and Split both
achieve bitwise parity between a continuous reference run and a
parent→stop→resumed-child lineage, with every process launched by the real
`WorkerManager` as a separate OS process.**

Three items from §6.3, §9 and §10 were not executed, listed in §8 below.

## 2. Provenance

All required ancestry verified in the feature worktree: `c0eaaf7`, `48b0edb`,
`dfb932d`, `11a451d`, `de79d2a`, `a3ceff1`, `9eb4f08`, `7caeb04` (F-1 fix).
Baseline `5ea4546`; baseline suite reproduced at 534 collected / 533 passed /
1 skipped before any change.

New commits: `ba65524` (fix), `7296294` (test), plus this docs commit.

## 3. What was wired

See `production_worker_graceful_stop_contract.md`. Minimal connection of
existing pieces: no migration 4, no Checkpoint v3, no new action table,
service, worker manager or training loop.

## 4. Second bug found by the same E2E

Split's certification row was keyed on `bench_split_knet_adapter_v1`, but the
system derives `bench_split_adapter_v1` — `identity.py` uses that exact string
as its own docstring example. The row was therefore **unreachable**: Split
resolved to `legacy_train_v1` on every real launch, got no stop wiring, and
had never actually been certified in production despite passing every
service-level test.

Only the certification data was corrected. The identity derivation is untouched
because it feeds `variant_id`. A new guard test pins every certified
`implementation_id` to what `draft_from_suite` actually derives, so an
unreachable certification cannot be reintroduced.

## 5. KNet real-worker parity — PASS

```
A continuous : COMPLETED, 40 updates, sha256 7c10a4145ffdb1a6…
B parent     : INTERRUPTED, exit 10, stopped at K_actual = 21 (observed, not hard-coded)
C child      : COMPLETED, 40 updates, sha256 7c10a4145ffdb1a6…
```

**BITWISE PARITY: true.** Distinct PIDs for every process; child lineage
complete (`parent_run_id`, `resumed_from_run_id`, `resumed_from_checkpoint_id`);
`variant_id`, `training_path_id` and `structural_config_hash` inherited;
parent immutable (state, state_version, exit code and run-directory file list
unchanged); 5 identical resume requests produced exactly 1 child.

## 6. Split real-worker parity — PASS

```
A continuous : COMPLETED, 40 updates, sha256 a704f59ce873cae6…
B parent     : INTERRUPTED, exit 10, stopped at K_actual = 1
C child      : COMPLETED, 40 updates, sha256 a704f59ce873cae6…
```

**BITWISE PARITY: true**, same lineage and immutability checks. Split's updates
are fast enough that the stop is honoured at the first safe boundary; the
cursor used is the actual one, never hard-coded.

This certifies the current **single-optimizer** Split implementation. It is not
a paper-fidelity claim.

## 7. Regression and invariants

| Gate | Baseline | Now |
|---|---|---|
| `pytest --collect-only -q` | 534 | **541**, 0 errors |
| `pytest -q` | 533 passed, 1 skipped | **540 passed, 1 skipped, 0 failed** |
| 28 init-provenance | pass | **pass** |
| API methods | GET | **GET only** |
| Dash action buttons | 0 | **0** |
| Third-party tracked diff | empty | **empty** |

+7 tests. Nothing deleted, skipped, xfailed or ignored. All E2E used temporary
control roots; no production `runs/` or `reports/` were written.

## 8. Not executed

| Required item | Status |
|---|---|
| §6.3 real worker checkpoint-write failure → exit 50 | **not executed** |
| §9.1/9.2 two child-failure fault cases in a real child process | **not executed** |
| §10 coordinator restart while child RUNNING | **not executed** |

These are the specific fixes remaining. The code paths exist and are covered at
service level (`test_control_graceful_stop.py`, `test_control_worker_resume_child.py`),
but this tranche did not drive them through real child processes.

Real-process stop idempotency **was** verified (5 identical requests → 1 action,
1 checkpoint, 1 terminal transition), as was resume idempotency (5 requests →
1 child).

## 9. Final gate

**READY_AFTER_SPECIFIC_FIXES.** The numerical gate the API/UI tranche depends
on is closed for both certified models on real workers. What remains is fault
and restart coverage at the real-process level, not new capability.

## 10. Evidence

`artifacts/benchmark_write_control_worker_wiring/<timestamp>/` (not committed):
preflight git/environment/schema snapshot and pytest logs. Parity runs executed
under temporary control roots `/tmp/w19-*`.
