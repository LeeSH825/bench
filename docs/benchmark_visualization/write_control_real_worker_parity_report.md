# Write-Control Real-Worker Parity Report

> **Superseded for the blocker it identified.** F-2 was closed by commit
> `ba65524`; KNet and Split real-worker bitwise parity now both pass — see
> `write_control_worker_wiring_report.md` and
> `worker_level_exact_resume_certification.md`. This page keeps its original
> NOT_READY verdict as historical evidence and is not rewritten.

## 1. Executive Verdict

**NOT_READY.**

The micro-tranche ran the real-worker E2E first, before touching production
code, exactly as instructed. That ordering did its job: it found two things
that no unit test had, and the second one blocks the gate outright.

| Finding | Severity | Status |
|---|---|---|
| **F-1** `prepare_run()` never persisted the decided training path onto the run row | real bug | **FIXED** (`7caeb04`) |
| **F-2** `StopCoordinator` / `settle_graceful_stop` are not wired into the production worker at all | blocker | **OPEN** |

Because of F-2 a real parent worker cannot receive a stop request and cannot
write an interrupt checkpoint. The certification the gate asks for —
continuous reference vs `parent → stop → Checkpoint v2 → resumed child` bitwise
identical, both through `WorkerManager` OS processes — **cannot be executed
today**. It is not that the test failed; the production path it would exercise
does not exist yet.

This contradicts the implied readiness of the previous continuation, which
reported gate C (durable RESUME_EXACT + stop backend) as done. That report was
accurate about the *service layer* — `settle_graceful_stop` is implemented and
tested at service level, and the checkpoint tranche certified it there — but
nothing connects it to a running worker process.

## 2. Baseline and Provenance

| Item | Value |
|---|---|
| Authoritative repo (untouched) | `/home/dss-pc-05/bench`, `d92cd0c` |
| Branch | `benchmark-viz/write-control` |
| Worktree | `/tmp/bench-wc-tranche` (reused, safe) |
| Baseline tip | `9eb4f08` |
| New commit | `7caeb04` (fix + regression tests) |

Commit ancestry verified: `c0eaaf7`, `48b0edb`, `dfb932d`, `11a451d`,
`de79d2a`, `a3ceff1`, `9eb4f08` are all ancestors of the tip.

**`9e4fd8` does not exist in this repository.** The semantically corresponding
commit is **`9eb4f08`** — `docs(write-control): child/action contracts and
backend continuation report` — which is the branch tip and the docs commit the
continuation described. Recorded here as required rather than assumed.

Baseline before changes: **532 collected, 531 passed, 1 skipped, 0 failed**,
28 init-provenance passed. Matches the reported baseline exactly.

Environment: Python 3.10.13, torch 2.9.1+cu128, registry schema 3, checkpoint
schema 2 (reads 1 and 2), config schema 1. Submodules at expected revisions.

## 3. F-1 — Training path not persisted (fixed)

A real `WorkerManager` launch of a certified KNet suite entry produced:

```
resolved_run_spec.json : training_path_id = control_resumable_v1  ✔
event journal          : "adapter resumable_train: KalmanNetTSPAdapter
                          path=control_resumable_v1"               ✔
registry run row       : training_path_id = legacy_train_v1        �’ WRONG
```

So the resolver decided correctly and the worker genuinely executed the
resumable loop, but `prepare_run()` built the `RunRecord` without copying the
decision across. Resume eligibility is answered *from the registry*, so every
real control run would have looked ineligible for resume while in fact being
resumable.

The continuation tests missed it because they construct `RunRecord` directly
with the field already populated — the one place a unit test cannot catch a
propagation gap.

Fixed minimally in `prepare_run()`; verified by relaunching the same suite
(row now reads `control_resumable_v1` / `CERTIFIED` / contract v1) and by two
regression tests that go through the real `WorkerManager` in both directions.

## 4. F-2 — Stop is not wired into the worker (blocker)

```
$ grep -rn "StopCoordinator\|settle_graceful_stop" bench/
  → only bench/control/checkpoints/stop.py and lifecycle.py (definitions)
```

Nothing constructs a `StopCoordinator`, nothing supplies
`contract["stop_requested"]`, and nothing calls `settle_graceful_stop` from a
worker. `run_suite._call_resumable_train` *reads*
`contract.get("stop_requested")` and passes it to the trainer, so the seam
exists — but the worker never fills it in.

Consequences, all verified by real launch:

- A control run completes normally and writes **no checkpoint** (confirmed:
  `checkpoints: []` after a completed `control_resumable_v1` run).
- A stop action recorded in the registry would never be observed by the worker.
- No interrupt Checkpoint v2 is ever produced by a real worker.
- Therefore the resume child has nothing to resume from, and the parity gate
  cannot run.

### What closing F-2 requires

The interrupt checkpoint must be written where the adapter is in scope, which
is inside `_call_resumable_train`, not in `worker_cli`:

1. `worker_cli` constructs a `StopCoordinator` for the run and puts it in the
   execution contract for `control_resumable_v1` runs.
2. `_call_resumable_train` already forwards it to `resumable_train`; on
   `result.interrupted` it must call `settle_graceful_stop` with the live
   adapter, a `CheckpointService` bound to the run directory, and
   `training_path_id="control_resumable_v1"` so the package is Checkpoint v2
   and launch-eligible.
3. The worker must then exit with code 10 and let the `INTERRUPTED` terminal
   state stand.

This is bounded and uses only existing pieces — no new schema, no migration 4,
no redesign. It was simply never connected.

## 5. What was *not* executed, and why

| Required gate | Status |
|---|---|
| KNet real-worker continuous vs stop/resume bitwise parity | **blocked by F-2** |
| Split real-worker parity + `hn1_init`/`hn2_init` restore | **blocked by F-2** |
| Two dependent child-failure fault cases | **blocked by F-2** |
| Idempotency / restart under real process launch | **blocked by F-2** (needs a real interrupt checkpoint to resume from) |

No substitute was accepted: the prompt explicitly forbids certifying via
`SyntheticExecutor`, direct adapter calls, in-process coordinator-only calls,
or mocked `Popen`, and none of those were used to claim a pass.

The existing service-level evidence still stands and still passes — adapter
fresh-process exact-resume parity for both models, checkpoint atomicity and
mutation probes, graceful stop at service level, coordinator idempotency and
crash recovery. That is a different certification level from real-worker
parity, and this report does not conflate them.

## 6. Regression

| Gate | Baseline | After F-1 fix |
|---|---|---|
| `pytest --collect-only -q` | 532 | 534, 0 errors |
| `pytest -q` | 531 passed, 1 skipped | **533 passed, 1 skipped, 0 failed** |
| 28 init-provenance | pass | pass |

+2 regression tests. No test deleted, skipped, xfailed or ignored.

## 7. Non-Regression Invariants

| Invariant | Result |
|---|---|
| API methods | **`GET` only** |
| Dash action buttons | **0** |
| Third-party tracked diff | **empty** |
| Authoritative dirty tree | untouched |

## 8. Certification Levels

| Level | Status |
|---|---|
| Adapter/service fresh-process exact-resume parity | **certified** (checkpoint tranche, still passing) |
| WorkerManager real-subprocess parity | **not certified** — blocked by F-2 |
| Public API/UI exposure | not implemented, deliberately |

`worker_level_exact_resume_certification.md` is deliberately **not** written:
the prompt says not to record worker-level certification as complete before
real-worker parity passes, and it has not.

## 9. Final Gate

**NOT_READY.**

Specific blocker: wire `StopCoordinator` and `settle_graceful_stop` into the
production worker for `control_resumable_v1` runs (§4), then run the KNet and
Split real-worker parity E2E, the two dependent fault cases, and the real
process idempotency/restart checks.

Everything else the gate lists is in place and verified. The remaining work is
one connection, not a redesign.

## 10. Evidence

`artifacts/benchmark_write_control_real_worker/<timestamp>/` (not committed):
`preflight/` (git ancestry, environment, schema versions), `pytest/` (baseline
and post-fix logs). Real-launch evidence for F-1 and F-2 was produced under
temporary control roots and is summarised above.
