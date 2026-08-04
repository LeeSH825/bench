# Production Worker Graceful-Stop Contract

```yaml
git_common_repository: /home/dss-pc-05/bench/.git
user_working_tree: /home/dss-pc-05/bench          # untouched
feature_worktree: /tmp/bench-wc-tranche
branch: benchmark-viz/write-control
head_commit: 729629437e361d5a177cb81177607b03e27de6f8
verification_layer: production_worker
```

## 1. What this closes

`StopCoordinator` and `settle_graceful_stop()` existed and were covered at
service level, but nothing constructed them in the real worker. A production
run could not observe a stop action and never wrote an interrupt checkpoint.

## 2. Wiring

```
worker_cli
  └─ training_path_id == control_resumable_v1 ?
       ├─ StopCoordinator(run_id, registry)      -> contract["stop_requested"]
       └─ _build_interrupt_settlement(...)        -> contract["on_interrupt"]

run_suite._call_resumable_train
  └─ result.interrupted -> contract["on_interrupt"](adapter, cursor, progress, plan)
       └─ settle_graceful_stop(..., training_path_id="control_resumable_v1")
```

**Gating.** `legacy_train_v1` and `not_applicable` receive no callback at all.
Their loops have no safe boundary at which a stop could be honoured, so
offering one would be a promise the path cannot keep.

**Why settlement lives in the runner.** The interrupt checkpoint needs the live
adapter, and the adapter only exists inside `_call_resumable_train`. The worker
supplies a closure over the registry, event writer and checkpoint service, so
the runner itself stays free of control-plane wiring.

**No callback is a hard error.** If an interrupt arrives and the contract has no
settlement callback, the runner raises. Continuing would let an interrupted run
report `COMPLETED` with no checkpoint — the false terminal state DND-CSR-004
forbids.

## 3. Terminal ordering

```
RUNNING → STOP_REQUESTED → CHECKPOINTING
        → interrupt Checkpoint v2 published and validated
        → action COMPLETED → INTERRUPTED → worker exit 10
```

Checkpoint write failure:

```
CHECKPOINTING → FAILED → worker exit 50, no valid checkpoint row,
                         not launch-eligible
```

## 4. Terminal-overwrite protection

The worker skips its `COMPLETED` handler when the run is already in a
runner-settled state (`INTERRUPTED` or `FAILED`) and returns the matching exit
code. Without this the graceful-stop terminal would be overwritten and an
interrupted run would be reported as finished.

## 5. Signal discipline unchanged

Signal handlers still only set a flag. No `torch.save`, no SQLite write, no
checkpoint work happens in a handler (ADR-CSR-010). The durable path is the
registry action, which is why a stop survives the requester exiting.

## 6. Verified on real processes

| Check | KNet | Split |
|---|---|---|
| Transitions | RUNNING→STOP_REQUESTED→CHECKPOINTING→INTERRUPTED | same |
| Exit code | 10 | 10 |
| Interrupt package | Checkpoint **v2**, `VALID`, `control_resumable_v1` | same |
| Stop cursor | actual (21), not hard-coded | actual (1) |
| 5 identical stop requests | 1 action, 1 checkpoint, 1 terminal | same |

## 7. Historical clarification

> The checkpoint tranche certified graceful-stop behaviour at the **service**
> layer. Production `WorkerManager` integration was not certified until commit
> `ba65524`. Earlier phrasing such as "the worker polls for its own outstanding
> action" described service-level behaviour and was not true of the production
> worker before that commit.
