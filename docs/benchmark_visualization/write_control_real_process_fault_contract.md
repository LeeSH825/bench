# Real-Process Fault Contract

```yaml
git_common_repository: /home/dss-pc-05/bench/.git
user_working_tree: /home/dss-pc-05/bench          # untouched
feature_worktree: /tmp/bench-wc-tranche
branch: benchmark-viz/write-control
baseline_commit: 0b197ae
head_commit: a869b280d92435cc09a518ac983f3643aa4b3dc6
control_root: temporary per-scenario roots under /tmp
verification_layer: real_process_fault_restart
```

## 1. Terminal-state contract

| Situation | Run state | Exit | Checkpoint | Action |
|---|---|---|---|---|
| Graceful stop, checkpoint published | `INTERRUPTED` | 10 | v2, `VALID`, launch-eligible | `COMPLETED` |
| Graceful stop, checkpoint write fails | `FAILED` | **50** | none valid | `FAILED` |
| Child restore/protocol failure before first update | `FAILED` | 40 | parent's unchanged | launch `COMPLETED` |
| Child ordinary failure after ≥1 update | `FAILED` | 40 | parent's unchanged | launch `COMPLETED` |
| Child allocated but never started | `CANCELLED` | — | — | `FAILED` |

**Action completion ≠ training completion.** A `RESUME_EXACT` action is
`COMPLETED` once exactly one child worker was launched. A child that later
fails is a child run state; it never reopens the launch action. Scenarios B and
C both show action `COMPLETED` with child `FAILED`, which is correct.

## 2. Failure is an allow-list, not a deny-list

`run_one()` reports internal failure by **returning** `status="failed"`, not by
raising. `SuiteExecutor` now accepts only `ok`/`success`/`completed` and raises
`ExecutionError` for anything else, so an unrecognised status fails closed.

Before this, any runner-level failure became a `COMPLETED` run with exit 0.

## 3. Budget accounting on the resumable path

The trained-plan policy check reads `adapter.train_updates_used`. Legacy
`train()` sets it; `_call_resumable_train` did not, so every resumed child
tripped `policy_violation: trained plan requires positive
train_outer_updates_used` at report time — invisible while §2 was swallowing
failures. The resumable path now sets the attribute and updates the ledger.

## 4. What was never wrong

The numerics. The resumed child's final model stayed **bitwise identical** to
the continuous reference before and after both fixes. Only the terminal state
and the accounting were wrong.

## 5. Checkpoint-failure and reconciliation

A checkpoint publication failure leaves no valid row and no `INTERRUPTED`
transition, so nothing is launch-eligible. This does not conflict with the
reconciler's adopt-orphan-package rule: adoption only applies to a **complete,
digest-verified** package that lost its registry row between publication and
the registry transaction. A publication that failed never produced one, and
partial writes carry the temp suffix and are never catalogued.

## 6. Restart contract

A coordinator restart re-runs `reconcile-actions`. A child already launched is
re-discovered by its registry worker row — same `run_id`, same
`worker_instance_id`, same PID — and is never relaunched. A second reconcile is
a no-op. Verified with a live child that progressed from update 28 to 128
across the restart.
