# Durable Resume-Action Contract

Gate C. `RESUME_EXACT` reuses the existing registry `run_actions` table; there
is no second action store.

## 1. States

| State | Meaning |
|---|---|
| `REQUESTED` | durable row exists |
| `ACKNOWLEDGED` | coordinator has ownership |
| `COMPLETED` | **exactly one child worker was launched** |
| `FAILED` | validation, allocation, or launch failed |

`COMPLETED` is about the *launch*, never about the child finishing training
(ADR-WC-016). A child that fails later is a child run state and does not
reopen a launch action that genuinely succeeded — asserted by test.

Stop-action semantics are unchanged: `RUNNING → STOP_REQUESTED →
CHECKPOINTING → INTERRUPTED`, `COMPLETED` only after a valid interrupt
checkpoint.

## 2. Idempotency

```
same key + same checkpoint/parent  -> same action, same child, same worker
same key + different payload       -> conflict, no side effect
```

Five identical requests yield one action row, one child run, one child
directory, one worker launch. Asserted by test.

## 3. Optimistic concurrency

`expected_parent_state_version` is checked **before** anything durable is
written. A stale request conflicts with no action row and no child.

## 4. Crash windows

Every step is durable before the next irreversible one, so each window is
recoverable:

| Crash point | Recovery |
|---|---|
| After action row | `reconcile-actions` completes it, once |
| After validation | same |
| After child allocation | the linked child is adopted, never a second one |
| After action→child link | same |
| After launch, before completion | existing worker row adopted, no relaunch |
| Coordinator restart | open actions settled or explicitly failed |
| Request retry | idempotency key resolves to the same action |

A second `reconcile-actions` is a no-op. Asserted by test.

The coordinator is a plain service: it does not depend on an HTTP process, so a
launched worker keeps running whether or not anything else is alive.

## 5. Worker identity

Adoption uses the registry worker row and `WorkerManager`'s existing
PID/start-time/token defences. A PID alone is never enough to reuse or signal a
worker.
