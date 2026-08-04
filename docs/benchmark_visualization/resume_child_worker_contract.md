# Resume Child-Worker Contract

Gate B1. Turns `plan_resume()` (validation only) into a real child run started
by `WorkerManager`.

## 1. Flow

```
checkpoint_id
→ manifest-only validation (digest, schema, inventory) — no unpickling
→ registry parent/checkpoint lookup
→ eligibility: certification tuple + training path + parent terminal
→ optimistic concurrency on parent state_version
→ durable RESUME_EXACT action
→ immutable child allocation (new run_id, new directory)
→ child ResolvedRunSpec persisted
→ lineage recorded on the child
→ action → child link  (before launch)
→ WorkerManager.launch(child)
→ fresh child process
→ checkpoint restore
→ resumable_train() for the remaining updates
→ child terminal state
```

## 2. Child invariants

| Inherited from parent | New on child |
|---|---|
| `variant_id` | `run_id` |
| `training_path_id` (`control_resumable_v1`) | run directory |
| structural config hash | `parent_run_id` |
| implementation identity | `resumed_from_run_id` |
| dataset fingerprint | `resumed_from_checkpoint_id` |

The child uses the **ordinary** lifecycle — `CREATED → VALIDATING → QUEUED →
STARTING → RUNNING → terminal`. The parent is never moved to `RESUMING` or
back to `RUNNING`; resume is expressed as lineage plus a `RESUME_RESTORE` start
event (ADR-WC-008).

## 3. Parent immutability

Asserted by test down to: state, `state_version`, exit code, transition count,
checkpoint rows, and the checkpoint directory file list.

## 4. Restore ordering

The child worker reads its own lineage from the registry, resolves the parent's
run directory, and hands the checkpoint id to the runner. The runner restores
model, optimizer, RNG, batch-plan cursor, best state and adapter extra state
(including Split's `hn1_init`/`hn2_init`) **before** the first new optimizer
update.

There is no fallback: if restore fails, the run fails. A child that silently
continued on the legacy loop would be labelled a resumed exact continuation
while being nothing of the sort (ADR-WC-003).

## 5. Launch failure

| Situation | Result |
|---|---|
| Launch raises | action `FAILED`, reason recorded |
| Child allocated but never started | child `CANCELLED` (nothing ran) |
| Child had entered the running path | child `FAILED` |
| Retry with the same key | same child adopted, never a second one |

`CANCELLED` rather than `FAILED` for a never-started child is deliberate:
nothing executed, so calling it a failure would misreport what happened, and
`CREATED → FAILED` is not a legal transition.

## 6. CLI

```bash
# plan only — unchanged, no side effect
python -m bench.control.cli resume --checkpoint-id <id>

# actually allocate a child and launch its worker
python -m bench.control.cli resume --checkpoint-id <id> --launch \
    --idempotency-key <key> [--expected-parent-state-version <n>]

# restart recovery
python -m bench.control.cli reconcile-actions
```

`--idempotency-key` is **required** with `--launch`, so a retried invocation
cannot create a second child.
