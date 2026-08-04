# Write-Control Operator Guide

## 1. Enabling write control

```bash
export BENCH_CONTROL_ENABLE_WRITES=1
python -m bench.control.api.app --host 127.0.0.1 --port 8765
python -m bench.ui.dash_app --host 127.0.0.1 --port 8766 --api http://127.0.0.1:8765
```

Both must be loopback. A write-enabled server on a non-loopback address
**refuses to start** — there is no authentication, and
`BENCH_CONTROL_ALLOW_PUBLIC_BIND` does not change that.

Without the variable everything stays read-only: no write routes, no buttons.

## 2. Stop safely

Run Detail → **Stop safely**.

The run finishes its current optimizer update, writes a verified interrupt
checkpoint, and exits. It may not stop immediately.

| Outcome | Meaning |
|---|---|
| `INTERRUPTED`, exit 10 | Checkpoint written and validated; resumable |
| `FAILED`, exit 50 | Checkpoint could not be written; **not** resumable |

This is not a kill. There is no force-terminate button; an external `SIGKILL`
still produces `ORPHANED`.

## 3. Resume training

Run Detail → **Resume training**, on an `INTERRUPTED` run with a valid
Checkpoint v2.

Creates a **new child run** with its own id and directory. The parent, its
events and its checkpoints are never modified. The child inherits the parent's
variant, structural hash and training path.

Read the two states separately:

- the **action** completes when the child worker has launched;
- the **child run** state tells you whether its training finished.

## 4. Retries

Clicking twice, refreshing, or losing the API does **not** create a second
action, child or worker — the UI reuses a stable idempotency key. If the API
is unreachable, retry; do not construct a new request.

## 5. When a button is missing

The reason is shown next to it. Common ones:

| Reason | Meaning |
|---|---|
| `TRAINING_PATH_NOT_RESUMABLE` | Run used `legacy_train_v1`; no safe boundary |
| `RUN_NOT_RUNNING` | Only a RUNNING run can be stopped |
| `NO_LIVE_WORKER` | Worker is gone; reconcile instead |
| `CHECKPOINT_TRAINING_PATH_UNPROVEN` | Checkpoint v1 predates the contract |
| `CHECKPOINT_SCHEMA_NOT_WRITE_CONTROL_CERTIFIED` | v2 required for child launch |
| `UNCERTIFIED_TUPLE` | Outside CPU/fp32/0-workers certification |
| `PARENT_NOT_TERMINAL` | Parent still executing |

## 6. Certified envelope

Stop/Resume are offered only for `kalmannet_tsp` and `split_knet` on
CPU / fp32 / `num_workers=0` / `control_resumable_v1` / Checkpoint v2. GPU,
AMP, multi-worker, and the other adapters are not certified and are refused by
the backend, not merely hidden in the UI.

## 7. Launching from the UI

Launching a run from the browser now exists, as a separate page and a separate
contract: see `config_gui_operator_guide.md` and `launch_api_contract.md`. A
run started there is an ordinary control-plane run and gets the Stop/Resume
controls described above when its training path is `control_resumable_v1`.

## 8. Not available

Force terminate, warm start, evaluate-checkpoint, sweeps, batch launch, GPU
scheduling, authentication and remote workers are all absent by design.
