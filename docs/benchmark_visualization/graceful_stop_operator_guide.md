# Graceful Stop — Operator Guide

## 1. What a graceful stop is

A request to finish the current optimizer update, write a verified checkpoint,
and exit cleanly so the run can be continued later as a child run.

It is **not** a kill. Killing a worker (`SIGKILL`, machine loss) still produces
`ORPHANED` with no checkpoint, exactly as before — the two are deliberately
distinguishable (DND-CSR-006).

## 2. Requesting one

```bash
PY=~/.pyenv/versions/3.10.13/bin/python

$PY -m bench.control.cli stop --run-id <run_id>

# Retry-safe: reusing a key makes the retry the same logical request.
$PY -m bench.control.cli stop --run-id <run_id> --idempotency-key my-key-1
```

The command **records a request and returns immediately**. It does not wait,
and it does not signal anything.

That is the design. The request is a row in the registry, so:

- it survives the API being down (the API is not involved at all);
- it survives your shell exiting;
- repeating it does not produce a second interrupt checkpoint.

The worker polls for its own outstanding action and honours it at the next
checkpoint-safe boundary.

## 3. What happens next

```
RUNNING → STOP_REQUESTED → CHECKPOINTING → INTERRUPTED     exit code 10
                                    └────→ FAILED          exit code 50
```

`INTERRUPTED` is recorded **only after** the interrupt checkpoint is written
*and* validated. If the checkpoint cannot be written — disk full, permissions —
the run becomes `FAILED` with exit code **50**, because a terminal state that
implies resumable state exists when it does not is worse than an honest
failure.

| Exit code | Meaning |
|---|---|
| 0 | Completed normally |
| 10 | Gracefully interrupted; a validated interrupt checkpoint exists |
| 40 | Ordinary run failure |
| 50 | Stop requested, checkpoint could not be persisted |

## 4. Watching it

```bash
$PY -m bench.control.cli show <run_id>
$PY -m bench.control.cli checkpoints list --run-id <run_id>

curl -s .../api/v1/runs/<run_id>/actions   | jq   # request history
curl -s .../api/v1/runs/<run_id>/checkpoints | jq
```

The dashboard shows the state and the checkpoint list. It has **no Stop
button** — stopping is a CLI operation in this build, and a button whose
backend is not exposed would be a promise this build does not make.

## 5. Verifying a checkpoint before relying on it

```bash
$PY -m bench.control.cli checkpoints validate --checkpoint-id <id>
```

Re-hashes the payload and checks it against the manifest and the registry row.
Exit code 0 means valid. This also updates `validation_status`.

If a machine died mid-write:

```bash
$PY -m bench.control.cli checkpoints reconcile --run-id <run_id>
```

- complete package with no catalog row → digest verified, then adopted
- catalog row whose package is missing or corrupt → marked `INVALID`
- `*.tmp-write` leftovers → **reported, not deleted** (may belong to a live
  writer). Remove them yourself once you are sure no worker is running.

## 6. Resuming

```bash
$PY -m bench.control.cli resume --checkpoint-id <id>
```

This validates the checkpoint and prints the cursor, identity, and the lineage
a child run would carry. Executing the child run is intentionally not wired to
a public write API in this tranche.

Resume is only offered from a run that is no longer executing
(`INTERRUPTED`, `COMPLETED`, `FAILED`, `CANCELLED`, `ORPHANED`). Resuming from
a `RUNNING` run is refused.

**A resume is a child run.** It gets its own `run_id` and its own directory.
The parent's directory, events, and checkpoints are never modified. The child
inherits the parent's `variant_id`, because a resume is execution lineage, not
a new model variant — do not treat parent and child as two independent runs
when aggregating results.

## 7. Certified envelope

Exact resume is certified for `kalmannet_tsp` and `split_knet` on
**CPU / fp32 / num_workers=0**, at the completed-optimizer-update boundary,
and only for runs that used the checkpointable training path.

GPU, AMP, multi-worker loaders, and the other adapters are **not** certified.
See `exact_resume_certification_matrix.md`, or:

```bash
curl -s .../api/v1/capabilities/exact-resume | jq
```

## 8. Things that will not work, by design

| Attempt | Result |
|---|---|
| Stop via a dashboard button | No such button exists |
| `POST /api/v1/runs/<id>/stop` | 405 — no write routes |
| Resume with a different learning rate | Refused: that is a forked continuation |
| Resume into a different dataset | Refused on `dataset_fingerprint` |
| Resume a corrupted checkpoint | Refused on digest |
| Resume `adaptive_knet` / `maml_knet` | Uncertified |
| Force-terminate a run | Not implemented; use OS signals, and expect `ORPHANED` |
