# Launch API Contract

`POST /api/v1/runs/launch` — the only surface that can create work from a
browser. Read `write_control_api_contract.md` first; everything there (write
mode, loopback bind, `X-Bench-Control-Request`, `Idempotency-Key`, no wildcard
CORS) applies unchanged.

## 1. Registration

```
POST /api/v1/runs/launch          write mode only
POST /api/v1/config/validate      always (read-only preview, no side effects)
GET  /api/v1/config/presets       always
GET  /api/v1/config/presets/{id}  always
GET  /api/v1/config/schema        always
```

In the default build the launch path is not registered. A POST to it returns
**405**, not 403 — the read-only build still serves `GET /api/v1/runs/{run_id}`,
which the path matches. Either way no handler runs and nothing is allocated.

## 2. Request

```json
{
  "preset_id": "suite_train_smoke.1493c274c28b",
  "preset_digest": "sha256:…",
  "task_id": null,
  "model_id": "kalmannet_tsp",
  "init_id": "trained",
  "overrides": {"training.max_updates": 600},
  "expected_structural_config_hash": "sha256:…",
  "expected_operational_config_hash": "sha256:…"
}
```

The body carries **data and handles**, never a filesystem path, class name,
module, or command. `overrides` accepts only descriptor-declared field paths.

## 3. Ordering, and why it is this way

```
1. resolve preset  → digest must still match      (else 409)
2. validate        → must be valid and eligible   (else 422)
3. compare hashes  → must match the preview       (else 409)
4. record LAUNCH_RUN action        (durable, keyed by Idempotency-Key)
5. ACKNOWLEDGED
6. allocate the immutable run, link it to the action, write provenance
7. WorkerManager.launch
8. COMPLETED
```

The action row is written **before** the run exists. That required migration
0004 to make `run_actions.run_id` nullable: an action that *creates* a run
cannot name it in advance, and allocating first would leave an unattributable
run behind if the process died between the two writes. The run it created is
`result_child_run_id`.

`GET /api/v1/actions/{id}` reports that child as `run_id` for a `LAUNCH_RUN`
action. Returning the raw NULL made the UI lose the link to the run it had just
started as soon as the POST response was replaced by a poll — fixed, and pinned
by `test_polling_a_launch_action_still_names_its_run`.

## 4. Status codes

| Code | Meaning |
|---|---|
| 202 | accepted, action recorded (first time for this key) |
| 200 | same key replayed, action already terminal |
| 400 | missing `X-Bench-Control-Request` or `Idempotency-Key` |
| 404 | unknown preset |
| 409 | preset digest drift, preview-hash mismatch, or key reused with a different request |
| 422 | invalid config, or a model that is not GUI-launchable |
| 503 | the coordinator could not be reached at all |

Every refusal allocates **nothing** — no run row, no action row, no run
directory, no worker. Asserted for each case individually rather than once.

## 5. Idempotency

The `Idempotency-Key` decides identity of the *request*, not of the config:

- same key, same request, five times → `[202, 200, 200, 200, 200]`, **one**
  action, **one** run, **one** worker;
- same key, different draft → 409. A replay with different content is a client
  bug, not a retry;
- same config, different keys → distinct immutable runs. Launching the same
  experiment twice on purpose is legitimate and must not be deduplicated.

The key is a client credential and is **not** stored in run provenance.
`test_launch_writes_provenance_without_the_idempotency_key` scans every file in
the run directory for it.

## 6. Restart boundaries

`LaunchCoordinator.settle` is re-runnable and `reconcile_open_actions` adopts
open actions after a restart:

| Crash point | Recovery |
|---|---|
| after the action row, before allocation | reconcile allocates once; a second reconcile is a no-op |
| after allocation, before launch | settle adopts the existing run, launches once |
| after launch | the worker is adopted by `worker_for_run`, not respawned |

## 7. Failure semantics

Action completion is **not** workload completion, and the two are never merged:

| Failure | Action | Run |
|---|---|---|
| validation | never created | never created |
| allocation | FAILED | none |
| worker spawn | FAILED | CANCELLED, `exit_code 52`, `terminal_reason launch_failed` |
| training | COMPLETED | FAILED with the worker's own exit code |

A never-started run is CANCELLED rather than FAILED: it never ran, and calling
it a failure would corrupt failure statistics.

## 8. Provenance written at allocation

Inside the run directory, alongside `resolved_run_spec.json`:

```
original_preset.yaml     the tracked file, byte-for-byte
submitted_draft.yaml     the canonical document that was validated
config_validation.json   the full validation result, including the diff
launch_request.json      launch_source, preset id/digest/path, overrides,
                         hashes, variant_id, training_path_id, action_id
```

This is what makes a GUI-launched run auditable: the file it started from, what
the operator submitted, what it resolved to, and which action created it.

## 9. Security

- loopback bind enforced at startup; write mode fails closed on an unrecognised
  env value;
- no shell invocation anywhere on this path;
- request bodies bounded (512 KiB of YAML) before parsing;
- error bodies carry a `reason_code` and a message — no stack traces, no
  absolute paths;
- no pickle load in any handler;
- presets restricted to the tracked allowlist (`config_gui_contract.md` §1).
