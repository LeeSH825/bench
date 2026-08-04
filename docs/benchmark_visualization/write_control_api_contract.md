# Write-Control API Contract

## 1. Write mode is explicit and local-only

```bash
export BENCH_CONTROL_ENABLE_WRITES=1     # only "1" or "true"
```

Unset/0/false — and **anything unrecognised** ("yes", "on", "2") — leaves the
build read-only. A write surface must fail closed on an ambiguous value.

The routes are *registered* only in write mode, so by default a POST to a write
path is a **routing miss**, not a permission decision, and nothing about them
appears in the OpenAPI schema.

Write mode additionally requires a loopback bind, checked at **startup**:

```
Write control requires a loopback bind because authentication is not implemented.
```

`BENCH_CONTROL_ALLOW_PUBLIC_BIND` does **not** unlock public writes. It exists
for read-only exposure behind a trusted proxy; reusing it here would silently
widen its meaning.

## 2. Endpoints

```
POST /api/v1/runs/{run_id}/actions/stop
POST /api/v1/checkpoints/{checkpoint_id}/actions/resume
POST /api/v1/runs/launch                     # config-GUI tranche
GET  /api/v1/actions/{action_id}
```

`POST /api/v1/runs/launch` follows every rule on this page; its own semantics
(preset handles, preview-hash agreement, allocation ordering) are in
`launch_api_contract.md`.

One read-only exception was added deliberately: `POST /api/v1/config/validate`
is registered in **all** builds. It resolves and previews a config and has no
side effect of any kind. The read-only invariant is therefore "no *mutating*
routes are registered", not "no POST verb exists" — stated that way in
`tests/test_control_write_api.py` so the distinction cannot be lost.

Required headers on POST:

```
Idempotency-Key: <client key>
X-Bench-Control-Request: 1
Content-Type: application/json
```

The custom header is a CSRF speed bump, not a security control: a plain HTML
form or naive cross-site fetch cannot set it without a preflight this service
never grants. There is no wildcard CORS.

Bodies:

```json
{"expected_state_version": 12}          // stop
{"expected_parent_state_version": 19}   // resume
```

No learning-rate, budget or config override is accepted.

## 3. Responses

| Situation | Status |
|---|---|
| New action, or same action still in progress | **202** |
| Same idempotent action already finished | **200** |
| Malformed body / missing header or key | 400 |
| Unknown run / checkpoint / action | 404 |
| Stale state version, invalid state, key reused, corrupt checkpoint, no live worker | 409 |
| Uncertified envelope or unsupported training path | 422 |
| Registry / manager unavailable | 503 |

Every error carries a machine-readable `reason_code`.

Action resource:

```json
{
  "schema_version": 1, "action_id": "...", "action_type": "STOP_GRACEFUL",
  "state": "REQUESTED", "terminal": false, "run_id": "...",
  "checkpoint_id": null, "child_run_id": null, "result_checkpoint_id": null,
  "requested_at": "...", "acknowledged_at": null, "completed_at": null,
  "error": null, "idempotency_reused": false,
  "status_url": "/api/v1/actions/<id>"
}
```

The idempotency key is **never** echoed back.

## 4. Action completion is not training completion

```
RESUME_EXACT action COMPLETED  = exactly one child worker launched
child run COMPLETED            = the resumed training finished
```

A child that later fails never reopens a launch action that genuinely
succeeded.

## 5. Handlers are thin

A handler validates the request, delegates to the already-certified durable
action service, and returns the action. It never starts training, restores a
checkpoint, spawns a process, or mutates SQLite directly — so a request can
never block on a worker.

## 6. Eligibility read model

`GET /api/v1/runs/{run_id}` carries `action_eligibility`, the single source of
truth the UI renders:

```json
{
  "write_control_enabled": true,
  "stop_action": {"eligible": false, "reason_code": "TRAINING_PATH_NOT_RESUMABLE",
                  "reason": "Safe stop unavailable: this run used legacy_train_v1. …"},
  "resume_action": {"eligible": false, "reason_code": "CHECKPOINT_TRAINING_PATH_UNPROVEN",
                    "reason": "Exact resume unavailable: …", "checkpoint_id": null}
}
```

The UI must not recompute these conditions, and the POST path re-checks all of
them anyway — including worker liveness — so UI gating and enforcement cannot
drift (ADR-WC-020).

## 7. Auditing the surface

Audit via the **OpenAPI schema**, not by walking `app.routes`: this FastAPI
version stores included routers as opaque `_IncludedRouter` objects, so
route-walking reports no POST methods even when they exist.

```python
spec = create_app().openapi()
sorted({m.upper() for p in spec["paths"].values() for m in p})
```
