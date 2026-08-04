# Write API / Dash Controls Tranche Report

```yaml
git_common_repository: /home/dss-pc-05/bench/.git
user_working_tree: /home/dss-pc-05/bench          # untouched
feature_worktree: /tmp/bench-wc-tranche
branch: benchmark-viz/write-control
baseline_commit: 2133709
head_commit: ca05ece978cd1f3e96fc212ff0afcbc2757d3e69
control_root: temporary per-scenario roots under /tmp
verification_layer: api_ui_browser
```

## 1. Executive Verdict

**READY_FOR_CONFIG_GUI_LAUNCH_TRANCHE.**

Default read-only is preserved, write mode is explicit and loopback-only, the
POST action API is idempotent and restart-safe, both models complete a real
API→worker Stop/Resume cycle with bitwise parity intact, and the full browser
workflow passes.

## 2. Read-only default

| Check | Result |
|---|---|
| OpenAPI methods, writes disabled | **GET only** |
| Write paths in schema | none |
| POST to a write path | 404/405 (route not registered) |
| Capability flags | all write flags **false** |
| Dash action buttons | **0** (verified in a real browser) |

Write routes are *registered* only in write mode, so by default a POST is a
routing miss rather than a permission decision.

## 3. Write-mode security

| Check | Result |
|---|---|
| `"1"`/`"true"` enable | yes |
| `"yes"`, `"on"`, `"2"`, unrecognised | **false** (fails closed) |
| Loopback (`127.0.0.1`, `localhost`, `::1`) | allowed |
| `0.0.0.0`/`::`/LAN with writes | **startup refused** |
| `ALLOW_PUBLIC_BIND=1` + writes | **still refused** |
| Missing `X-Bench-Control-Request` | 400 |
| Missing `Idempotency-Key` | 400 |

## 4. POST Stop / Resume

Stop: 202 on first request; unknown run 404; legacy path 422; non-RUNNING 409;
stale state version 409 **with no action row created**; no live worker 409.

Resume: unknown checkpoint 404; ineligible envelope 422; invalid/parent-not-
terminal 409. `GET /actions/{id}` round-trips and never echoes the key.

## 5. Idempotency and restart

| Case | Result |
|---|---|
| 5 identical POST stops | 1 action (202 then reused) |
| 5 identical POST resumes, **spanning an API restart** | 1 action, 1 child, 1 worker |
| Same key, different target | 409 |
| Double-click in the browser | 1 action |

## 6. KNet API→worker E2E

```
reference COMPLETED 40 updates
parent    INTERRUPTED exit 10, interrupt Checkpoint v2 at update 17
action    COMPLETED, terminal, result checkpoint present
child     COMPLETED exit 0, 40 updates, lineage correct
BITWISE PARITY: true
```

## 7. Split API→worker E2E

```
parent INTERRUPTED exit 10, Checkpoint v2
child  COMPLETED exit 0, 600 updates
BITWISE PARITY: true — 1 action, 1 child
```

## 8. Playwright browser workflow

| Step | Result |
|---|---|
| Read-only mode buttons | **0** |
| Stop safely visible in write mode | yes |
| Double-click → actions created | **1** |
| Action panel shows `STOP_GRACEFUL` | yes |
| Parent reaches `INTERRUPTED` exit 10 | yes |
| `INTERRUPTED` visible after refresh | yes |
| Resume training visible | yes |
| Child link rendered | yes |
| Child COMPLETED (400 updates) | yes |
| Child page shows its state | yes |
| Ineligible reason shown | yes |
| Force/Warm/Evaluate/Delete/Clone/Launch buttons | **none** |
| Console errors | none |

## 9. Two UI bugs found and fixed

**Action panel lifetime.** The status panel was inside the polled controls
block, so every poll re-rendered the block and destroyed the element the action
callback had just written into — the status and the child link both vanished
within a second. Moved outside the polled block.

**Navbar badge.** The header still read "read-only" with write control enabled.
Now reflects the actual mode.

## 10. Correction to earlier audits

Earlier tranches audited the HTTP surface by walking `app.routes`. This FastAPI
version stores included routers as opaque `_IncludedRouter` objects, so that
walk reports no POST methods **even when they exist**. Those audits reached the
right conclusion — no write routes existed at all — but by an unsound method.
This tranche audits the OpenAPI schema instead, and the tests do too. Earlier
reports are left unmodified; the correction lives here.

## 11. Regression

| Gate | Baseline | Now |
|---|---|---|
| `pytest --collect-only -q` | 547 | **582**, 0 errors |
| `pytest -q` | 546 passed, 1 skipped | **581 passed, 1 skipped, 0 failed** |
| 28 init-provenance | pass | **pass** |
| Third-party tracked diff | empty | **empty** |

+35 tests. Nothing deleted, skipped, xfailed or ignored. All E2E used temporary
control roots; no tracked `runs/` or `reports/` were written.

## 12. Layering

Dash callbacks call only `ApiClient`; none imports a registry, adapter, trainer
or `WorkerManager`. API handlers delegate to the certified action services and
never wait on a worker. The browser never calls the API directly.

## 13. Final gate

**READY_FOR_CONFIG_GUI_LAUNCH_TRANCHE.**

## 14. Evidence

`artifacts/benchmark_write_api_ui/<timestamp>/` — browser screenshots
(read-only, pre-stop, post-interrupt, post-resume, child), `result.json`, and
pytest logs. Not committed.
