# Write-Control Real-Process Fault and Restart Report

```yaml
git_common_repository: /home/dss-pc-05/bench/.git
user_working_tree: /home/dss-pc-05/bench          # untouched, no source edits or tests run
feature_worktree: /tmp/bench-wc-tranche
branch: benchmark-viz/write-control
baseline_commit: 0b197ae
head_commit: a869b280d92435cc09a518ac983f3643aa4b3dc6
control_root: temporary per-scenario roots under /tmp
verification_layer: real_process_fault_restart
```

## 1. Executive Verdict

**READY_FOR_WRITE_API_UI_TRANCHE.**

All four required scenarios pass on real `WorkerManager` OS processes, both
parity regressions still pass bitwise, and the full suite is green.

Getting there required two production fixes, both false-success paths that no
service-level test could reach — and both of which had been quietly inflating
earlier results.

## 2. Scenario results

| Scenario | Result |
|---|---|
| **A** real worker checkpoint-write failure | **PASS** — `FAILED`, exit **50**, action `FAILED`, 0 checkpoint rows, no `INTERRUPTED` transition |
| **B** child restore failure before first update | **PASS** — child `FAILED` exit 40, **0 updates**, traceback present, 1 child, parent unchanged |
| **C** child ordinary failure after ≥1 update | **PASS** — advanced past resume cursor 17→120, `FAILED` exit **40**, parent checkpoint still `VALID`, lineage intact |
| **D** coordinator restart while child RUNNING | **PASS** — child PID alive, 28→128 updates across restart, same run_id/worker_instance/PID, 1 child, 1 action, second reconcile no-op, child reached `COMPLETED` |

Faults were injected as **real conditions** — an unwritable checkpoint
directory, a truncated payload, an unwritable output tree — not by mocking. No
`SyntheticExecutor`, direct adapter call, in-process service call or mocked
`Popen` was used.

## 3. Two production bugs found

### F-3 — SuiteExecutor swallowed runner failures

`run_one()` reports internal failure by **returning** `status="failed"` rather
than raising. The executor passed it through, so the worker recorded
`COMPLETED` / exit 0 for a run that had failed.

Found in Scenario B: the runner correctly raised
`CheckpointValidationError: payload digest mismatch` on the corrupted
checkpoint, and the child still reported success with 0 updates — the worst
possible outcome, a silent false success.

Fixed: success is an allow-list; anything else raises `ExecutionError` and
reaches the worker's ordinary failure path.

### F-4 — resumable path recorded no budget accounting

The trained-plan policy check reads `adapter.train_updates_used`. Legacy
`train()` sets it; `_call_resumable_train` did not, so **every resumed child**
tripped `policy_violation: trained plan requires positive
train_outer_updates_used` at report time.

This was invisible while F-3 was swallowing failures. Fixing F-3 exposed it
immediately: a KNet resumed child that had been reported `COMPLETED` now
reported `FAILED` — with its model still bitwise identical to the reference.

Fixed: the resumable path sets the attribute and updates the ledger.

### Correction to earlier reporting

> The tranche-19 report recorded resumed children as `COMPLETED`. Those runs
> were passing on a swallowed failure (F-3 + F-4). The **bitwise parity claim
> itself still holds** — verified again here, before and after both fixes —
> but the terminal state reported at the time was wrong. This report supersedes
> that detail; the earlier report is left unmodified as historical evidence.

## 4. Parity regression

| Model | Reference sha256 | Child sha256 | Result |
|---|---|---|---|
| `kalmannet_tsp` | `7c10a4145ffdb1a6…` | `7c10a4145ffdb1a6…` | **BITWISE PASS**, child `COMPLETED` exit 0 |
| `split_knet` | identical | identical | **BITWISE PASS**, child `COMPLETED` exit 0 |

Parent immutable, lineage complete, variant/path/structural hash inherited,
5 resume requests → 1 child in both cases.

Split's certified implementation id remains `bench_split_adapter_v1`, matching
the resolver; the earlier unreachable id is not reseeded (guard test in
`test_control_training_path_selection.py`).

## 5. Idempotency

| Request | Result |
|---|---|
| Stop ×5, same key | 1 action, 1 acknowledgement, 1 checkpoint, 1 terminal transition |
| Resume ×5, same key (incl. across restart) | 1 action, 1 child run, 1 worker |

## 6. Regression and invariants

| Gate | Baseline | Now |
|---|---|---|
| `pytest --collect-only -q` | 541 | **547**, 0 errors |
| `pytest -q` | 540 passed, 1 skipped | **546 passed, 1 skipped, 0 failed** |
| 28 init-provenance | pass | **pass** |
| API methods | GET | **GET only** |
| Dash action buttons | 0 | **0** |
| Third-party tracked diff | empty | **empty** |

+6 tests. Nothing deleted, skipped, xfailed or ignored. All scenarios used
temporary control roots; no tracked `runs/` or `reports/` were written.

## 7. Final gate

**READY_FOR_WRITE_API_UI_TRANCHE.** Every backend gate the write API and Dash
controls depend on is now closed at the real-process level: normal path, stop,
resume, both parity models, all four fault/restart scenarios, and idempotency.

This does not mean POST routes or Dash buttons exist — they deliberately do
not. It means the layer beneath them is certified.

## 8. Evidence

`artifacts/benchmark_write_control_fault_restart/<timestamp>/` (not committed):
preflight snapshot, pytest logs, and per-scenario JSON results. Scenario roots
were temporary `/tmp/w20-*` control roots.
