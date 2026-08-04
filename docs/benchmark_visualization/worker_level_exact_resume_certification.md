# Worker-Level Exact-Resume Certification

Certifies exact resume through **real `WorkerManager` OS processes**, the level
above the adapter/service fresh-process parity certified in the checkpoint
tranche.

Evidence commit: `ba65524` (wiring) / `7296294` (tests).
Feature worktree: `/tmp/bench-wc-tranche`, branch `benchmark-viz/write-control`.

## 1. Three distinct levels — do not conflate

| Level | Status |
|---|---|
| Adapter/service fresh-process parity | certified (checkpoint tranche) |
| **WorkerManager real-subprocess parity** | **certified here** |
| Public API/UI exposure | not implemented |

## 2. Certified envelope

```
device_class          cpu
precision             fp32
num_workers           0
resume_boundary       optimizer_update
training_path_id      control_resumable_v1
checkpoint_schema     v2
```

| model_id | implementation_id | result |
|---|---|---|
| `kalmannet_tsp` | `bench_kalmannet_tsp_adapter_v1` | **BITWISE PASS** |
| `split_knet` | `bench_split_adapter_v1` | **BITWISE PASS** |

Note the Split implementation id: `bench_split_adapter_v1` is what the system
actually derives. The earlier certification row named
`bench_split_knet_adapter_v1`, which the system never produces, so it was
unreachable and Split silently ran the legacy path. A guard test now pins
certified ids to the derived ones.

## 3. What was compared

Continuous reference run vs `parent → stop → interrupt Checkpoint v2 →
resumed child`, all launched by the real `WorkerManager`:

- final model `state_dict` — **sha256 identical** for both models
- update count reaches the same N
- lineage: `parent_run_id`, `resumed_from_run_id`, `resumed_from_checkpoint_id`
- inherited `variant_id`, `training_path_id`, `structural_config_hash`
- parent immutable: state, `state_version`, exit code, run-directory file list

Allow-listed as operational and excluded: run ids, PIDs, directories,
timestamps, telemetry samples.

The stop cursor is the **actual** one observed from the interrupt checkpoint
(KNet K=21, Split K=1), never hard-coded.

## 4. Process identity

Parent and child are separate OS processes with distinct PIDs, both launched
through `WorkerManager` via `bench.control.process.worker_cli`, running the
production executor/runner path against the real pinned third-party KNet and
Split modules. No `SyntheticExecutor`, direct adapter call, in-process
coordinator call, or mocked `Popen` was used.

## 5. Scope limits

Certifies the current **single-optimizer** Split implementation — this is not a
paper-fidelity claim. Not certified: GPU, AMP, `num_workers != 0`, distributed,
gradient accumulation, `adaptive_knet`, `maml_knet`, `me_split_knet_v0`, the
legacy training path, and any checkpoint schema below v2.

## 6. Real-process fault and restart — now certified

All four scenarios pass on real `WorkerManager` processes: checkpoint-write
failure (`FAILED`/exit 50, no false checkpoint), child restore failure before
the first update (`FAILED`/exit 40, 0 updates), child ordinary failure after
updates (`FAILED`/exit 40, lineage and parent checkpoint preserved), and
coordinator restart while a child is RUNNING (same child/worker/PID recovered,
second reconcile a no-op). See `write_control_fault_restart_report.md`.

**Correction.** The parity runs recorded here originally reported the resumed
child as `COMPLETED`. Those runs were passing on a swallowed runner failure
(F-3) plus a missing budget-accounting update (F-4), both fixed in `cf09078`.
The bitwise parity claim is unaffected and has been re-verified before and
after the fixes; the terminal state reported at the time was wrong.

## 7. Outstanding at this level

Nothing at this level. The items previously listed here — real-process
checkpoint-write failure, both child-failure cases, and coordinator restart
during a RUNNING child — are now certified (§6).
