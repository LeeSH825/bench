# Write-Control Backend Continuation Report

## 1. Executive Verdict

**READY_AFTER_SPECIFIC_FIXES.**

Gates B0, B1 and C are implemented and tested. One mandated item was not
executed: the real-subprocess KNet/Split parent→stop→child→completion **bitwise
parity** E2E (§9.3, §9.4). The coordinator, eligibility gating, idempotency and
crash recovery are covered against a real registry, real checkpoint packages and
the real `WorkerManager` surface, but the end-to-end numerical parity of a child
launched as an actual OS process was not re-demonstrated in this continuation.

That is the specific fix required before `READY_FOR_WRITE_API_UI_TRANCHE`.
Everything the API/UI tranche depends on structurally — durable actions,
eligibility reason codes, immutable children, lineage — exists and is verified.

| Gate | Status |
|---|---|
| B0 training-path persistence + checkpoint v2 | **DONE** |
| B1 immutable child launched via WorkerManager | **DONE** (parity E2E outstanding) |
| C durable RESUME_EXACT + recovery | **DONE** |
| D CLI backend E2E | **PARTIAL** — CLI wired; full real-worker parity run not executed |

## 2. Source Branch, Commits, Worktree

| Item | Value |
|---|---|
| Authoritative repo (untouched) | `/home/dss-pc-05/bench`, `d92cd0c` |
| Continuation baseline | `dfb932d` on `benchmark-viz/write-control` |
| Partial commits verified | `48b0edb` ✔ present, ancestor · `dfb932d` ✔ present |
| `c0eaaf7` ancestry | confirmed |
| Worktree | `/tmp/bench-wc-tranche` (reused, safe) |
| New commits | `11a451d`, `de79d2a`, `a3ceff1`, plus this docs commit |

Nothing was re-implemented from the checkpoint branch; work continues on top of
`48b0edb`/`dfb932d`.

## 3. Partial Tranche Inheritance

Step A is preserved unchanged: `control_resumable_v1` / `legacy_train_v1` /
`not_applicable`, decided once in the resolver, no user toggle, no silent
fallback, old specs permanently legacy, explicit `BatchPlan` retained, and the
direct legacy-vs-resumable characterization still asserting bitwise equality on
an identical batch sequence.

## 4. Baseline Tests

Partial baseline reproduced before changes: **515 collected / 514 passed /
1 skipped**, 28 init-provenance passed. (One in-flight run raced an early edit;
re-verified clean afterwards.)

## 5. Training-Path Persistence

Registry **migration 3**, forward-only and additive:
`runs.training_path_id` / `training_path_reason_code` /
`training_path_contract_version`; `checkpoints.training_path_id` /
`checkpoint_schema_version`; `exact_resume_certifications.training_path_id`;
`run_actions.result_child_run_id` / `result_worker_instance_id`.

Defaults are `legacy_train_v1`. Verified by upgrading a synthetic v2 registry:
existing rows preserved with correct `state_version`, old runs default to
legacy, old checkpoints report schema 1 with no training path, automatic backup
taken. **No old row is ever promoted.**

## 6. Checkpoint Versioning Decision

Schema **v2** added; v1 explicitly not redefined. The version is *derived* from
provenance — a save with a training path is v2, a save without is v1 — which is
why every pre-existing v1 test passes untouched. Both versions readable;
unknown/future refused, never migrated. Full contract in
`checkpoint_v2_schema.md`.

The valid / restorable / launch-eligible distinction is implemented in
`eligibility.py` with machine-readable reason codes, decided from manifest and
registry only, never by unpickling the payload.

## 7. Certification Update

Certification key extended with `training_path_id` (and now carries schema
version 2). The same model on the legacy loop is a different tuple and is **not**
certified — asserted by test. Rows are seeded as reference data on registry
migration so an eligibility lookup never fails merely because a registry was
never seeded. v1-era evidence in the checkpoint tranche is retained.

## 8. Resume Child Architecture

See `resume_child_worker_contract.md`. Child gets a new run id and directory,
inherits variant, structural hash and training path, records full lineage, and
uses the ordinary lifecycle — the parent is never moved to `RESUMING`.
Parent immutability is asserted down to the checkpoint directory file list.

## 9. Durable Action and Recovery

See `durable_resume_action_contract.md`. Verified: five identical requests →
one action, one child, one worker; same key with a different checkpoint →
conflict; stale parent version → conflict with no side effect; launch failure →
action `FAILED` and child `CANCELLED` (never left live); crash after action row
and crash after child allocation both recover without a second child; a second
reconcile is a no-op; a completed action is not reopened by a child that later
fails.

## 10. CLI Execution Workflow

`resume` remains plan-only. `--launch` is explicit and requires
`--idempotency-key`. `reconcile-actions` added for restart recovery.

## 11–12. KNet / Split Full Worker Result

**Not executed.** This is the outstanding item. The existing checkpoint-tranche
parity evidence (in-process, through-package, and fresh-process resume, all
bitwise for both models) still passes as part of the suite, but it does not
exercise a child launched by `WorkerManager` as an OS process.

## 13. Fault Injection

Covered: corrupt checkpoint, v1 package launch attempt, legacy training path,
non-terminal parent, uncertified tuple, launch exception, stale version, crash
after action row, crash after child allocation, duplicate retry. Not covered:
child ordinary training exception and child SIGKILL in a real resumed child
(depends on §11–12).

## 14. Parent Immutability / Lineage

Passing. State, version, exit code, transition count, checkpoint rows and
directory contents all unchanged after a resume; child lineage complete.

## 15. Read-Only API / UI Non-Regression

API methods exposed: **`GET`** only. Dash action buttons: **0**. No POST route,
no write-mode switch, no UI control was added — as required.

## 16. Full Regression

| Gate | Baseline | Now |
|---|---|---|
| `pytest --collect-only -q` | 515 | **532**, 0 errors |
| `pytest -q` | 514 passed, 1 skipped | **531 passed, 1 skipped, 0 failed** |
| 28 init-provenance | pass | **pass** |

+17 tests. No test deleted, skipped, xfailed or ignored. Two existing tests were
updated because their premise changed by design (registry schema is now 3; a
save without provenance now correctly publishes v1). New tests write only to
`tmp_path`.

## 17. Third-Party Isolation

`git diff --submodule=log -- third_party` is **empty**. No tracked vendored file
modified; revisions unchanged.

## 18. Remaining API/UI Work

Deferred to the next tranche, as instructed: POST write routes, Dash Stop/Resume
controls, write-mode env switch behaviour, Playwright action workflow.

Outstanding in *this* layer: the real-worker KNet/Split bitwise parity E2E
(§11–12) and the two child-failure fault cases that depend on it.

## 19. Final Gate

**READY_AFTER_SPECIFIC_FIXES** — run the real-subprocess KNet and Split
stop→resume→child parity E2E and assert bitwise equality against a continuous
reference. All other acceptance items pass.

## 20. Evidence Index

Root: `artifacts/benchmark_write_control_backend/<timestamp>/` (not committed):
`preflight/` (branch log, partial commit stats, submodules, schema versions),
`pytest/` (baseline and final collect/full logs).

Commits: `11a451d`, `de79d2a`, `a3ceff1`.
