# Benchmark Execution Visualization — Write-Control Tranche Report

> **Status: historical partial report.** Backend continuation is complete for
> gates B0/B1/C — see `write_control_backend_continuation_report.md`. This page
> is preserved as written; it is not retroactively edited to claim work that
> did not exist at the time.

## 1. Executive Verdict

**NOT_READY.**

This tranche is **incomplete**. Step A of six was implemented and verified; the
remaining five were not started. Reporting it as anything else would be false.

| Step | Scope | Status |
|---|---|---|
| A | Canonical `control_resumable_v1` training path | **DONE, verified** |
| A′ | Direct `train()` vs `resumable_train()` characterization (§6 of the prompt) | **DONE, verified** |
| B | `WorkerManager` resumed-child execution | **NOT IMPLEMENTED** |
| C | Durable resume-action orchestration and recovery | **NOT IMPLEMENTED** |
| D | Write API (`POST` stop/resume, `GET /actions/{id}`) | **NOT IMPLEMENTED** |
| E | Dash Stop/Resume controls | **NOT IMPLEMENTED** |
| F | Real worker + Playwright write E2E | **NOT IMPLEMENTED** |

What was delivered is genuinely finished: implemented, tested, committed, and
regression-clean. It is also the ordering the prompt itself mandates — the
canonical training path first, UI last — so the work that exists is the correct
foundation rather than a partial slice of each layer.

The single most valuable result is §6: the two training loops were an open risk
carried over from the checkpoint tranche, and they are now shown to be
**bitwise identical** given the same batch sequence.

The acceptance gate in §16 of the prompt requires all of B–F. None are present,
so the gate fails and the verdict is `NOT_READY`.

## 2. Authoritative Checkout and Provenance

| Check | Value |
|---|---|
| Repository root | `/home/dss-pc-05/bench` |
| Authoritative HEAD (untouched) | `d92cd0c` on `benchmark-viz/stabilize-release-baseline` |
| Baseline for this tranche | `9454766` on `benchmark-viz/checkpoint-stop-resume` |
| `c0eaaf7` ancestry | confirmed ancestor of the baseline |
| New branch | `benchmark-viz/write-control` |
| Isolated worktree | `/tmp/bench-wc-tranche` |
| Implementation commit | `48b0edb` |
| Submodules | 4, expected revisions, clean at start |

The baseline was taken from Git, not assumed: `9454766` is the checkpoint
branch tip and includes `c0eaaf7` plus the three doc commits after it.

Docs 13–16 existed only as untracked files in the authoritative tree. Exactly
those four were copied into the isolated worktree; nothing else from the dirty
tree was brought across.

## 3. Safety and Worktree Protection

All work happened on a new branch in a separate worktree. The authoritative
tree — carrying ~999 dirty non-`.pyc` entries of unrelated ADCS / Vizard / SNN
research — was never modified.

`git add -A`, `git commit -am`, `git clean`, `git reset --hard`, and broad
checkout were not used. Staging was explicit-path only. Nothing under `runs/`,
`reports/`, or `artifacts/` is in any commit.

## 4. Baseline Results

Re-verified in the clean worktree before any change:

| Gate | Result | Expected |
|---|---|---|
| `pytest --collect-only -q` | 497 collected, 0 errors | 497 |
| `pytest -q` | 496 passed, 1 skipped | 496 / 1 |

Matches the checkpoint tranche exactly. No pre-existing regression, no
environment or submodule discrepancy.

## 5. Canonical Training-Path Implementation

Full contract: `control_plane_training_path_contract.md`.

`bench/control/training_path.py` decides the path once, in `resolve_run_spec`,
from the full certification tuple — model, implementation, device, precision,
`num_workers`, gradient accumulation, and the certification row. Every outcome
carries machine-readable reason codes.

Verified behaviour:

| Case | Path |
|---|---|
| `kalmannet_tsp`, CPU/fp32/0 workers | `control_resumable_v1` |
| `split_knet`, CPU/fp32/0 workers | `control_resumable_v1` |
| `kalmannet_tsp` on CUDA | `legacy_train_v1` (`UNCERTIFIED_DEVICE`) |
| fp16 / `num_workers=4` | `legacy_train_v1` (respective code) |
| `adaptive_knet`, `maml_knet`, `me_split_knet_v0` | `legacy_train_v1` |
| training disabled | `not_applicable` |

Persistence and identity:

- serialised into the resolved spec's `execution` block and round-trips;
- included in `structural_document`, so a legacy run and a control run have
  **different structural hashes** (verified);
- a spec without the field is `legacy_train_v1` permanently — verified for a
  missing block, `None`, and an unrecognised value.

Execution:

- `run_one()` takes an optional `execution_contract`, default `None`, so every
  legacy CLI call is unchanged;
- with `control_resumable_v1` the runner drives `resumable_train()` from update
  0 and emits the path plus certification id on the start event;
- if the adapter cannot honour it, it **raises** — verified that no fallback to
  `train()` occurs, and separately that a legacy/absent contract still reaches
  `train()`.

## 6. Direct Path Parity Characterization

The headline result, and the risk this tranche was most likely to trip over.

**Given an identical batch sequence, the two loops are bitwise identical.**

| | kalmannet_tsp | split_knet |
|---|---|---|
| Initial weights | equal | equal |
| **Final weights** (sha256 over tensor bytes) | **equal** | **equal** |
| Update count | 6 = 6 | 6 = 6 |
| Best step | 4 = 4 | 6 = 6 |

To make the comparison fair, `BatchPlan` gained `shuffle=False` so it can
reproduce `DataLoader(shuffle=False)` exactly. Any difference surviving that
setup would have been a difference between the *loops*; none did.

With shuffling on, the orders **do** differ, for two torch implementation
reasons rather than any semantic one:

1. `DataLoader.__iter__` draws a `random_()` for its worker base seed before
   the sampler draws anything;
2. `RandomSampler` evaluates and discards a trailing `randperm` each epoch.

Matching a discarded draw would pin the plan to torch's sampler internals, so
the explicit plan is kept and the two paths are separated in structural
provenance instead. Per ADR-WC-006 this is recorded as a migration
characterization: exact-resume certification is unaffected (it is
continuous-vs-resumed *within* the resumable path), and shuffled legacy runs
must not be compared directly against shuffled control runs.

No legacy default was changed and no tolerance was widened.

## 7. WorkerManager Child-Resume Implementation

**Not implemented.** `plan_resume()` still validates and reports lineage only;
nothing launches the child through `WorkerManager`. No child allocation, no
child `ResolvedRunSpec`, no lineage registry transaction, no `--launch` flag.

## 8. Action / Idempotency / Recovery

**Not implemented.** The registry's durable `run_actions` table, the stop
action lifecycle, and stop idempotency all remain exactly as the checkpoint
tranche left them — working, but with no resume-action type, no coordinator,
and no restart reconciler.

## 9. Write API

**Not implemented.** No `POST` route exists. The API is still GET-only, and
`BENCH_CONTROL_ENABLE_WRITES` is not read anywhere. Read-only mode is therefore
trivially intact, but that is the previous tranche's property, not a new one.

## 10. Dash Controls

**Not implemented.** No Stop or Resume control was added. The dashboard still
renders zero action buttons.

## 11–12. KNet / Split Full E2E

**Not run.** These depend on §7–§10. The existing checkpoint-tranche E2E
(adapter/service-level stop and resume, fresh-process parity) still passes
unchanged as part of the full suite.

## 13. Fault Injection

**Not run for write-control paths.** The checkpoint tranche's 13 checkpoint
fault points still pass.

## 14. Security Boundary

Unchanged from the previous tranche: loopback-only bind, non-loopback refused
without an explicit override, GET-only surface. The write-mode gate
(`BENCH_CONTROL_ENABLE_WRITES`, loopback-only writes) is **not** implemented,
because there is nothing to gate yet.

## 15. Full Regression

Clean worktree, `pyenv` 3.10.13:

| Gate | Baseline | After this tranche |
|---|---|---|
| `pytest --collect-only -q` | 497, 0 errors | 515, 0 errors |
| `pytest -q` | 496 passed, 1 skipped | **514 passed, 1 skipped, 0 failed** |
| New tests | — | 18 |
| 28 init-provenance regression | pass | pass |
| Observer/telemetry parity | pass | pass |
| Checkpoint atomicity + mutation probes | pass | pass |
| Graceful-stop tests | pass | pass |

No test was deleted, skipped, xfailed, or ignored. New tests write only to
`tmp_path` and temporary directories; none touches repository `runs/` or
`reports/`.

## 16. Third-Party Isolation

No third-party tracked file was modified. All four submodules remain at their
expected revisions with no modified or deleted tracked file. (Importing the
vendored modules during tests leaves untracked `__pycache__`, the pre-existing
V-006 condition — byte cache, not source.)

## 17. Remaining Risks

1. **The tranche is five-sixths unfinished.** B–F are the bulk of the
   write-control work and none of it exists.
2. **`training_path_id` is not yet propagated** to the registry run row, the
   checkpoint manifest, or the API/UI — ADR-WC-004 requires all of these. It is
   currently in the resolved spec, the structural hash, and the start event
   only. Until the registry carries it, capability gating cannot be driven from
   it, which is a prerequisite for D and E.
3. **`_call_resumable_train` is wired but not exercised end-to-end.** Its unit
   behaviour (dispatch, refusal to fall back) is tested; a real certified suite
   run through `run_one` on the resumable path has not been executed, so
   dataset-split/`batch_size` plumbing inside it is unproven.
4. **Shuffled legacy and control runs are not comparable.** Documented, and
   enforced via differing structural hashes, but anyone aggregating historical
   results must respect it.
5. `BatchPlan.shuffle` changes `plan_id` inputs. No production checkpoint
   exists that predates it, but a stored plan from before this commit would
   hash differently.

## 18. Explicitly Deferred Features

Unchanged and still absent: GUI config editor and launch, force terminate,
warm-start API/UI, GPU queue/lease, shared GPU, GPU/AMP/multi-worker exact
resume, Adaptive/MAML/ME-Split exact resume, remote worker,
multi-user/authentication, WebSocket/SSE, arbitrary batch-midpoint resume,
third-party patches. No stub route, disabled-but-promised button, or false
capability was added.

## 19. Final Gate

**NOT_READY.**

The §16 acceptance gate requires certified fresh runs on the resumable path
(met), plus full worker stop/resume for both models, an idempotent restart-safe
write API, real-browser Dash workflows, and write-disabled safety (all unmet).

Recommended next step: propagate `training_path_id` into the registry row and
checkpoint manifest (Risk 2), then implement §7 child launch, since D and E
both depend on it.

## 20. Evidence Index

Root: `artifacts/benchmark_write_control/<timestamp>/` (not committed).

| Path | Contents |
|---|---|
| `preflight/` | git status, worktrees, submodules, branch log |
| `pytest/` | baseline and post-change collect + full-suite logs |
| `parity/` | legacy-vs-resumable characterization output |

Commit: `48b0edb`.
