# Benchmark Execution Visualization — Checkpoint/Stop/Resume Tranche Report

## 1. Executive Verdict

**READY_FOR_WRITE_CONTROL_TRANCHE.**

All three tranche goals are met with executed evidence:

- **Checkpoint v1** — versioned typed manifest + payload, published atomically,
  digest-verified, with crash reconciliation. 21 tests including six fault
  injection points.
- **Exact resume** — certified for `kalmannet_tsp` and `split_knet` at the
  completed-optimizer-update boundary on CPU/fp32/`num_workers=0`, proven by
  **fresh-process** bitwise parity, with mutation probes that fail when state
  is removed.
- **Graceful stop** — a persistent, idempotent registry-backed request drives
  `RUNNING → STOP_REQUESTED → CHECKPOINTING → INTERRUPTED` with the interrupt
  checkpoint validated *before* the terminal state, and completes with no API
  process involved at all.

Full suite: **497 collected, 496 passed, 1 skipped, 0 failed** (baseline was
449/448/1 — +48 tests, none lost). The API remains GET-only and the dashboard
still renders **zero** action buttons.

The most consequential finding is in §4: Split-KalmanNet keeps
numerically-significant state outside `state_dict()`. A checkpoint built the
obvious way loads the right weights and still diverges. It is now captured
explicitly, and a payload lacking it is refused rather than resumed
approximately.

## 2. Authoritative Checkout and Provenance

| Check | Value |
|---|---|
| Repository root | `/home/dss-pc-05/bench` |
| Baseline commit | `d92cd0c` (stabilization tranche HEAD) |
| Stabilization ancestry | `355b6ca`, `179a42d`, `ee862a2` all confirmed ancestors |
| Feature branch | `benchmark-viz/checkpoint-stop-resume` |
| Implementation commit | `c0eaaf7` |
| Work worktree | `/tmp/bench-ckpt-tranche` (isolated, submodules initialised) |
| Submodules | 4, at expected revisions; no tracked file modified (see §17) |

`bench/control/**`, `bench/ui/**`, the control tests, FastAPI and Dash were all
verified present before any change. This is the correct checkout;
`WRONG_CHECKOUT_OR_UNSAFE_WORKTREE` does not apply.

## 3. Safety Snapshot and Unrelated Work Protection

Preflight snapshot: `artifacts/benchmark_checkpoint_resume/20260731T020915Z/preflight/`
(status, untracked list — 221 entries, submodule state, diffstat, log).

All work was done in a **separate worktree on a new branch** cut from the
baseline, so the authoritative working tree — which carries ~999 dirty
non-`.pyc` entries of unrelated ADCS / Vizard / SNN / Phase-6 research work —
was never touched.

`git add -A`, `git commit -am`, `git clean`, `git reset --hard`, and forced
checkout were not used. Staging was by explicit path only.

Pre-existing untracked checkpoint-adjacent work was identified and left alone:
`bench/visualization/replay_checkpoint_contract.py` and friends implement a
**replay** contract (inference-only: weights + config + normalisation for
Vizard trajectory replay, schema `replay_checkpoint_contract_v1`). That is a
different concern from control-plane exact-resume state, and there is no
namespace collision with `bench/control/checkpoints/`.

**V-008 confirmed again.** Running the suite mutates tracked `runs/` and
`reports/` data. Per ADR-CSR-014 this was not refactored; the full suite was
run in the disposable worktree and the resulting diff was excluded from
staging. Nothing under `runs/` or `reports/` is in any commit.

## 4. KNet/Split Training-State Audit

Full audit: `checkpoint_resume_state_audit.md`. Summary of what the code
actually does — read from the adapters and the pinned submodules, not assumed
from the papers:

Both adapters share one shape: a single `Adam`, no scheduler, no `GradScaler`,
`while updates < max: for batch in DataLoader(shuffle=True, generator=g,
num_workers=0)`, `best_state` held **in memory** and only persisted at the end,
and recurrent hidden state re-initialised **per batch** (KNet via
`init_hidden_KNet()`/`InitSequence()`, Split via `filter.reset(clean_history=True)`).

No persistent hidden state crosses an update boundary, which is what makes the
optimizer-update boundary sufficient here rather than merely convenient.

### The finding

`third_party/Split_KalmanNet/GSSFiltering/dnn.py`:

```python
self.hn1 = torch.randn(...);  self.hn1_init = self.hn1.detach().clone()
self.hn2 = torch.randn(...);  self.hn2_init = self.hn2.detach().clone()

def initialize_hidden(self):
    self.hn1 = self.hn1_init.detach().clone()
    self.hn2 = self.hn2_init.detach().clone()
```

`hn1_init`/`hn2_init` are **seed-dependent random constants** created at
construction and used as the GRU's initial hidden state for every sequence of
every update — and they are plain attributes, so `state_dict()` does not
contain them.

Found because every resume in the harness deliberately builds the resuming
adapter with a *different seed*. KNet matched bitwise; Split diverged from the
first post-resume update. Re-running with the same seed made Split match, which
localised the cause to seed-dependent setup state rather than the cursor or the
optimizer.

Fixed in the adapter layer (`_ckpt_extra_state` / `_ckpt_restore_extra`),
declared in `required_conditional_state`, and a payload lacking it is refused.
**No third-party file was modified**, so no exception record is required.

Side note worth recording: because `model.pt` also omits these tensors, a warm
start from `model.pt` does not reproduce Split's evaluation bitwise either.
Pre-existing and out of scope, but a real reproducibility caveat.

## 5. Checkpoint v1 Schema and Atomicity

Schema: `checkpoint_v1_schema.md`. `CHECKPOINT_SCHEMA_VERSION = 1`; a mismatched
version is refused, never migrated.

Publication follows the mandated order — payload temp write → fsync → digest →
manifest temp write → fsync → payload replace → manifest replace → dir fsync →
registry transaction → event append. The digest is computed on the temp file
*before* the rename, so what is catalogued is exactly what became durable.

`kind` is a typed field (`periodic`/`best`/`interrupt`/`final`), never inferred
from a filename. One directory per checkpoint; re-saving an existing id raises.
`*.tmp-write` files are never listed, validated, or auto-deleted.

Loading is gated three ways: digest, schema/inventory, and a control-root path
check — payloads are pickle, so path provenance is part of the trust boundary.

## 6. Registry Migration and Reconciliation

Migration **2**, forward-only and additive (`ADD COLUMN` / new index / new
table only). Verified by upgrading a synthetic v1 registry: pre-existing rows
preserved, new columns defaulting to `UNVERIFIED` (**not** a false `VALID`),
automatic backup taken.

Adds checkpoint validation/compatibility/lineage columns, the acknowledge and
complete split plus `result_checkpoint_id` on `run_actions`, and a keyed
`exact_resume_certifications` table.

Reconciliation outcomes, all under test:

| Situation | Outcome |
|---|---|
| Complete package, no row | Digest verified, then **adopted** |
| Row, package missing | Marked `INVALID`; row kept as evidence |
| Row, package corrupt | Marked `INVALID` |
| Temp leftovers | Reported, **not deleted** |

## 7. Resumable Cursor and RNG/Sampler State

The batch schedule is made explicit rather than reverse-engineered from
DataLoader internals (§7 option B). `BatchPlan(dataset_length, batch_size,
seed, drop_last)` draws each epoch's permutation from a generator seeded from
`(seed, epoch)`, so epochs are independent and any global position is seekable
in O(1). The cursor is `global_update` + `batch_plan_position` + `plan_id`.

Skip-*K*-from-a-fresh-iterator was rejected: it makes the claim depend on
sampler internals and worker semantics this tranche has not certified.

Python, NumPy, and torch CPU RNG are captured and restored. This is **opt-in**:
`train()` is untouched and still uses the existing DataLoader path, so no
existing numerical result moves and last tranche's observer/telemetry inertness
certification still holds.

## 8. KNet Exact-Resume Certification

CPU / fp32 / deterministic algorithms / `num_workers=0` / single thread.
6 updates continuous vs 3 + interrupt + resume.

Bitwise identical on: final `state_dict` (sha256 over tensor bytes), Adam
state, full per-update loss sequence, full validation history, update count,
batch-plan position, best step, best metric. Equality is bitwise, not
`allclose`; no tolerance was introduced.

Verified three ways: in-process, through a real on-disk checkpoint package, and
in a **freshly spawned interpreter**. The resuming adapter is always built with
a different seed.

## 9. Split-KNet Exact-Resume Certification

Identical harness and identical result — after the §4 hidden-state fix. Same
three paths including fresh-process, same bitwise comparison set.

`training_mode` is recorded as
`supervised_single_optimizer_split_deviation`: this adapter uses one optimizer
slot, not the paper's alternating optimization. Certifying exact resume says
the *implementation* resumes exactly; it says nothing about paper fidelity, and
`paper_fidelity_status` is unchanged.

### Mutation probes

A parity test that cannot fail proves nothing. Each probe must fail, and does:

| Probe | Result |
|---|---|
| Drop Adam optimizer state | **Detected** — weights diverge (both models) |
| Batch cursor off by one | **Detected** — weights diverge (both models) |
| Drop Split GRU hidden state | **Refused** at restore; without the guard, weights diverge |

## 10. Graceful Stop State Machine

A stop request is a **registry row**, not a signal (ADR-CSR-011): idempotent by
key, acknowledged when the worker sees it, completed only when the interrupt
checkpoint is durable. Signal handlers set a flag and nothing else — no
`torch.save`, no SQLite write (ADR-CSR-010), verified by test.

Observed transition sequence: `RUNNING → STOP_REQUESTED → CHECKPOINTING →
INTERRUPTED`, exit code **10**, with the checkpoint validated before the
terminal transition.

Checkpoint write failure → `FAILED`, exit code **50**, action marked `FAILED`,
and **no checkpoint row** — a stop that cannot persist state never looks
resumable (DND-CSR-004).

Repeating a request with the same key yields one action row and one interrupt
checkpoint.

**API independence** is tested directly: the request is written, the registry
closed, and a separate process with no API, no server, and no live requester
completes the whole stop lifecycle.

## 11. Resume Child-Run Lineage

`plan_resume()` validates without mutating. Lineage carries `parent_run_id`,
`resumed_from_run_id`, `resumed_from_checkpoint_id`; the child inherits the
parent's `variant_id` (a resume is execution lineage, not a new variant).

Parent immutability is asserted explicitly: state, `state_version`, exit code,
checkpoint rows, and the on-disk checkpoint file list are all unchanged after
planning a resume. Resuming from a `RUNNING` run is refused.

End-to-end: stop at update 3 → plan resume → restore into a differently-seeded
adapter → finish → **bitwise identical to continuous**.

## 12. Capability Certification Matrix

Full matrix: `exact_resume_certification_matrix.md`.

Certified: `kalmannet_tsp` and `split_knet`, schema 1, `optimizer_update`
boundary, fp32/cpu/0 workers. Not certified: `adaptive_knet`, `maml_knet`,
`me_split_knet_v0`, `mb_kf` — and these do not inherit the mixin, so they
cannot acquire the capability accidentally (asserted by test).

Uncertified for *every* model: `cuda`/`gpu`/`mps`, fp16/bf16/AMP, any
`num_workers != 0`, distributed, gradient accumulation, any other boundary.
`is_certified(..., device_class="cuda")` returns `False` even for a certified
model.

The coarse `supports_exact_resume` flag in `capabilities.py` was deliberately
**left `False`**: it is config-independent and would over-claim for a GPU run.
The certified envelope is exposed as keyed rows instead.

## 13. CLI and Read-Only API Changes

CLI (write operations live here, not on HTTP):

```
checkpoints list --run-id | show --checkpoint-id
            validate --checkpoint-id | reconcile --run-id
stop   --run-id [--idempotency-key KEY]
resume --checkpoint-id
```

Read-only API, all GET:

```
GET /api/v1/runs/{run_id}/checkpoints
GET /api/v1/checkpoints/{checkpoint_id}
GET /api/v1/runs/{run_id}/lineage
GET /api/v1/runs/{run_id}/actions
GET /api/v1/capabilities/exact-resume
```

Verified: the app exposes only `GET`/`HEAD`; `POST` to a checkpoint route
returns **405**. `graceful_stop_api`, `resume_api`, `exact_resume`,
`warm_start_api`, `checkpoint_catalog_write` all remain `false`; the new
`graceful_stop_backend` / `exact_resume_backend` flags describe the backend
without implying a write surface.

## 14. Fault Injection Results

| Fault point | Expected | Result |
|---|---|---|
| Crash after payload write | No checkpoint, no row | PASS |
| Crash before payload rename | No checkpoint, no row | PASS |
| Crash before manifest write | No checkpoint, no row | PASS |
| Crash after manifest rename, before registry | Package adopted after digest check | PASS |
| Payload bit-flip | Digest mismatch; load refused | PASS |
| Payload truncated | Detected; load refused | PASS |
| Payload deleted | Detected; row marked INVALID | PASS |
| Manifest deleted | Not a checkpoint | PASS |
| Future schema version | Refused, not migrated | PASS |
| Package deleted after cataloguing | Row marked INVALID | PASS |
| Temp leftover present | Reported, not catalogued, not deleted | PASS |
| Load from outside control root | Refused | PASS |
| Payload without optimizer slots | Rejected as not-a-resume | PASS |

No fault point produced a partial checkpoint marked valid.

## 15. Full Regression and Legacy Results

Clean worktree, `pyenv` 3.10.13:

| Gate | Result |
|---|---|
| `pytest --collect-only -q` | **497 collected, 0 errors** |
| `pytest -q` | **496 passed, 1 skipped, 0 failed** |
| Baseline for comparison | 449 / 448 / 1 |
| New tests added | 48 |
| `tests/test_viz_init_provenance_comparison.py` | 28 passed |
| `tests/test_control_*.py` | all passed |
| Observer/telemetry parity (previous tranche) | still passes |
| Playwright DOM | **0 action buttons** on Runs and Run Detail |

Two pre-existing tests were **updated, not skipped or deleted**, because their
premise changed by design: the registry schema version is now 2, and
`STOP_REQUESTED`/`CHECKPOINTING`/`INTERRUPTED` are no longer schema-only since
this build genuinely produces them. `RESUMING` remains schema-only — a resume
creates a child run rather than returning the parent to a running state.

New tests write only to `tmp_path`; none touches repository `runs/` or
`reports/`.

## 16. Performance Characterization

Not a headline goal, but measured while running the suite: the full checkpoint,
resume, and stop test suite (48 tests, real third-party models, one spawned
interpreter per fresh-process test) runs in **~13 s**. A tiny checkpoint
package is ~200 KB; `torch.save` plus fsync plus sha256 dominates and is
milliseconds at this size.

No scale claim is made for large models — payload size scales with parameter
count and the fsync cost with it. That remains uncharacterised.

## 17. Third-Party Isolation

**No third-party tracked file was modified.** All four submodules remain at
their expected revisions, and `git status --porcelain` inside each shows no
modified or deleted tracked file.

Stated precisely, because the distinction matters: after running the suite two
submodules do show *untracked* `__pycache__` entries
(`KalmanNet_TSP/{KNet,Simulations}/__pycache__/`,
`MAML_KalmanNet/MAML-KalmanNet/__pycache__/*.cpython-310.pyc`). These are byte
caches produced by importing the vendored modules during tests — the same V-006
condition recorded in the previous tranche — not source changes. They were not
deleted here: MAML_KalmanNet vendors some `.pyc` files as *tracked* upstream
content, and an over-broad `__pycache__` sweep in the previous tranche briefly
removed them. Cleanup, if wanted, must exclude tracked paths.

Split's hidden-state problem was the one place where a third-party change might
have seemed easier (registering `hn1_init`/`hn2_init` as buffers upstream). It
was solved in the adapter layer instead, via the `_ckpt_extra_state` hook. No
exception record is required.

## 18. Remaining Risks and Unsupported Modes

1. **Certification is narrow by construction** — CPU/fp32/0 workers, tiny
   fixtures, short runs. Nothing here says anything about GPU, AMP, multi-worker
   loaders, or long runs, and the matrix says so explicitly.
2. **Resumable training is a second code path.** `train()` and
   `resumable_train()` share the numerical body through hooks but not the loop.
   They are asserted equivalent only through each adapter's own forward/validate
   hooks, not by a direct `train()`-vs-`resumable_train()` parity test. Worth
   adding.
3. **The resume child run is planned, not executed.** `plan_resume()` validates
   and reports lineage; actually launching the child through `WorkerManager` is
   left to the write-control tranche, so end-to-end resume is proven at the
   adapter/service level rather than through a full supervised child worker.
4. **`best_state` lives in memory during training.** It is checkpointed, but a
   hard crash between checkpoints still loses it — unchanged from before.
5. **V-008 persists.** The suite still mutates tracked `runs/`/`reports/`.
   Deliberately not addressed here.
6. **Split's `model.pt` warm start is not bitwise reproducible** for the reason
   in §4. Pre-existing; worth a follow-up.
7. **No `CANCELLED`-before-first-update policy is implemented.** ADR-CSR-012
   allows it; a stop requested before any update currently still produces an
   interrupt checkpoint at update 0 rather than `CANCELLED`.

## 19. Explicitly Deferred Write-Control/UI Features

Not implemented, and not exposed: public write API (`POST`/`PUT`/`PATCH`/
`DELETE`), Dash Stop/Resume/Terminate buttons, force-terminate API, warm-start
launch API, config GUI launch, GPU scheduler/lease enforcement, shared-GPU
execution, remote workers, authentication/multi-user, WebSocket migration,
MAML/Adaptive/ME-Split exact resume, arbitrary batch-midpoint resume.

External `SIGKILL` → `ORPHANED` semantics are unchanged.

## 20. Final Gate Decision

**READY_FOR_WRITE_CONTROL_TRANCHE.**

Checkpoint v1 is versioned and atomic; corruption, partial writes, and
incompatible checkpoints are all refused; both certified adapters pass
fresh-process bitwise resume parity; the mutation probes prove the tests are
sensitive; the stop state machine holds its ordering and validates before going
terminal; resume is an immutable child with complete lineage; uncertified
models and envelopes are refused; the full suite and the legacy regressions
pass; the API and dashboard are still read-only with zero action buttons; and
third-party source is clean.

The next tranche can expose the write-control surface. It should start with
§18.3 — launching the resumed child run through the supervised worker path.

---

## Appendix A. Changed File Manifest

New:
```
bench/control/checkpoints/{__init__,schema,atomic,batchplan,payload,
    compatibility,validation,service,reconciliation,training,stop,
    lifecycle,certification}.py
bench/models/checkpoint_support.py
tests/checkpoint_fixtures.py
tests/test_control_checkpoint_schema_atomicity.py
tests/test_control_exact_resume_certification.py
tests/test_control_graceful_stop.py
docs/benchmark_visualization/{checkpoint_resume_state_audit,
    checkpoint_v1_schema, exact_resume_certification_matrix,
    graceful_stop_operator_guide, checkpoint_stop_resume_tranche_report}.md
docs/benchmark_visualization/checkpoint_stop_resume_tranche_summary.json
```

Modified:
```
bench/models/{kalmannet_tsp,split_knet}.py      # checkpoint hooks only
bench/control/registry/{migrations/__init__,schema,sqlite}.py
bench/control/cli.py                            # checkpoints/stop/resume
bench/control/api/routers/{runs,system}.py      # read-only additions
tests/test_control_{registry_events,api_dashboard}.py   # premise changed
```

## Appendix B. Commands

```bash
# preflight
git rev-parse --show-toplevel && git rev-parse HEAD
git merge-base --is-ancestor 355b6ca HEAD && echo baseline-ok
git worktree add -b benchmark-viz/checkpoint-stop-resume /tmp/bench-ckpt-tranche d92cd0c
git submodule update --init --recursive

# regression
python -m pytest --collect-only -q
python -m pytest -q
python -m pytest -q tests/test_control_exact_resume_certification.py
python -m pytest -q tests/test_control_checkpoint_schema_atomicity.py
python -m pytest -q tests/test_control_graceful_stop.py

# operator surface
python -m bench.control.cli checkpoints list --run-id <id>
python -m bench.control.cli checkpoints validate --checkpoint-id <id>
python -m bench.control.cli stop --run-id <id> --idempotency-key k1
python -m bench.control.cli resume --checkpoint-id <id>
```

## Appendix C. Evidence Index

Root: `artifacts/benchmark_checkpoint_resume/20260731T020915Z/` (not committed).

| Path | Contents |
|---|---|
| `preflight/` | git status, untracked list, submodule state, diffstat, log |
| `pytest/` | collect and full-suite logs, before and after |
| `browser/` | `result.json` (0 action buttons), `runs.png`, `run_detail.png` |
| `resume/` | parity probe output for both adapters |
