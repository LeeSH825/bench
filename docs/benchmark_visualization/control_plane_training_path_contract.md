# Control-Plane Training-Path Contract

Implements Decision WC-A (ADR-WC-001 … ADR-WC-006): a certified control-plane
run uses `resumable_train()` from update 0, decided once and recorded as
structural provenance.

## 1. The three path ids

| `training_path_id` | Meaning |
|---|---|
| `control_resumable_v1` | Certified control-plane run. `resumable_train()` from update 0. Stop/Resume eligible. |
| `legacy_train_v1` | The existing adapter `train()` loop. Unchanged. Not Stop/Resume eligible. |
| `not_applicable` | No learning lifecycle (evaluation-only, model-based baselines). |

There is **no user-facing toggle**. Offering one is how a run ends up
half-certified — Stop/Resume advertised for a loop that was never resumable.

## 2. Selection algorithm

Executed exactly once, in `resolve_run_spec()`, in
`bench/control/training_path.py`:

```
not trainable                      -> not_applicable      (NOT_TRAINABLE)
training disabled                  -> not_applicable      (TRAINING_DISABLED)
model ∉ {kalmannet_tsp,split_knet} -> legacy_train_v1     (MODEL_NOT_RESUMABLE)
device != cpu                      -> legacy_train_v1     (UNCERTIFIED_DEVICE)
precision != fp32                  -> legacy_train_v1     (UNCERTIFIED_PRECISION)
num_workers != 0                   -> legacy_train_v1     (UNCERTIFIED_NUM_WORKERS)
grad accumulation != 1             -> legacy_train_v1     (UNCERTIFIED_GRAD_ACCUM)
no certification row               -> legacy_train_v1     (NO_CERTIFICATION_ROW)
otherwise                          -> control_resumable_v1 (CERTIFIED)
```

The decision is keyed on the **full certification tuple**, never on the model
name (ADR-WC-005). Every outcome carries machine-readable `reason_codes` plus
human messages, so the UI never invents its own wording.

The worker does not re-run this. It executes the persisted value.

## 3. No silent fallback

If a run resolves to `control_resumable_v1` and the adapter cannot honour it,
`_try_call_train` raises:

```
training_path_id=control_resumable_v1 was resolved for an adapter that does not
implement the resumable contract: <Adapter>. Refusing to silently fall back to
train() (ADR-WC-003).
```

A run that quietly fell back would be labelled certified while having produced
numbers from the other loop. Tested.

## 4. Old specs

A resolved spec without an `execution` block predates this contract and is
`legacy_train_v1` — permanently. It is never reinterpreted, and neither is an
unrecognised value. Tested in three forms: missing block, `None`, and a
nonsense id.

## 5. Structural identity

`training_path_id` is part of `structural_document()`, so it feeds
`structural_config_hash` and therefore `variant_id`. Two runs differing only in
path get **different structural hashes**.

This is deliberate. The paths derive batch order differently (§6), so treating
them as the same structural condition would silently merge two populations.

## 6. Direct `train()` vs `resumable_train()` characterization

The checkpoint tranche listed "no direct parity test between the two loops" as
an open risk. Closed here.

### Result: the loops are equivalent

Given an **identical batch sequence** (`DataLoader(shuffle=False)` against
`BatchPlan(shuffle=False)`), the two loops are **bitwise identical** for both
certified adapters:

| | kalmannet_tsp | split_knet |
|---|---|---|
| Initial weights | equal | equal |
| **Final weights (sha256 over tensor bytes)** | **equal** | **equal** |
| Update count | 6 = 6 | 6 = 6 |
| Best step | 4 = 4 | 6 = 6 |

So the update ordering, validation cadence, best-state handling and early-stop
semantics of `resumable_train()` match `train()` exactly.

### What still differs: how the order is drawn

With shuffling on, the orders differ:

```
DataLoader : [5,3] [4,2] [1,0] ...
BatchPlan  : [1,0] [4,2] [5,3] ...
```

Two torch implementation details cause this, and neither is a semantic
difference:

1. `DataLoader.__iter__` draws one `random_()` from the generator for its
   worker base seed, before the sampler draws anything.
2. `RandomSampler` with `num_samples == n` evaluates a second `randperm` and
   discards it (`yield from randperm(...)[:0]`), consuming RNG per epoch.

Reproducing that bit-for-bit would pin the batch plan to torch's sampler
internals across versions. Matching a *discarded* draw is not a contract worth
depending on, so the explicit plan is kept.

### Consequences

1. Exact-resume certification is unaffected: it is continuous-vs-resumed
   equivalence **within** `control_resumable_v1`, and that still holds bitwise.
2. `legacy_train_v1` and `control_resumable_v1` are separated in structural
   provenance.
3. **A shuffled legacy run and a shuffled control run must not be compared
   directly as the same condition.** They will differ, and the difference is
   batch order, not method.
4. This is a characterization, not a defect: no legacy default was changed, and
   no tolerance was widened.

## 7. Where the path is recorded

| Location | Status |
|---|---|
| Resolved RunSpec (`execution` block) | **done** |
| `structural_config_hash` | **done** |
| Worker start event (`PHASE_START` payload) | **done** |
| Runner execution contract | **done** |
| Registry run row | *not done* — see the tranche report |
| Checkpoint manifest | *not done* — see the tranche report |
| API run detail / UI | *not done* — see the tranche report |

The last three are outstanding work, listed honestly rather than implied.

## 8. Resume children

A resume child inherits the parent's `training_path_id` unchanged
(ADR-WC-004/007). Child launch itself is not implemented in this tranche.
