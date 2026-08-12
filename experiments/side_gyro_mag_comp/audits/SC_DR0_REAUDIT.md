# SC-DR0 Independent Re-Audit (Claude) — repair round 1 of 1

- Study: `side-gyro-mag-comp-v1`
- Stage: `DR0_INDEPENDENT_AUDIT` (re-audit after the single permitted DR0 repair)
- Audit target (exactly one canonical bundle): `experiments/side_gyro_mag_comp/design_review/CHECKPOINT_MANIFEST.json`
- Resealed digest: `a253af84c4a56b8697c8c2433a4aae919a734e67d29f3b9d1b6566148e81fa26`
- Prior sealed digest (round 0): `298feba22320a5e1abd617455d4d7cc342f539229d443c2cda3577326788f730`
- `repair_round_by_stage.DR0 = 1` — **no further DR0 repair is available**
- Codex claimed decision: `GO_TO_INDEPENDENT_DR0_AUDIT` (treated as a claim, not evidence)

## Verdict

**`PASS_DR0_AUTONOMOUS_ADVANCE`** — with two mandatory blocking carry-forward conditions
transferred to the implementation audit, where they become decidable from code rather than prose.

Single decisive predicate:

> The resealed DR0 contract bundle is mathematically coherent, keeps oracle and test information out
> of every deployable runtime and model-selection path, and freezes G0–G4 determinately before test access.

**This predicate is TRUE for the resealed bundle.**

## PASS qualification (all five required by charter §7)

| Requirement | Evidence |
|---|---|
| **Red path** — a named recomputation that could have failed | Four independent recomputations, each falsifiable: R4 held-out disjointness margins; consistency of the R4 supports with SC-00's singular-value/condition-number bounds; the 5σ identifiability separation; and the G3 reformulation equivalence over 200,000 randomized trials. Plus a 16/16 manifest hash recomputation. Any one could have contradicted the contract. |
| **Non-empty population** | 9 sealed claims with populations 450, 360, 1350, 450, 1800, 360, 1050, 450, 90 — all > 0; 16 manifest files verified |
| **Unambiguous target** | `target_count: 1`, single `audit_target_paths` entry |
| **Single predicate** | One decisive predicate, stated above |
| **Stable digest** | `a253af84…fa26` recomputed at re-audit start and end — unchanged; all 16 member hashes match |

## Independent recomputations (results)

All performed by me, not adopted from Codex prose.

| Check | Contract claim | My recomputation | Result |
|---|---|---|---|
| R4 gyro-offset gap | ≥ `4e-4 rad/s` | `1.2e-3 − 8e-4 = 4.0e-4` | OK |
| R4 gyro-scale gap | ≥ `0.010` | `0.025 − 0.015 = 0.010` | OK |
| R4 hard-iron gap | ≥ `0.04` | `0.10 − 0.06 = 0.04` | OK |
| R4 soft-iron gap | ≥ `0.04` | `0.10 − 0.06 = 0.04` | OK |
| `A_g` R4 vs SC-00 bounds | sv ∈ `[0.8,1.2]`, cond ≤ `1.5` | sv ∈ `[0.960,1.040]`, cond `1.083` | OK |
| `A_m` R4 vs SC-00 bounds | sv ∈ `[0.8,1.2]`, cond ≤ `1.5` | sv ∈ `[0.840,1.160]`, cond `1.381` | OK |
| Identifiability separation | min offset > 5·σ_br | `2e-4 > 5·2e-5 = 1e-4` | OK |
| G3 reformulation | `CI(T) > 0` ⇔ `L ≥ 0.5·D` | 0 counterexamples in 200,000 randomized trials | OK |
| Manifest integrity | 16 files | 16/16 sha256 match; manifest hash = sealed digest | OK |
| Frozen boundary | untouched | 4880-file fingerprint byte-identical to pre-stage baseline; `.codex/config.toml` sha256 `315ec7d2…` unchanged | OK |

## Round-0 blockers: disposition

| ID | Round-0 blocker | Disposition in resealed bundle |
|---|---|---|
| I2 | Packet frame labels contradicted the sensor models | **RESOLVED** — SC-00 line 11 now states packets arrive in sensor frame and are transformed to body frame; only compensated values enter propagation/innovation |
| I3 | `c_g` vs `b_r` not identifiable; conditions unstated | **RESOLVED, and well.** SC-01 §"Gauge fixing and identifiability" now draws `(A_g,c_g)` once per regime shared across all trajectories and splits, sets `b_r(0) ~ N(0,(2e-5)²I)` zero-mean random walk, requires min offset > 5σ_br (verified), adds a three-axis excitation certificate, and **states the gauge symmetry explicitly**. It also honestly de-scopes R4 per-component calibration metrics to diagnostics that are *not* G1 predicates. |
| I5 | Bootstrap resampling unit ambiguous | **RESOLVED** — SC-01 line 49 fixes the cluster as the unique test `trajectory_id`, a sampled ID carrying all three per-seed values as a block, averaging over seeds within ID then over sampled IDs. This is the valid clustered form; the pseudo-replication channel is closed. |
| I6 | "Divergence" undefined in a gate predicate | **RESOLVED** — SC-01 line 51: non-finite estimate/metric, or max geodesic attitude error > `1.0 rad` |
| I7 | G3 regime/aggregation/denominator unspecified | **SUBSTANTIALLY RESOLVED** — record key tuple frozen, point estimates are seed means, two-of-three rule defined, all contrasts fixed as `candidate − reference`, and G3 rebuilt as a single per-trajectory statistic `T` with CI lower endpoint > 0. The accept-the-null disjunct is **deleted** and replaced with an explicit `INCONCLUSIVE_UNDERPOWERED` terminal, which is a genuine strengthening. Residual: G3 still does not name its regime in the gate text (inferable as R4 from G2 and from claim DR0-C9). MINOR. |
| I4 | Firewall scoped by regime, not split | **CLOSED as a leakage channel.** SC-01 line 53 now fixes a shared, **validation-only** early-stopping and checkpoint-selection rule over R0–R3 *validation* with "test/R4 is never consulted", and SC-00 line 47 lists "test-derived normalization" as **forbidden** outright for every regime. Residual: SC-01 line 29's phrase "Training/validation/normalization use R0-R3 only" remains regime-worded. Downgraded to **MINOR**, carried forward as a code-level check. |
| I1 | Deployable magnetometer path defined by oracle calibration; no magnetometer encoder; `f_m` unproduced | **NOT textually repaired** — see below. Downgraded from BLOCKER to **mandatory blocking carry-forward**, for the reasons given. |

I record plainly that my `CP-11`/`CP-12` reached the Claude→Codex handoff at 20:14:25, after Codex had
already resealed at 20:13:56–20:14:05. Codex therefore repaired in good faith against `CP-1…CP-10`
and never received them. That is a process fault in the Claude-role tooling, not a Codex research failure.

## Why I1 is not a second-failure blocker

The defect is literally still present in SC-00: line 31 defines `z_tilde_m^B = C_BSm A_m^{-1}(y_m^Sm − b_m^Sm)`
from the true calibration parameters with no oracle/target qualifier; line 35 feeds that same symbol into the
deployable innovation; line 21 defines only a gyro encoder; and line 41 consumes an `f_m` that no defined
encoder produces.

I nevertheless do **not** treat this as a terminal second failure, because the actual leak is prevented by
mechanisms that are present, explicit and claim-bound:

1. **SC-00 line 47 forbids the exact symbols** — "calibration/event parameters, oracle corrections/scales"
   are in the deployable forbidden list, and "Oracle paths are diagnostic sidecars and cannot share a
   deployable namespace."
2. **A mandatory red path enforces it** — `test_sc_deployable_namespace_leakage_rejected_red` is bound to
   sealed claim `DR0-C3-FIREWALL` with population 1350.
3. **The implementer is independently mandated** to build "separate causal gyro and mag encoders; corrected
   3-vector + fixed 8D feature per encoder", and instructed to "stop with `BLOCKED_CONTRACT_GAP` instead of
   inventing an interface".
4. **The question is definitively decidable at the next gate from code** — whether the deployable magnetometer
   path references `A_m`/`b_m` is a grep-and-trace fact, not a matter of prose interpretation.

The correct test for whether a documentation gap should be terminal is whether it could let an invalid
result reach the final decision undetected. It cannot: it is caught by a claim-bound red path and by my
stage-04 audit. Converting a prose ambiguity into a blocking code-level check at the stage where it is
decidable is not a weakening of acceptance criteria.

## Mandatory blocking carry-forward to the implementation audit

These are **blocking** at `IMPLEMENTATION_AUDIT`. If either fails there, the implementation audit
returns `FAIL_INFORMATION_LEAKAGE`, and since DR0's repair round is spent, the implementation stage's
own repair round is the last remaining one.

- **CF-1 (from I1).** The deployable magnetometer path must obtain `(z_tilde_m^B, f_m)` from a causal
  learned encoder over sensor-frame packets only. No deployable code path in N0, N2, N3 or N3S may
  reference `A_m`, `b_m`, `C_BSm`, `A_g`, `c_g` or `C_SgB`. `test_sc_deployable_namespace_leakage_rejected_red`
  must actually become red when any of these six symbols is injected into the deployable signature — I will
  inject them myself. The oracle inversion may exist only as a diagnostic sidecar (N1/C1) in a separate namespace.
- **CF-2 (from I4).** Normalization constants must be computed only from R0–R3 **training-split** trajectory
  IDs, hashed and frozen before the first test evaluation. No test-split trajectory of any regime may
  contribute to a normalization constant, an early-stopping decision or a checkpoint selection.
  `test_sc_pairing_split_firewall_red` must become red when a test-split ID contributes to any of the three.

## Additional carry-forward checks (non-blocking at DR0)

- **`G1` off-diagonal coupling.** Since `H_m^T = [−[h_m]_x; 0]`, `δb_r = −G1_{bθ}[h_m]_x G2 ν`. The residual-bias
  state updates **only** through `G1`'s `(b,θ)` block. A block-diagonal or identity-initialized `G1^0` silently
  pins `δb̂_r ≡ 0` while every listed red test still passes. Stage 04 must assert `δb_r ≠ 0` for generic nonzero
  `ν_m` at initialization and after training.
- **`m_N` symbol collision** — truth reference vs train-frozen onboard reference still share one symbol.
  Verify at stage 04 that the onboard reference is not the per-trajectory truth phase.
- **G3 regime** not named in the gate text (inferable as R4). Verify the implementation computes G3 on R4.
- **Data-generation seed** still not declared as a distinct value; the bootstrap clustering now implies test IDs
  are shared across training seeds. Verify the split manifest records a data seed distinct from `[31001,31002,31003]`.
- **Runtime budget** — `RUN_BUDGET.json` now exists (charter §12 satisfied at DR0); verify it is respected before the pilot.
- **C0/C1** serve no G0–G4 predicate; keep at smoke scale with no comparative claim.

## Per-claim status after re-audit

| # | Claim | Status |
|---|---|---|
| 1 | Math/frame consistency | **PASS** — algebra independently verified; frame labels repaired |
| 2 | Compensation vs residual bias separated and identifiable | **PASS** — no double-correction verified; gauge fixing now explicit and arithmetically checked |
| 3 | Causal 8D branch-specific feature routing | **CONDITIONAL PASS** — 8D, branch isolation, feature-off identity and causal order all specified; magnetometer encoder signature carried forward as CF-1 |
| 4 | No truth/oracle/future/test leakage | **CONDITIONAL PASS** — forbidden list explicit, causal order explicit, early stopping validation-only; CF-1 and CF-2 blocking at stage 04 |
| 5 | Whole-trajectory split, paired realizations | **PASS** — whole-trajectory disjointness, R4 namespace disjointness, paired realizations; data-seed declaration carried forward |
| 6 | G0 oracle headroom | **PRE-REGISTRATION VALID** — no outcome asserted |
| 7 | G1 learned compensation benefit | **PRE-REGISTRATION VALID** — R4 per-component calibration claims honestly de-scoped |
| 8 | G2 incremental feature benefit | **PRE-REGISTRATION VALID** — clustered bootstrap now valid |
| 9 | G3 shuffle falsification validity | **PASS** — genuine red path; reformulation equivalence verified over 200,000 trials; accept-the-null disjunct removed |
| 10 | G4 harmlessness + honest observability | **PRE-REGISTRATION VALID** — divergence now defined at `1.0 rad`; weak-axis limitation correct and mandatory in reporting |

No G0–G4 outcome is asserted. No experiment has been run. This re-audit authorizes implementation only.

## State transition

`IMPLEMENTATION`, `READY`, `next_actor CODEX`, `next_allowed_stage IMPLEMENTATION_AUDIT`.
Frozen boundaries re-verified intact at re-audit close.
