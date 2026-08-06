# SC-DR0 Independent Audit (Claude)

- Study: `side-gyro-mag-comp-v1`
- Stage audited: `DR0_DESIGN_REVIEW` sealed bundle, audited at `DR0_INDEPENDENT_AUDIT`
- Audit target: `experiments/side_gyro_mag_comp/design_review/CHECKPOINT_MANIFEST.json` (single target, `target_count = 1`)
- Sealed digest: `298feba22320a5e1abd617455d4d7cc342f539229d443c2cda3577326788f730`
- Source commit: `052d2f7217b964b1fa4e80bd643716b433780f08`
- Repair rounds consumed on `DR0` before this audit: `0`

## Verdict

**`FAIL_REVISE_MATH` — repairable. Implementation is NOT authorized.**

Single audited predicate: *the sealed DR0 bundle is mathematically coherent, information-boundary clean, and
sufficiently frozen that G0-G4 verdicts are determined before test access.* This predicate is **FALSE**.

The failure is repairable: every defect is a text-level change to
`docs/research/side_gyro_mag_comp/SC_00_*`, `SC_01_*`, the design-review bundle, and the Codex handoff.
No architecture, model family, feature dimension, conditioning mechanism, split, or numeric threshold changes.
`DR0` has consumed zero repair rounds, so one automatic repair round remains.

## Checkpoint qualification

| Requirement | Result |
|---|---|
| Stable digest before and after audit | PASS — manifest bytes and all 7 member hashes re-verified unchanged at audit start and end |
| Unambiguous target | PASS — exactly one `CHECKPOINT_MANIFEST.json` repo-wide; `target_count = 1` |
| Non-empty population | PASS — all 7 declared claim populations > 0 |
| Frozen-artifact integrity | PASS — 7/7 manifest sha256 recomputed and matched independently by two auditors |
| Path ownership | PASS — no Codex write into Claude-owned paths; `.codex/config.toml` byte-identical to pre-install backup |
| Boundary preservation | PASS — no DR0-window write touched main Phase 2, SpikeRA-KalmanNet, or frozen Phase 0-1 |
| Single predicate per verdict | PASS |
| Red path for the verdict | PASS — see "Red path" below |

The digest-composition rule (`sha256` over raw bytes of `CHECKPOINT_MANIFEST.json`) is **undocumented**; three
plausible alternative compositions were computed and falsified before the rule was accepted.

## Red path for this audit

This verdict is falsifiable, not narrative. Each blocker below is accompanied by a recomputation or a
counterexample that fails when the blocker is absent:

- **FC-1** falsified by direct substitution: the reparameterization `(c_g, b_r) -> (c_g + A_g C_SgB v, b_r - v)`
  leaves `y_g^Sg(t)` invariant for every `t` and every constant `v in R^3`. If SC-01 anchored the gauge, no such
  `v` would exist.
- **FC-2** falsified by exhaustive grep of the side-study canon: zero numeric parameter supports for R1-R4.
- **FC-3** falsified by the algebraic identity `(N3S - N2) = L - D`, verified over 100,000 random draws, plus a
  Monte-Carlo false-pass rate under a true null.
- **FC-5** falsified by set intersection of red-path identifiers: SC-00's 13 frozen names against the handoff's
  7 cited names gives the **empty set**.

## Blocking failure classes

### FC-1 (BLOCKER, math) — the declared deterministic/residual role split is unidentifiable

SC-00 requires the compensator to remove deterministic calibration only and to leave `b_r` for the MEKF.
For any constant `v in R^3`, `(c_g, b_r) -> (c_g + A_g C_SgB v, b_r - v)` leaves the gyro packet stream
bitwise invariant. The likelihood is exactly flat along a 3-dimensional gauge orbit, so single-trajectory
separation is impossible **in principle**, not merely hard. `A_g` is separable only under persistent
three-axis excitation, which SC-01 never requires.

SC-01 does not break the degeneracy: it never states that `(A_g, c_g)` are shared across trajectories within a
regime, and never states `E[b_r] = 0` or any prior scale separation. Its "in-support parameter ranges" wording
affirmatively suggests per-trajectory sampling, under which the degeneracy survives at population level.

Consequence: on R4 — the **declared primary method-lock endpoint** — `c_g` is drawn from a support disjoint
from training, so no causal deployable encoder can recover it. The R4 secondary metrics
"gyro corrected-rate RMSE against the residual-bias-retaining target" and "residual gyro-bias RMSE" are not
separately interpretable, and G1's R1 gyro criterion can be satisfied by a compensator that violates the
declared role split. `SC_DR0_REVIEW.md:5` ("SC-00 closes the deterministic-calibration versus residual-bias
ambiguity") is an overclaim: SC-00 states a requirement, not an identifiability result.

### FC-2 (BLOCKER, scope) — no numeric regime supports exist

SC-01 refers to "in-support parameter ranges" (R3) and "held-out parameter ranges disjoint from R1-R3 support"
(R4) but defines **no numeric support anywhere** for R1, R2, R3, or R4. Therefore: the primary R4 endpoint is
not implementable as written; "disjoint" is uncheckable and unfalsifiable; and SC-01:43's ban on support
changes *after* test access has no referent, since no support is frozen *before* it. `SC_DR0_REVIEW.md:5`
("SC-01 defines all R0-R4 supports") is false.

### FC-3 (BLOCKER, gate arithmetic) — G3, the only falsification gate, passes null results

With `D = RMSE_N2 - RMSE_N3` and `L = RMSE_N3S - RMSE_N3`, the identity `(N3S - N2) = L - D` holds exactly.
G3's disjunct therefore fires whenever `1.96 * SE(N3S - N2) >= D - L`, i.e. whenever the CI half-width exceeds
the *unremoved* feature gain. `SE(N3S - N2)` is constrained by no gate and is inflated by the shuffle itself.

Under a true null (`L = 0`, features decorative), conditioned on G2 having genuinely passed, `n = 90`:
a fully decorative feature path passes G3 at rates up to **~0.73** in a plausible corner (`D` at G2's own 5%
floor, noisy metric). The defect is self-reinforcing — every source of imprecision an implementer controls
*increases* the pass rate. This inverts the charter's own statement of G3 ("feature shuffling removes a
material portion of the feature gain") and is the concrete path by which a null becomes a reported pass.

Compounding: the primary condition `L >= 0.5 D` is point-estimate-only while G0, G1, and G2 all require a CI.
The gate meant to be hardest to pass is the least protected. G3 is also the only disjunctive gate; disjunction
inflates type-I error while the conjunctive gates are conservative.

Until FC-3 is repaired, a `LOCK_COMPENSATION_CONDITIONED_SPLIT_MEKF_KALMANNET` outcome would be unsupportable.

### FC-4 (BLOCKER, gate arithmetic) — gate verdicts remain free after test access

- **Divergence is undefined.** G4's second conjunct is "N3 adds no divergence relative to N0", but no
  divergence criterion or threshold exists in SC-00, SC-01, SC-02, or the DR0 bundle. G4 is not decidable from
  frozen scope, and any threshold chosen once results exist is test-driven tuning.
- **Seed aggregation is specified only for G1 and G2.** G0, G3, and G4 have no rule for combining the three
  training seeds. G4's `<= 0.03` does not say whether it holds per seed, on the seed mean, or on any seed.
- **The bootstrap resampling unit is ambiguous.** The handoff implies 90 = 30 unique trajectories x 3 seeds
  pooled into one i.i.d. frame. If the 30 test trajectories are shared across seeds, the 90 rows are 30
  correlated clusters and an i.i.d. bootstrap understates the CI half-width by up to `sqrt(3) ~ 1.73x`,
  biasing G0, G1, G2 toward PASS and G3 toward its escape clause.
- **The "two of three seeds" rule has no defined statistic** (per-seed mean? median? fraction improved?
  per-seed CI?) and no pre-registered per-seed machine-record schema, so it is negotiable after test access.

### FC-5 (BLOCKER, traceability) — sealed claims are not traceable to the frozen contract

SC-00 freezes 13 red-path identifiers (`test_sc_*_red`). The sealed handoff cites 7 entirely different
identifiers. **The intersection is empty.** No manifest claim covers `test_sc_causal_prefix_invariance_red` or
`test_sc_deterministic_vs_residual_bias_separation_red`.

Missing red paths for falsifiable claims:
- **learned-compensator residual-bias retention** — `test_sc_gyro_oracle_retains_residual_bias_red` tests only
  the *oracle* target, which retains `b_r` by construction. Nothing falsifies the load-bearing claim that the
  **learned** N2/N3 compensator does not absorb `b_r` — precisely the failure FC-1 shows is forced;
- **intra-timestamp stage order** — prefix invariance constrains only the past/future boundary; an
  implementation that updates before propagating at equal timestamps passes all 13 tests yet violates SC-00;
- **right-error reset** — "the existing right-error reset" has no red path.

Several existing red paths are additionally **vacuous unless fixture conditions are stated**: the injection and
Jacobian tests require `q_hat != I` (at identity, left and right error coincide); the magnetometer inverse test
requires `b_m != 0`, anisotropic `A_m`, and `C_SmB != I` (each wrong ordering has a null case otherwise).

## Material findings (must be covered by the same repair round)

- **FC-6 (causality wording).** SC-00:11's "nor the correction produced at `t`" admits two incompatible
  readings, because SC-00 itself uses "correction" for the compensator output at line 21. Reading A
  (posterior-only) permits same-`t` `f_m -> G2`; Reading B forbids it and would make the entire N3/G2/G3
  architecture illegal. Two incompatible implementations of `test_sc_causal_prefix_invariance_red` follow. A
  frozen pre-registration must not require a charitable reader.
- **FC-7 (claim populations and seal coverage).** C3 counts 1350 but its predicate also covers N3 branch
  routing (+450 = 1800); C6 counts 270 but G1 also requires R1 and R2 populations (+180 = 450). Separately,
  the handoff is listed in `excluded_from_manifest`, so **every claim and population can change without
  altering the sealed digest**.
- **FC-8 (N3S single intervention).** SC-01 does not say whether the derangement permutation is shared across
  time indices; an independent per-index permutation destroys temporal feature coherence too — two bundled
  interventions, inflating `L` in the pass-friendly direction. Strata of size 1 make derangement impossible and
  silently degrade to identity, making N3S = N3 without record.
- **FC-9 (R4 disjointness is partial).** R4 is held out only over calibration-parameter support; nothing
  requires disjoint initial states, orbit phases, or noise streams, so R4 may reuse memorized training dynamics
  and the OOD framing of the primary endpoint is weakened.
- **FC-10 (contrast sign conventions).** G2 uses `(N3 - N2)`, G3 uses `D = N2 - N3`, and G1's CI contrast is
  never named. The `upper < 0` / `lower <= 0 <= upper` definitions themselves are consistent.
- **FC-11 (charter section 12).** No wall-time, CPU/GPU/RAM, parallelism, disk, or stop-condition record exists
  anywhere in the bundle, though 12 trainings plus ~3,150 replays plus 10,000-resample bootstraps certainly
  exceed 30 minutes. Prompt 05's required-output list has no carrier for it, so it will be structurally missed.
- **FC-12 (reuse pinning).** The study imports frozen code (`bench/estimators/mekf.py`, `bench/metrics/mekf.py`,
  `bench/tasks/generator/mekf_fusion_events.py`) that the manifest does not hash, against a dirty working tree,
  so the commit pin does not establish the content of the reused math.
- **FC-13 (ledger).** `DECISION_LEDGER.md` is byte-identical to its template and records no DR0 seal entry,
  against its own rule that every automatic transition record its artifact.

## What passed

The core estimator algebra is **correct** and consistent with the reused `bench/estimators/mekf.py`:

- `nu_m = z_tilde - h_m` and `H_m = [+[h_m]_x, 0]` — sign independently derived and confirmed;
- gyro inverse algebra and the no-double-count argument for `omega_tilde - b_hat_r`;
- magnetometer hard-iron -> soft-iron -> mount inverse ordering;
- `K = G1 H_m^T G2 in R^(6x3)`, `delta_x = K nu_m`, right injection consistent with the right-error convention;
- `rank([h_m]_x) = 2` with body-frame null direction `span{h_m}`; the weak-axis statement is **honest**, not
  overclaimed, and correctly limited to instantaneous observability.

Scope is minimal in size — nothing should be deleted. All five regimes and N0-N3S are load-bearing; C0/C1 are
justified diagnostic grounding, since charter section 11 makes a failed G0 terminal and a classical arm is what
distinguishes "no headroom exists" from "the error injection was too weak". All 8 secondary metrics are
zero-marginal-cost functions of gate-run outputs and should be kept. No excluded topic (SNN, SoW, attention,
Transformer, learned Q/R, uncertainty head, extra sensors, broad comparison, sweeps) is smuggled in; every hit
is inside an exclusion or a correctly-gated deferral. Governance, path ownership, checkpoint integrity, target
uniqueness, handoff freshness, and boundary isolation all pass.

## Boundary statement

This audit authorizes nothing outside the side study. Main AI-ADCS Phase 2, SpikeRA-KalmanNet, and frozen
Phase 0-1 evidence are untouched and unaffected. No numeric performance, covariance/NIS/NEES, observability,
novelty, or high-cost-sensor-equivalence claim is made or supported by this audit; no experiment has been run.
