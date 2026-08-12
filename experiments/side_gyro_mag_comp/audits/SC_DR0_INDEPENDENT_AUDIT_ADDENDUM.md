# SC-DR0 Independent Audit — Addendum (Claude, second independent pass)

- Study: `side-gyro-mag-comp-v1`
- Stage: `DR0_INDEPENDENT_AUDIT`
- Audit target (single canonical bundle): `experiments/side_gyro_mag_comp/design_review/CHECKPOINT_MANIFEST.json`
- Sealed digest: `298feba22320a5e1abd617455d4d7cc342f539229d443c2cda3577326788f730`
- Digest recomputed at start, twice mid-audit, and at end: **stable, unchanged**
- Relationship to `SC_DR0_INDEPENDENT_AUDIT.md`/`.json`: **additive. Nothing in the primary audit is overwritten, retracted, or weakened.**

## Why this addendum exists

This audit pass was conducted independently and concurrently with the primary
`SC_DR0_INDEPENDENT_AUDIT`. Both passes reached the same direction — **FAIL, repairable, one DR0
repair round, implementation not authorized** — so the recorded state transition
(`DR0_REPAIR`, `next_actor CODEX`, `repair_round_by_stage.DR0 = 1`) is the transition this pass
would also have made. No corrective transition is required and none was made.

Two blockers found by this pass are **not covered anywhere** in the primary audit or in the
sealed counterproposal CP-1…CP-10. Both are information-boundary (leakage) defects, the most
severe class in the charter. Because DR0 has now consumed its only repair round, a defect omitted
from the counterproposal will survive into implementation and can then only be resolved by closing
the study as `COMPLETE_BLOCKED`. They are therefore recorded here and appended to the Claude→Codex
handoff as `CP-11` and `CP-12`.

## Governance incident (recorded, not adjudicated)

Claude-owned artifacts (`audits/SC_DR0_INDEPENDENT_AUDIT.md` at 20:07:41,
`audits/SC_DR0_INDEPENDENT_AUDIT.json` at 20:08:39,
`handoffs/claude/CLAUDE_TO_CODEX.json` at 20:09:44), the `DECISION_LEDGER.md` rows SC-D001/SC-D002,
and the `STAGE_STATE.json` transition to `DR0_REPAIR` consuming
`repair_round_by_stage.DR0 = 1` were written by a concurrent writer during this audit pass.

Assessment:
- **No frozen boundary was violated.** Re-verified: the 4880-file fingerprint over Phase 0–1,
  main Phase 2 and `phase2_reconfirmation` (SpikeRA-KalmanNet) is byte-identical to the pre-stage
  baseline, and `.codex/config.toml` still hashes to `315ec7d2282939ea0344b6de5ec5dc2c6dbab3bbee91fa3d1e63912b29a2c20d`.
- **No Codex-owned artifact was altered by any Claude-role writer**, and the sealed DR0 checkpoint
  digest never moved.
- **The substantive verdict is unaffected** — both passes independently concluded FAIL/repairable.
- **The residual risk is the repair budget**: the single DR0 repair round was committed to a
  counterproposal that omitted the two leakage blockers below. `CP-11`/`CP-12` correct that before
  Codex begins the repair.
- This is a charter §6 ("one writer and four eyes") concurrency exception and is reported to the
  user in the final report rather than escalated mid-workflow, per the no-mid-workflow-escalation rule.

## Additional confirmed blockers

### I1 — The deployable magnetometer path is defined by oracle calibration parameters, and no magnetometer encoder exists

SC-00 treats the two sensors asymmetrically. The gyro side is correct:

- line 19 — `The causal encoder returns (omega_tilde_g^B, f_g)` — a deployable encoder is defined;
- line 21 — `omega_target^B = C_BSg A_g^-1 (y_g^Sg - c_g^Sg)` — explicitly a **target**, so its use of
  true `A_g, c_g` is a legitimate training label.

The magnetometer side is not:

- line 29 — `z_tilde_m^B = C_BSm A_m^-1 (y_m^Sm - b_m^Sm)` — **no `target`/`oracle` qualifier**;
- line 33 — `nu_m = z_tilde_m^B - h_m` — the *same symbol* feeds the **deployable** innovation;
- line 39 — `G2 = FiLM_2(G2^0; f_m)` — **`f_m` is consumed but no magnetometer encoder ever produces it.**

`A_m`, `b_m` and `C_BSm` are precisely the "calibration parameters" that SC-00 line 45 forbids in the
deployable namespace. As frozen, the contract authorises a deployable estimator plumbed with true
calibration, and `test_sc_deployable_namespace_leakage_rejected_red` would be written to permit it.
There is also no defined producer for `f_m`, so the magnetometer half of the study's central
mechanism has no specified interface at all.

This defeats claim 4 (no oracle leakage into any deployable runtime path) and leaves claim 3
(branch-specific causal feature routing) only half specified. The primary audit's CP-5 and CP-6
touch `f_m` only in the context of causal ordering and sign gauge; neither requires a magnetometer
encoder nor marks `z_tilde_m^B` non-deployable.

### I4 — The train/test firewall is scoped by regime, not by split

SC-01 line 13: *"Training/validation/normalization use R0-R3 only."* — this restricts by **regime**.
SC-01 line 15 declares *"test 30 per R0-R4"*, so R0–R3 **test splits exist**. Only R4 is protected
from "training, early stopping, normalization, thresholds, or selection".

Therefore **R0–R3 test-split trajectories may lawfully feed normalization constants, early stopping
and checkpoint selection** — and those regimes are gate-bearing: G0 is scored on R3, G1's sensor
metrics on R1 and R2, G4 on R0. This is a live contamination channel through the very data that
decides three of the five gates, and it is entirely uncovered by CP-1…CP-10 (CP-8 addresses only
R4 stream disjointness).

SC-00's "train-frozen normalization constants" does not close the hole: that phrase is a
runtime-input adjective, not a computation-scope definition, so the two documents do not combine to
forbid it.

## Minimum counterproposal (additive; two clauses, text-only)

Appended to the existing handoff as `CP-11` and `CP-12`. Both are pure text clarifications of what
the contract already intends. **No threshold, gate value, model family, feature dimension,
conditioning mechanism, regime, population, split size, seed, or endpoint changes. No experiment,
architecture, sensor, or comparison is added.**

**CP-11 — SC-00: symmetric deployable encoder interface.**
Rename line 29 to `z_m^{B,oracle} = C_BSm A_m^{-1}(y_m^Sm − b_m^Sm)` and mark it, together with
`omega_target^B`, as a **label/diagnostic definition only**. Define the deployable pair explicitly and
symmetrically:

`(omega_tilde_g^B, f_g) = GyroEncoder(y_g^Sg, t; theta)` and `(z_tilde_m^B, f_m) = MagEncoder(y_m^Sm, t; theta)`

and state that `(A_g, c_g^Sg, C_SgB, A_m, b_m^Sm, C_SmB)` are **non-deployable in all of N0–N3S**.
Extend `test_sc_deployable_namespace_leakage_rejected_red` to reject each of these six symbols in the
deployable signature. (This also supplies the missing producer for `f_m`, without which the
magnetometer branch of N3 is unimplementable.)

**CP-12 — SC-01: split-scoped firewall.**
Replace *"Training/validation/normalization use R0-R3 only."* with: training, early stopping,
checkpoint selection, threshold setting and all normalization constants are computed **only from
R0–R3 training-split trajectory IDs**; **no test-split trajectory of any regime R0–R4** may influence
any of them; normalization constants are hashed and frozen before the first test evaluation, and the
hash is recorded in the per-seed training manifest. Extend `test_pairing_split_firewall_red` to fail
when any test-split trajectory ID of any regime contributes to a normalization constant, an
early-stopping decision, or a checkpoint selection.

## Re-audit contract for these two items

On reseal this pass will additionally verify:

1. SC-00 defines a `MagEncoder` producing `(z_tilde_m^B, f_m)` from causal sensor-frame input only,
   and `z_m^{B,oracle}` is marked non-deployable together with all six calibration symbols;
2. the deployable-namespace red path actually rejects `A_m`, `b_m`, `C_BSm`, `A_g`, `c_g`, `C_SgB`;
3. SC-01's firewall is scoped by **split**, not regime, and names every one of training, early
   stopping, checkpoint selection, threshold setting and normalization;
4. the pairing/split red path fails on a test-split contribution to normalization or selection;
5. a new stable checkpoint digest, recomputed before and after re-audit.

## Per-claim status from this pass (DR0, contract admissibility only)

| # | Claim | Status |
|---|---|---|
| 1 | Math/frame consistency | Core algebra **independently verified correct**; FAIL on packet frame labels |
| 2 | Compensation vs residual bias separated | No double-correction **verified**; FAIL on identifiability (gauge degeneracy) |
| 3 | Causal 8D branch-specific routing | Gyro branch OK; **FAIL — no magnetometer encoder, `f_m` unproduced (I1)** |
| 4 | No truth/oracle/future/test leakage | **FAIL — two uncovered channels (I1, I4)** |
| 5 | Whole-trajectory split, paired realizations | Disjointness OK; pairing **UNKNOWN** — no data-generation seed declared |
| 6 | G0 oracle headroom | Pre-registration only; gate well-formed; exposed to I4 |
| 7 | G1 learned compensation benefit | Pre-registration only; exposed to I4 and the identifiability floor |
| 8 | G2 incremental feature benefit | Pre-registration only; exposed to bootstrap-unit ambiguity |
| 9 | G3 shuffle falsification validity | Design **verified a genuine red path** (fails correctly when shuffling is inert) |
| 10 | G4 harmlessness + honest observability | Observability disclosure **correct and honest**; FAIL — "divergence" undefined |

No G0–G4 outcome is asserted. No experiment has been run. No implementation is authorized.

## Independently verified as correct (not adopted from Codex prose)

1. `nu_m = z_tilde − h_m` ⇒ `H_m = [[h_m]_x, 0_3×3]`, matching `bench/estimators/mekf.py::body_vector_jacobian`
   (`result[:, :3] = skew(prediction)`, residual `z − h`). No sign or transpose discrepancy.
2. `C_BSg A_g^{-1}(y_g − c_g) = omega_true + b_r + noise`; propagation `omega_tilde − b_hat_r` matches
   `mekf.py` error dynamics `F[:3,3:] = −I`. No double-correction in the algebra.
3. Magnetometer inverse order is the exact inverse of the stated forward model.
4. `G1(6×6)·H_m^T(6×3)·G2(3×3) = K(6×3)`; right injection matches `mekf.py::inject_error_state`, not left.
5. `rank([h_m]_x) = 2`, null space `span{h_m}`; the weak direction is the body-frame attitude error
   parallel to `h_m^B`. Correctly and honestly disclosed.
6. SC-00 explicitly denies physical `P/Q/R/S^{-1}` status for `G1, G2, K` and disclaims NIS/NEES validity.
7. G3 `L ≥ 0.5·D` ⇔ `RMSE_N3S ≥ (RMSE_N2 + RMSE_N3)/2`; if shuffling is inert the gate correctly FAILS.
8. No charter §9 hard exclusion is violated or smuggled in; no proposed experiment is unable to move G0–G4.
9. Frozen boundaries intact: 4880-file fingerprint unchanged; `.codex/config.toml` sha256 unchanged.
