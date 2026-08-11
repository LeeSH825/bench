# SC Implementation Audit — Addendum (Claude, second independent pass)

- Study: `side-gyro-mag-comp-v1`
- Stage: `IMPLEMENTATION_AUDIT`
- Audit target: `experiments/side_gyro_mag_comp/implementation/CHECKPOINT_MANIFEST.json`
- Sealed digest: `68a08285878fa484291697c69e93ea855afe51521d90cc86061bc362b90a90a5` — recomputed at start, twice mid-audit, and at end: **stable**; 28/28 member hashes match
- Relationship to `SC_IMPLEMENTATION_AUDIT.md`/`.json`: **additive. Nothing is overwritten, retracted, or weakened.**

## Concurrence

This pass ran independently and concurrently with the primary implementation audit. Both reached the
**same verdict — `FAIL_RED_PATH`, repairable — and the same root cause**: the sealed claims are not
falsifiable where the study actually computes. The recorded transition to `IMPLEMENTATION_REPAIR` with
`repair_round_by_stage.IMPLEMENTATION = 1` is the transition this pass would also have made, so no
corrective transition was required and none was made.

The primary audit ran 24 mutations; this pass ran 5. They agree everywhere they overlap.

## My independent mutation evidence

Applied to a disposable copy of `bench/` and `tests/` under the scratchpad. **No repository file was edited.**
Baseline on the copy: `42 passed`. One mutation at a time, reverted between each; restoring returns `42 passed`.

| # | Mutation | Sealed claim falsified | Suite |
|---|---|---|---|
| MUT-A | `study.py` unroll: left injection `Exp(δθ) ⊗ q` | IMPL-C1 right injection | **42 passed** |
| MUT-B | `study.py` unroll: `H[:, :3] = -_torch_skew(h)` | IMPL-C1 `H_m` sign | **42 passed** |
| MUT-C | `study.py` unroll: `innovation = h - mag_value` | IMPL-C1 innovation sign | **42 passed** |
| MUT-D | `model.py` FiLM bypass: `g1 = 2.0 * base_g1` | IMPL-C5 feature-off exact identity | **42 passed** |
| MUT-E | `model.py`: `self.mag_encoder = self.gyro_encoder` | IMPL-C5 separate encoders | **42 passed** |

MUT-A/B/C corroborate the primary audit and are addressed by its **R4** (train/eval operator equivalence),
which would catch all three.

## Two findings NOT covered by the primary audit or its round-1 evidence

I verified by direct inspection of `SC_IMPL_RED_PATH_EVIDENCE.json` and the `U-1…U-6` list that neither of
these appears anywhere: the recorded mutation set varied `FEATURE_DIM` (a different failure class), and the
strings `alias`, `mag_encoder` and `separate` occur in no recorded mutation.

### MUT-D — FiLM feature-off is not provably exact identity

`test_sc_film_feature_off_exact_equivalence_red` compares two **feature-off** runs against each other, so it
establishes feature-independence only — never `gamma = 1, beta = 0`. The implementation is correct
(`model.py` bypasses to `g1, g2 = base_g1, base_g2`), but scaling that bypass by `2.0` leaves the suite green.
This matters because exact feature-off identity is what makes the N2 vs N3 contrast in **G2** a clean
single-variable comparison; if the bypass silently differed from identity, G2 would confound the FiLM path
with a changed base gain.

### MUT-E — the two encoders are not provably separate

Sealed claim IMPL-C5 asserts "separate causal gyro and mag encoders", and the frozen scope requires
per-sensor encoders with branch-specific conditioning. Aliasing the two into one shared module leaves the
suite green, so a design that collapses the entire two-branch premise of the study would pass every test.

Both are appended to the counterproposal as **R12** and **R13**. Because `IMPLEMENTATION` has consumed its
only repair round, a gap left here cannot be repaired later and would close the study as `COMPLETE_BLOCKED`.

## Independently verified as correct

- `ν_m = z̃ − h_m`, `H_m = [[h_m]_x, 0]`, `K = G1 H_m^T G2` at `(6,3)`, right injection, propagation `ω̃ − b̂_r`,
  right-Jacobian reset with an explicit anti-identity guard.
- **CF-1 SATISFIED** (my DR0 blocking carry-forward). A real `MagEncoder` exists symmetric to `GyroEncoder`;
  `model.py` contains no calibration/truth/oracle symbol; encoder input is exactly (3-vector, `dt`, `valid`);
  `RuntimeSensorPacket` is a **closed five-field allowlist**; and `TrajectoryTruth`, `CalibrationTruth`,
  `OracleSidecar` and `SensorTrajectory` are rejected by object identity. The oracle inversion exists only in
  the generator/diagnostic namespace. Residual: the tertiary name denylist misses `soft_iron`, `hard_iron`,
  `m_true_N`, `b_r_true`, `scale_factor_inverse` — a backstop gap, not a live leak, already covered by R2.
- **CF-2 SATISFIED** (my DR0 blocking carry-forward). `freeze_train_normalization` draws only from
  `split.train_ids`, validates R0–R3 membership, hashes the constants, records `frozen_before_test`;
  selection is validation-only and never consults test/R4. Strengthened further by R1.
- **`G1` bias-coupling carry-forward ADDRESSED**: `base_g1[3:, :3] = 0.02·I` is nonzero and trainable, so
  `δb_r ≠ 0` at init and the residual-bias state genuinely receives measurement updates.
- **`m_N` collision carry-forward ADDRESSED**: `m_model_N_onboard` is enforced distinct from truth.
- Scope clean: no attention, Transformer, SNN, SoW, reliability gate, learned Q/R, uncertainty head, extra
  runtime sensor, closed-loop, FPGA, sweep, or broad KalmanNet-family comparison. Feature dim 8,
  branch-specific FiLM, gain `[6,3]`, exactly the authorized variants.
- No premature pilot: smoke only (4/regime, one seed, 140 records). I reproduced Codex's `42 passed` in its
  pinned `uv` environment.
- Frozen boundary intact: 4880-file fingerprint byte-identical to the pre-stage baseline;
  `.codex/config.toml` sha256 unchanged; the three reused `bench/` modules hash-pinned and unmodified.

## Governance

- **Concurrent Claude-role writer recurred.** `SC_IMPLEMENTATION_AUDIT.md/.json`,
  `SC_IMPL_RED_PATH_EVIDENCE.json`, `SC_IMPL_ROUND1_COUNTERPROPOSAL.json` and the state transition consuming
  `IMPLEMENTATION` repair round 1 were written by a concurrent writer during this audit. Verdict direction
  identical; no Codex-owned artifact altered; no frozen boundary touched; sealed digest never moved. The
  residual risk is again the repair budget, which R12/R13 close.
- **prompts/05 amended outside the SC-02 allowlist.** Verified: its mtime (21:06:34) is unique among twelve
  siblings at 18:48:26, inside the implementation window. Verified the change is **strictly tightening** —
  G0/G1/G2/G4 thresholds and all populations unchanged; only G3 rewritten to the reformulation I proved
  equivalent (`CI(T)>0 ⇔ L ≥ 0.5D`, 0 counterexamples in 200,000 trials), removing the accept-the-null
  disjunct and adding `INCONCLUSIVE_UNDERPOWERED`. It preceded any test access and was disclosed rather than
  concealed. Scientific risk: none. Requires a traceability record, not a repair round.

## Observations carried forward (not requirements)

- `G1[3:6,0:3]` initializes at `+0.02·I`, giving `δb̂ ∥ +δθ̂` where the classical MEKF correlation is
  antiparallel. SC-00 disclaims covariance semantics, so this is **not** a violation and no change is
  requested — recorded only so it is never misread as a covariance claim.
- `P` never enters `K` and gets no measurement update in the learned path. **No NIS/NEES/covariance-validity
  claim may be made.**
- The onboard magnetic reference differs from truth by 2.24%, imposing a ≈1.28° innovation-model floor
  **equally on all arms** — must be disclosed in the final report; it bounds absolute accuracy without
  biasing between-arm contrasts.
- The generator hardcodes identity mounting (`C_SgB ≡ C_BSm ≡ I`) and `C_SgB` is read by no code path, so
  SC-00's "nonidentity mounting" non-vacuity clause is met only by fixtures. Resolve or withdraw the clause.

No G0–G4 outcome is asserted. No experiment has been run.
