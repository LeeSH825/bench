# Final Independent Audit (Claude)

- Study: `side-gyro-mag-comp-v1`
- Stage: `FINAL_AUDIT`
- Audit target (exactly one canonical bundle): `experiments/side_gyro_mag_comp/final/CHECKPOINT_MANIFEST.json`
- Sealed digest: `cb7b82943b6a0fa0d6d7e2740088f38228ecb1912f80c79954af9d586cff6ae7` — recomputed at start and end: **stable**; 9/9 member hashes match
- Source commit: `052d2f7217b964b1fa4e80bd643716b433780f08`
- Codex final decision: `BLOCKED_IMPLEMENTATION_OR_INTEGRITY` (treated as a claim, verified below)

## Verdict

**`PASS_FINAL_BLOCKED_WITH_VALID_EVIDENCE`**

Single decisive predicate:

> The sealed final bundle reports a blocked outcome that follows the frozen decision mapping, asserts no
> unsupported result, and represents every number and every audit verdict truthfully.

**This predicate is TRUE**, subject to two recorded clarifications below that do not change the decision.

## Why BLOCKED is the correct mapping

No experiment was executed. I verified this directly: there is **no** per-trajectory CSV, aggregate JSON,
paired-comparison JSON, or gate-decision record anywhere under `experiments/side_gyro_mag_comp/`
(search for `*per_trajectory*`, `*paired*`, `*GATE*`, `*aggregate*` returns **0** files), and no
`results/` or oracle-headroom directory exists.

Because no gate ran, none of charter §11's gate-outcome mappings (`STOP_NO_COMPENSATION_HEADROOM`,
`REVISE_COMPENSATION_NETWORK`, `LOCK_COMPENSATION_ONLY_REJECT_FEATURE_PATH`, `REVISE_FEATURE_INTERFACE`,
`LOCK_COMPENSATION_CONDITIONED_SPLIT_MEKF_KALMANNET`) is reachable — each requires an evaluated G0–G4
predicate. The only applicable mapping is the blocked terminal. **The mapping is applied correctly.**

The chain that produced it is legitimate and each link is independently verified:

1. **DR0 round 0** — I found the contract inadmissible and issued a repairable FAIL.
2. **DR0 repair (round 1 of 1)** — Codex resealed. I re-audited and issued `PASS_DR0_AUTONOMOUS_ADVANCE`
   after independently recomputing the R4 disjointness margins, the `A_g`/`A_m` conditioning consistency,
   the 5σ identifiability separation, and the G3 reformulation equivalence (0 counterexamples in 200,000 trials).
3. **Implementation** — sealed at `68a08285…`. I issued `FAIL_RED_PATH`: five mutations that each falsify a
   sealed claim all survived with **42/42 tests passing**.
4. **Implementation repair (round 1 of 1)** — Codex opened the round and stopped **before editing**, because
   the mandatory counterproposal required a frozen weak-axis/observable-plane membership threshold that exists
   in no canonical artifact. Inventing one would have created evaluation semantics after the DR0 freeze.

Stopping there was the **correct** action. Prompt 03 directs the implementer to "stop with
`BLOCKED_CONTRACT_GAP` instead of inventing an interface", and charter §10 makes a second failure terminal
rather than negotiable. Fabricating a threshold to force progress would have been the serious error.

## Prompt-13 verification checklist

| Requirement | Result | Evidence |
|---|---|---|
| Final decision follows the frozen mapping from G0–G4 and audit outcomes | **PASS** | No gate ran; only the blocked terminal is reachable |
| Every number resolves to machine evidence with exact scope | **PASS** | 42 tests and the 140-record smoke (7 variants × 20 test trajectories, 4/regime R0–R4, seed 31001, whole-trajectory `attitude_geodesic_rmse_rad`) — I reproduced both independently |
| No failed predicate presented as PASS | **PASS** | All G0–G4 are `NOT_RUN`/`UNKNOWN`; `code_and_information_boundary` and `governance_and_integrity` are reported `FAIL` |
| No external blocker misrepresented as a scientific result | **PASS** | `scientific_rejection: false`, `decision_class: TERMINAL_BLOCKED_PRE_EXPERIMENT`, and the report states plainly it "is a governance/contract blocker, not a negative scientific result" |
| No NIS/NEES/covariance claim from the direct-gain neural path | **PASS** | `physical_covariance_claim: false`; I confirmed in code that `P` never enters `K = G1 H^T G2` and receives no measurement update in the learned path |
| Single-magnetic-vector weak direction disclosed | **PASS** | Rank-2 sensitivity and the unobservable body-frame direction parallel to `h_m` are stated; I verified `rank([h_m]_x) = 2` with null space `span{h_m}` |
| Main Phase 2 and SpikeRA-KalmanNet untouched | **PASS** | 4880-file fingerprint byte-identical to the pre-study baseline; `.codex/config.toml` sha256 unchanged at `315ec7d2…` |
| Final artifact index resolves; populations non-empty | **PASS** | 19 indexed paths resolve; the only two absent were this audit's own files, now written |

## Two recorded clarifications (do not change the decision)

1. **Count of unfalsifiable invariants.** `FINAL_RESULT.md` states 18. An additive independent pass recorded
   in `audits/SC_IMPLEMENTATION_AUDIT_ADDENDUM.md` found **two more** that no recorded mutation covered —
   FiLM feature-off exact identity (`g1 = 2.0 · base_g1` survives) and encoder aliasing
   (`self.mag_encoder = self.gyro_encoder` survives). These were appended to the counterproposal as **R12**
   and **R13**. The correct figure is **20**, and the unapplied counterproposal is **R1–R13**, not R1–R11.
   This makes the blocker slightly larger, never smaller.
2. **Nature of the CF-1/CF-2 finding.** `FINAL_RESULT.md` line 30 says CF-1 and CF-2 were "not fully
   satisfied", which could be misread as an actual leak. **No leak occurred.** I verified in code that the
   deployable namespace contains no calibration, truth or oracle symbol; `RuntimeSensorPacket` is a closed
   five-field allowlist; truth/calibration/oracle containers are rejected by object identity; and
   normalization is computed only from training-split IDs, hashed and frozen before any test read. The
   deficiency is **falsifiability** — those controls can be removed without any test going red. That is a
   real and sufficient blocker, but it is a coverage failure, not a contamination event.

## Verified-correct findings worth preserving for a successor study

The blocked outcome is procedural; a substantial amount of verified work survives it.

- The **mathematics is correct**, independently derived and cross-checked against the repository's frozen
  conventions: `ν_m = z̃_m^B − h_m` with `H_m = [[h_m]_x, 0_3×3]` (matching `mekf.py::body_vector_jacobian`);
  `K = G1 H_m^T G2` at `(6,3)`; right injection `q+ = q− ⊗ Exp(δθ)`; propagation `ω̃ − b̂_r`; right-Jacobian
  covariance reset; and the magnetometer hard-iron → inverse-soft-iron → mounting order as the exact inverse
  of the forward model.
- The **deterministic/residual-bias split does not double-count**: the compensation target is `ω_true + b_r`,
  and `b̂_r` is subtracted exactly once in propagation.
- The **repaired identifiability treatment is sound**: `(A_g, c_g)` drawn once per regime and shared, `b_r(0) ~
  N(0,(2e-5)²I)` zero-mean random walk, minimum offset exceeding 5σ (verified: `2e-4 > 1e-4`), a three-axis
  excitation certificate, and an explicit statement of the gauge symmetry
  `(c_g, b_r) → (c_g + A_g C_SgB v, b_r − v)`.
- The **repaired G3 predicate is a genuine falsification test**: `CI(T) > 0` with
  `T = RMSE_N3S − 0.5·RMSE_N2 − 0.5·RMSE_N3` is exactly `L ≥ 0.5·D` (0 counterexamples in 200,000 trials),
  the accept-the-null disjunct is removed, and an inert shuffle correctly **fails** the gate.
- The **clustered bootstrap is now valid**: resampling unit is the test `trajectory_id` with the seed
  dimension nested, closing a pseudo-replication channel that would otherwise have inflated the one-sided
  error of every gate CI from 2.5% to roughly 12.5%.
- `G1`'s `(b_r, θ)` coupling block is initialized nonzero and trainable, so the residual-bias state genuinely
  receives measurement updates rather than being silently pinned at zero.
- **Scope stayed clean throughout**: no attention, Transformer, SNN, SoW, reliability gate, learned Q/R,
  uncertainty head, extra runtime sensor, closed-loop, FPGA, automated sweep, or broad KalmanNet-family
  comparison. Feature dimension 8, branch-specific FiLM, gain `[6,3]`, exactly the authorized variants.

## Governance findings (recorded; they do not soften the outcome)

- **Concurrent Claude-role writer.** At both DR0 and implementation, Claude-owned audit artifacts, handoffs
  and state transitions — including consumption of both repair rounds — were written by a concurrent writer
  during my audits. In both cases the **verdict direction was identical**, so no corrective transition was
  needed. The material risk was to the repair budget: each round was committed to a counterproposal missing
  findings from my pass. I closed that additively both times (CP-11/CP-12 at DR0, R12/R13 at implementation)
  without overwriting any existing record. No Codex-owned artifact was ever altered by a Claude-role writer,
  no sealed digest moved, and no frozen boundary was touched. **This is the principal process defect of the
  run and should be fixed before any successor study: exactly one Claude auditor instance must hold the
  audit-write and state-transition role.**
- **`prompts/05` amended outside the SC-02 allowlist** (mtime 21:06:34, unique among twelve siblings at
  18:48:26). I verified the change is **strictly tightening**: G0, G1, G2 and G4 thresholds and all
  populations are unchanged, and only G3 was rewritten to the equivalent predicate above, additionally
  forbidding the superseded disjunct and adding an `INCONCLUSIVE_UNDERPOWERED` terminal. It preceded any test
  access and was disclosed rather than concealed. **Scientific risk: none.** It still lacks an authorizing
  control-amendment record, which a successor study should require.
- **`DECISION_LEDGER.md` is incomplete** — it has no row for the DR0 repair seal, the
  `PASS_DR0_AUTONOMOUS_ADVANCE` re-audit, the implementation seal, or this finalization.
- **`sealed_claims_digest`** in state still resolves to the DR0 claims digest; the implementation and final
  claim sets are not covered by a state-level digest.

## Claim-by-claim final status

No experiment was run, so claims 6–10 have no evidential outcome. Nothing below asserts a G0–G4 result.

| # | Claim | Final status |
|---|---|---|
| 1 | Mathematical and frame consistency | **VERIFIED CORRECT** in derivation and code; **not falsifiably protected** (left injection and both sign flips survive the suite) |
| 2 | Deterministic compensation separated from residual bias; split identifiable | **VERIFIED CORRECT** — no double-correction; identifiability properly gauge-fixed after DR0 repair |
| 3 | Causal 8D branch-specific feature routing | **PARTIAL** — 8D enforced, branch isolation genuinely red-path covered, causality enforced; feature-off exact identity and encoder separation **not falsifiable** (R12/R13) |
| 4 | No truth/oracle/future/test leakage | **NO LEAK FOUND** — closed runtime allowlist, type-identity rejection of truth/calibration/oracle, train-split-only hashed normalization, validation-only selection. **Enforcement not falsifiable** (R1/R2) |
| 5 | Whole-trajectory split and paired realizations | **CORRECT IN CODE** — disjointness, R4 test-only, per-variant realization pairing; disjointness guard has **no genuine red path** |
| 6 | G0 oracle compensation headroom | **UNKNOWN — NOT RUN** |
| 7 | G1 learned compensation benefit | **UNKNOWN — NOT RUN** |
| 8 | G2 incremental feature benefit | **UNKNOWN — NOT RUN** |
| 9 | G3 feature-shuffle falsification validity | **DESIGN VERIFIED as a genuine red path** (fails correctly when shuffling is inert); **never executed** |
| 10 | G4 nominal harmlessness + honest observability | Harmlessness **UNKNOWN — NOT RUN**. Observability reporting is **honest**: rank-2 sensitivity and the weak body-frame direction are disclosed, with no observability or high-cost-sensor equivalence claim |

## Claims that must never be made from this run

Compensation headroom; learned-correction efficacy; incremental value of compensation features;
shuffle falsification; nominal harmlessness; any performance, efficiency, novelty, physical-covariance,
NIS/NEES, calibrated-uncertainty, flight, closed-loop, energy, or generality claim.

Additionally, the onboard magnetic reference differs from the truth reference by 2.24%, imposing a ≈1.28°
innovation-model floor applied **equally to every arm**. It bounds absolute accuracy for all variants without
biasing between-arm contrasts, and must be disclosed alongside any future absolute number.

## Resume commands

There is **no resume command for this run**: both repair rounds are consumed and the R8 contract gap is
terminal by charter §10. Retroactively defining the weak-axis population now and resuming would be exactly
the post-hoc semantics change the freeze exists to prevent.

A successor study must be newly authorized and must, in order:

1. Freeze in the canonical math/gate contract the **exact weak-axis and observable-plane membership formula
   and threshold**, plus the metric producers for every G1/G4 sensor-level predicate.
2. Carry forward counterproposal items **R1–R13** as pre-registered requirements — in particular a
   machine-recorded mutation matrix proving every mandatory red path goes red under its own counterexample.
3. Resolve or withdraw SC-00's "nonidentity mounting" clause, which the generator currently makes vacuous by
   hardcoding `C_SgB ≡ C_BSm ≡ I`.
4. Independently audit that contract **before any test access**, then restart the implementation cycle.
5. Assign the Claude audit-write and state-transition role to exactly one instance.

Verification commands used in this audit:

```
python3 agent_system/side_gyro_mag_comp/scripts/validate_stage_state.py agent_system/side_gyro_mag_comp/state/STAGE_STATE.json
python3 agent_system/side_gyro_mag_comp/scripts/validate_handoff.py experiments/side_gyro_mag_comp/handoffs/codex/CODEX_FINAL_TO_CLAUDE.json
sha256sum experiments/side_gyro_mag_comp/final/CHECKPOINT_MANIFEST.json   # cb7b8294...6ae7
uv run --no-project --python 3.12 --index https://download.pytorch.org/whl/cpu \
  --with 'numpy==2.1.3' --with 'scipy==1.14.1' --with 'torch==2.5.1' --with 'pyyaml==6.0.2' \
  --with 'pytest==8.3.4' python3 -m pytest -q tests/side_gyro_mag_comp_v1        # 42 passed
```

## Boundary statement

Main AI-ADCS Phase 2, SpikeRA-KalmanNet, and frozen Phase 0–1 evidence are **unmodified and unauthorized** by
this study: a 4880-file fingerprint over all eight frozen directories is byte-identical to the pre-study
baseline, and `.codex/config.toml` still hashes to `315ec7d2282939ea0344b6de5ec5dc2c6dbab3bbee91fa3d1e63912b29a2c20d`.
Claude edited no implementation, training, dataset, configuration, checkpoint, threshold, machine-result, or
Codex report file at any point.

Terminal state: `COMPLETE_BLOCKED`, `method_decision = BLOCKED_IMPLEMENTATION_OR_INTEGRITY`, `next_actor = NONE`.
