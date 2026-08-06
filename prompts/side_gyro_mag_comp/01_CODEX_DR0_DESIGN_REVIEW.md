# CODEX — SC-DR0 Minimal Design Review

Repository root: `/home/dss-pc-05/bench`

## Objective

Decide whether the following method is mathematically coherent and minimally testable:

- causal gyro compensation -> corrected gyro + 8D gyro feature;
- causal magnetometer compensation -> corrected mag + 8D mag feature;
- corrected gyro -> existing right-error MEKF propagation;
- corrected mag -> magnetometer innovation;
- gyro feature -> Split-KalmanNet prior-side FiLM;
- mag feature -> Split-KalmanNet measurement-side FiLM;
- neural gain -> 6D local attitude/residual-bias correction.

Do not implement neural code in this stage.

## Mandatory setup

1. Read `agent_system/side_gyro_mag_comp/AUTONOMOUS_DUAL_AGENT_CHARTER.md`.
2. Validate `agent_system/side_gyro_mag_comp/state/STAGE_STATE.json`.
3. Read the current Phase 0–1 master, updated exit review, source index, and numeric catalog.
4. Preserve the current main Phase 2 boundary.
5. Spawn in parallel:
   - `sc_repo_navigator`
   - `sc_math_guard`
   - `sc_lean_scope_guard`
6. Wait for all three before synthesis.

## Required decisions

Fix only:

- state and quaternion/error convention;
- deterministic compensation versus residual bias state;
- gyro and magnetometer equations, frames, units, dimensions, signs;
- causal asynchronous event order;
- gain dimension and multiplicative injection;
- exact runtime-visible and forbidden inputs;
- feature role as gain conditioning, not an independent measurement or physical covariance;
- single-magnetic-vector weak-axis limitation;
- minimum regimes R0-R4;
- minimum models C0, C1, N0, N1, N2, N3, N3S;
- primary endpoint and G0-G4 gate predicates.

Hard exclusions remain frozen.

## Outputs

Write only:

- `docs/research/side_gyro_mag_comp/SC_00_MATH_AND_INFORMATION_CONTRACT.md`
- `docs/research/side_gyro_mag_comp/SC_01_MINIMAL_HYPOTHESIS_AND_GATE_CONTRACT.md`
- `docs/research/side_gyro_mag_comp/SC_02_REPOSITORY_REUSE_MAP.md`
- `experiments/side_gyro_mag_comp/design_review/SC_DR0_REVIEW.md`
- `experiments/side_gyro_mag_comp/design_review/SC_DR0_DECISION.json`
- `experiments/side_gyro_mag_comp/handoffs/codex/CODEX_TO_CLAUDE.json`
- exact changed-path and command manifests

Every PASS claim must name a red-path test to be implemented later and a non-empty
target population. Use `TARGET_NOT_FOUND`, `AMBIGUOUS`, or `UNKNOWN` instead of
inventing certainty.

Allowed DR0 decisions:

- `GO_TO_INDEPENDENT_DR0_AUDIT`
- `REVISE_MATH_CONTRACT`
- `STOP_NO_COHERENT_METHOD`

Seal one checkpoint and run `validate_handoff.py`.
