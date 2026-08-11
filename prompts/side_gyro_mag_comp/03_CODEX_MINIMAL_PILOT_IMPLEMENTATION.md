# CODEX — Minimal Pilot Implementation

Proceed automatically when the validated DR0 audit verdict advances the state to IMPLEMENTATION.

## Coordination

Spawn `sc_repo_navigator` and `sc_math_guard` read-only in parallel.
After they return, activate exactly one writer: `sc_impl_worker`.
No second implementation worker may edit concurrently.
Before handoff, run `sc_local_evidence_reviewer`.

## Implement only

- new independent family `side_gyro_mag_comp_v1`;
- regimes:
  - R0 nominal
  - R1 gyro constant bias + diagonal scale
  - R2 mag hard iron + soft iron
  - R3 combined
  - R4 held-out combined parameters;
- separate causal gyro and mag encoders;
- corrected 3-vector + fixed 8D feature per encoder;
- approved branch-specific FiLM;
- variants C0, C1, N0, N1, N2, N3;
- N3S as evaluation-only feature association shuffle using the N3 checkpoint;
- existing right-error MEKF conventions and weak-axis metric.

## Required red-path tests

- identity compensation;
- noise-free oracle exact correction;
- wrong frame/sign fixture;
- gain shape not equal to 6x3;
- q/-q metric invariance;
- future-sample injection rejection;
- truth/oracle leakage rejection;
- duplicate or overlapping trajectory split rejection;
- different raw realization across compared variants rejection;
- feature-off equivalence;
- shuffled-feature path changes only association;
- nominal no-op;
- weak-axis metric non-empty population.

Do not implement attention, Transformer, uncertainty head, learned Q/R, temperature,
outlier, extra sensor, closed-loop, FPGA, or broad comparisons.

## Stop conditions

Stop with `BLOCKED_CONTRACT_GAP` instead of inventing an interface.
Do not execute the full pilot in this stage. Unit and tiny smoke only.

## Outputs

- code and tests in authorized new/touched paths;
- unit/smoke machine evidence;
- implementation report;
- exact command and changed-path manifests;
- sealed `CODEX_TO_CLAUDE.json`.

Allowed decisions:

- `READY_FOR_IMPLEMENTATION_AUDIT`
- `BLOCKED_CONTRACT_GAP`
- `BLOCKED_IMPLEMENTATION`
- `FAIL_RED_PATH_NOT_PROVEN`
