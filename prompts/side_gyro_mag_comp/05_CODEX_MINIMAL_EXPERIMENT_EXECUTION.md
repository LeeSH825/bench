# CODEX — Minimal Method-Lock Experiment

Proceed automatically when the validated implementation audit advances the state to ORACLE_HEADROOM.

Use one `sc_experiment_operator`. Read-only helpers may run in parallel, but no
other agent may alter configs, code, checkpoints, or result files.

## Frozen pilot

Smoke:

- train 4/regime
- validation 2/regime
- test 4/regime
- one training seed

Method-lock pilot:

- train 40/regime
- validation 10/regime
- test 30 paired trajectories/regime
- three training seeds

Primary endpoint:

- trajectory-level attitude geodesic RMSE on R4 combined OOD.

Required secondary metrics:

- gyro corrected-rate RMSE;
- gyro integrated increment error;
- mag corrected-vector angular error;
- residual gyro-bias RMSE;
- p95 attitude error;
- divergence count;
- magnetic-axis weak error;
- observable-plane error.

## Execution order and hard stops

1. unit tests
2. smoke
3. C0/C1/N0/N1 oracle-headroom experiment
4. evaluate G0
5. stop immediately if G0 fails
6. train/evaluate N2 learned compensation
7. evaluate G1
8. stop if G1 fails
9. train/evaluate N3 feature-conditioned path
10. evaluate G2
11. run N3S with same N3 checkpoint
12. evaluate G3
13. evaluate G4 nominal harmlessness
14. seal results for Claude

Do not change architecture, feature size, FiLM, loss weights, learning rate, epochs,
data ranges, or gates after test access. Do not add rescue experiments.

## Frozen gates

- G0: N1 improves over N0 by at least 10% on R3 primary metric and paired
  bootstrap 95% CI is below zero.
- G1: N2 improves sensor-level gyro and mag metrics and improves the primary
  filter metric over N0 by at least 5%, CI below zero, same direction in at least
  two of three seeds.
- G2: N3 improves over N2 by at least 5% on R4 primary metric, CI below zero,
  same direction in at least two of three seeds.
- G3: N3S loses at least half of N3's feature-related gain, or its N2-relative CI
  includes zero.
- G4: R0 nominal N3 penalty is at most 3%, with no added divergence.

## Required machine outputs

- per-trajectory CSV;
- aggregate JSON;
- paired comparison JSON;
- split/dataset manifest;
- training manifest per seed;
- gate decision JSON;
- exact command log;
- changed-path manifest;
- sealed Codex handoff.

Allowed decisions:

- `READY_FOR_EVIDENCE_AUDIT`
- `STOP_NO_COMPENSATION_HEADROOM`
- `REVISE_COMPENSATION_NETWORK`
- `REJECT_FEATURE_CONDITIONING`
- `FEATURE_SHORTCUT_OR_UNUSED`
- `FAIL_NOMINAL_HARMLESSNESS`
- `BLOCKED_EXECUTION_OR_INTEGRITY`
