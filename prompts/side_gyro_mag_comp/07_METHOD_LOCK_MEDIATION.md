# Automatic Method-Lock Synthesis

Run automatically after the Claude evidence audit. This stage produces a deterministic final recommendation from the preregistered gates; it does not ask for human authorization.

Read:

- approved DR0 contracts;
- sealed Codex implementation and experiment handoffs;
- Claude DR0, implementation, and evidence audit JSON;
- stage state and decision ledger;
- machine per-trajectory and paired-comparison records.

Do not average conflicting verdicts. Separate them by axis:

- mathematical validity;
- code and information boundary;
- statistical evidence;
- scope proportionality;
- governance/integrity.

For each axis, state `PASS`, `FAIL`, or `UNKNOWN`, cite the machine/audit artifact, and name the decisive predicate.

Apply the frozen automatic decision mapping:

- G0 fails -> `STOP_NO_COMPENSATION_HEADROOM`
- G0 passes, G1 fails -> `REVISE_COMPENSATION_NETWORK`
- G0-G1 pass, G2 fails -> `LOCK_COMPENSATION_ONLY_REJECT_FEATURE_PATH`
- G2 passes, G3 or G4 fails -> `REVISE_FEATURE_INTERFACE` unless evidence attributes failure to the compensator
- G0-G4 and all audits pass -> `LOCK_COMPENSATION_CONDITIONED_SPLIT_MEKF_KALMANNET`
- integrity/math/data leakage invalidates evidence -> `BLOCKED_IMPLEMENTATION_OR_INTEGRITY`
- non-research external execution failure -> `BLOCKED_EXTERNAL_EXECUTION`

Write:

- `experiments/side_gyro_mag_comp/final/FINAL_RESULT.md`
- `experiments/side_gyro_mag_comp/final/FINAL_DECISION.json`
- `experiments/side_gyro_mag_comp/final/FINAL_ARTIFACT_INDEX.json`
- a sealed Codex-to-Claude final handoff

The final report must include:

- the automatic decision and decisive predicates;
- what was actually implemented and run;
- exact data/split/model/window/metric/seed scope for every number;
- what is fixed now and what remains unfixed;
- gyro+single-mag observability limits;
- blockers and minimum next action where applicable;
- no proposal for attention, Transformer, uncertainty, or broad comparison as part of the current decision.
