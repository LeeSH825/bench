# AI-ADCS Autonomous Dual-Agent Charter

## 1. Scope

This charter governs only the side study:

> Gyro/Magnetometer compensation produces corrected values and compact features;
> those features condition a right-error MEKF-aware Split-KalmanNet gain network.

It does not authorize or modify the main AI-ADCS Phase 2, SpikeRA-KalmanNet, or frozen Phase 0-1 evidence.

## 2. Delegated operating mode

The user delegates all intermediate stage transitions to the preregistered state machine and independent audits.

```text
automation_mode = AUTONOMOUS_UNTIL_FINAL
human_review_mode = FINAL_ONLY
```

No agent asks for human authorization between DR0, implementation, audit, experiment, repair, or method-lock synthesis.

An unresolved failure is not overridden. It becomes a final blocked or rejected result.

## 3. Value order

1. Mathematical and scientific correctness
2. Reproducible evidence
3. Time and compute efficiency
4. Implementation convenience
5. Token efficiency
6. Interface polish

## 4. Single sources of truth

| Subject | Canonical artifact |
|---|---|
| Current autonomous stage | `agent_system/side_gyro_mag_comp/state/STAGE_STATE.json` |
| Side-study decisions | `agent_system/side_gyro_mag_comp/state/DECISION_LEDGER.md` |
| Deferred work | `agent_system/side_gyro_mag_comp/state/DEFERRED_REGISTER.md` |
| Math/information contract | `docs/research/side_gyro_mag_comp/SC_00_*` |
| Gate contract | `docs/research/side_gyro_mag_comp/SC_01_*` |
| Numeric results | machine JSON/CSV under `experiments/side_gyro_mag_comp/results/` |
| Independent verdicts | Claude audit JSON under `experiments/side_gyro_mag_comp/audits/` |
| Final user-facing result | `experiments/side_gyro_mag_comp/final/FINAL_RESULT.md` and `FINAL_DECISION.json` |

Reports summarize canonical machine artifacts; they do not replace them.

## 5. Ownership

### Codex-owned writes

- side-study design documents;
- authorized implementation and tests;
- side-study configs, manifests, results, and reports;
- Codex handoffs;
- stage transitions triggered by validated audit verdicts;
- final synthesis before Claude final audit.

### Claude-owned writes

- audit reports and machine audit JSON;
- Claude handoffs;
- final independent audit.

Claude does not edit implementation, checkpoints, experiment records, thresholds, or Codex result files.

### User interaction

The user reviews the final bundle only. Intermediate questions are prohibited unless the user independently intervenes.

## 6. One writer and four eyes

Only one implementation writer is active at a time. Codex may run read-only subagents in parallel, but only `sc_impl_worker` edits implementation in a stage.

A checkpoint advances only after Claude audits the sealed target. Codex local review is a pre-handoff check, not an independent signature.

## 7. PASS qualification

A PASS is valid only when all are true:

1. **Red path** — a named test or command fails when the claim is false.
2. **Non-empty population** — the inspected target population is greater than zero.
3. **Unambiguous target** — exactly one canonical target bundle is selected.
4. **Single predicate** — one verdict answers one question.
5. **Stable checkpoint** — the audited digest does not change during audit.

Fail-closed statuses:

- `TARGET_NOT_FOUND`
- `AMBIGUOUS`
- `STALE_CHECKPOINT`
- `UNKNOWN`
- `BLOCKED_CONTRACT_GAP`
- `BLOCKED_EXTERNAL_EXECUTION`

## 8. Minimum hypothesis set

- G0: oracle compensation headroom exists.
- G1: learned correction improves sensor and downstream metrics.
- G2: compensation features improve over corrected values alone.
- G3: feature shuffling removes a material portion of the feature gain.
- G4: nominal harmlessness holds.

No experiment is added unless one of G0-G4 would otherwise remain UNKNOWN.

## 9. Hard exclusions before final decision

- SNN or SoW
- reliability gating
- attention or Transformer
- learned Q/R or uncertainty head
- temperature, vibration, outlier, saturation, MTQ interference
- sun sensor or star tracker as runtime estimator input
- closed-loop and FPGA work
- broad KalmanNet-family comparison
- automated hyperparameter search
- test-driven tuning

## 10. Automatic audit and repair policy

For each auditable stage:

1. Codex seals a checkpoint.
2. Claude audits in fresh context.
3. If the verdict is repairable, Codex performs one minimum repair round.
4. Claude re-audits.
5. A second failure closes the workflow with a final rejected or blocked result.

No human escalation occurs mid-workflow. Every blocker includes a counterproposal in the final bundle.

## 11. Automatic gate behavior

- Failed G0: stop further neural compensation work and finalize `STOP_NO_COMPENSATION_HEADROOM`.
- Failed G1: skip feature experiments and finalize `REVISE_COMPENSATION_NETWORK`.
- Failed G2: finalize `LOCK_COMPENSATION_ONLY_REJECT_FEATURE_PATH` when compensation itself passed.
- Failed G3: finalize `REVISE_FEATURE_INTERFACE`.
- Failed G4: finalize `REVISE_FEATURE_INTERFACE` or `REVISE_COMPENSATION_NETWORK` according to the responsible path.
- Passed G0-G4 and audits: finalize `LOCK_COMPENSATION_CONDITIONED_SPLIT_MEKF_KALMANNET`.

## 12. Runtime and compute

Before a run expected to exceed 30 minutes, the experiment operator records wall-time, CPU/GPU/RAM assumptions, parallelism, disk estimate, and stop condition. This is a research workstation, not a data center. Precision or controls that cannot affect G0-G4 are deferred.

## 13. Existing configuration preservation

The repository's existing `.codex/config.toml`, including SpikeRA-KalmanNet settings, is authoritative. Package installation is additive. It must not remove or rewrite unrelated sections, profiles, models, MCP servers, approval policies, or existing agent limits.
