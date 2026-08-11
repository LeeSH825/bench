# Autonomous State-Machine Specification

## General

- No intermediate user approval.
- Codex and Claude alternate through `next_actor` in `STAGE_STATE.json`.
- Every state update uses `agent_system/side_gyro_mag_comp/scripts/update_stage_state.py` and is validated immediately.
- The actor that does not own the next state waits with `wait_for_peer.py`.
- One repair round per audited stage.
- Terminal states always produce a final bundle.

## DR0

1. Codex runs prompt 01 and seals the design bundle.
2. State -> `DR0_INDEPENDENT_AUDIT`, `WAITING_FOR_PEER`, next actor Claude.
3. Claude runs prompt 02.
4. PASS -> implementation.
5. Repairable FAIL -> DR0 repair once, then re-audit.
6. Second FAIL/UNKNOWN -> terminal blocked result.

## Implementation

1. Codex runs prompt 03 with one writer.
2. State -> implementation audit, next actor Claude.
3. Claude runs prompt 04.
4. PASS -> oracle headroom.
5. Repairable FAIL -> implementation repair once, then re-audit.
6. Second FAIL/UNKNOWN -> terminal blocked result.

## Experiment

1. Codex runs prompt 05 in gate order.
2. Failed G0/G1/G2/G3/G4 follows the frozen decision mapping and skips unnecessary later work.
3. Codex seals evidence.
4. Claude runs prompt 06.
5. Repairable machine-evidence defect gets one repair; no architecture or threshold change.
6. Irrecoverable evidence failure -> terminal blocked result.

## Final

1. Codex runs prompt 07 and seals final result.
2. Claude performs final audit and writes `FINAL_AUDIT.md/.json`.
3. If final audit matches evidence, state -> COMPLETE.
4. If final bundle misstates existing evidence, Codex may make one reporting-only repair and Claude re-audits.
5. If conflict remains, state -> COMPLETE_BLOCKED.

## External blockers

Authentication, permissions, missing executables, disk exhaustion, or unrecoverable resource failure produce `BLOCKED_EXTERNAL_EXECUTION`; agents record exact resume commands and finalize without asking mid-stage questions.
