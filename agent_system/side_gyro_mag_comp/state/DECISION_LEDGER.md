# Side Gyro–Mag Compensation Decision Ledger

| ID | Date | Decision | Status | Decider | Evidence | Supersedes |
|---|---|---|---|---|---|---|
| SC-D000 | YYYY-MM-DD | Initialize autonomous-until-final side-study workflow | current | delegated state machine | package install report | — |
| SC-D001 | 2026-08-06 | Seal DR0 design bundle | superseded by repair seal | Codex | `experiments/side_gyro_mag_comp/design_review/CHECKPOINT_MANIFEST.json` digest `298feba2...f730` | — |
| SC-D002 | 2026-08-06 | DR0 independent audit requires one minimum repair | current | Claude | `experiments/side_gyro_mag_comp/audits/SC_DR0_INDEPENDENT_AUDIT.json` | SC-D001 PASS implication only |
| SC-D003 | 2026-08-06 | Reseal DR0 after the one authorized repair | current | Codex | `experiments/side_gyro_mag_comp/design_review/CHECKPOINT_MANIFEST.json` digest `a253af84...fa26` | SC-D001 |
| SC-D004 | 2026-08-06 | DR0 re-audit passes and authorizes implementation | current | Claude | `experiments/side_gyro_mag_comp/audits/SC_DR0_REAUDIT.json` | SC-D002 |
| SC-D005 | 2026-08-06 | Seal minimal implementation for audit | superseded by repair | Codex | `experiments/side_gyro_mag_comp/implementation/CHECKPOINT_MANIFEST.json` digest `68a08285...a90a5` | — |
| SC-D006 | 2026-08-06 | Implementation audit fails red-path predicate; authorize one minimum repair | current | Claude | `experiments/side_gyro_mag_comp/audits/SC_IMPLEMENTATION_AUDIT.json`; `SC_IMPL_ROUND1_COUNTERPROPOSAL.json` | SC-D005 PASS implication only |
| SC-D007 | 2026-08-06 | Restore prompt 05 operational text to installed-package bytes; canonical repaired G3 remains SC-01 | current | Codex orchestrator | prompt 05 restored; SC-01 and DR0 re-audit remain gate authority | temporary in-stage prompt edit |
| SC-D008 | 2026-08-06 | Terminate implementation repair with `BLOCKED_CONTRACT_GAP` and skip experiments | current | Codex orchestrator | Claude R9 clarification resolves sensor aggregation only; canonical sources and R8 provide no frozen weak-axis population threshold/membership rule; writer made no repair edits | SC-D006 advancement counterproposal |

## Rules

- Every automatic transition records the exact audit or gate artifact.
- Thresholds and architecture are not changed after test access.
- A second failed audit closes the study rather than forcing a pass.
- Main Phase 2 and SpikeRA-KalmanNet authorization are outside this ledger.
