# CLAUDE — Independent SC-DR0 Audit

Operate in fresh context. Do not read or adopt Codex's concluding prose before
independent derivation; read canonical source, equations, and changed files first.

## Setup

1. Read `CLAUDE.md` and the dual-agent charter.
2. Validate stage state.
3. Read the sealed Codex handoff and verify one stable checkpoint.
4. Spawn in parallel:
   - `sc-independent-math-auditor`
   - `sc-lean-scope-auditor`
   - `sc-governance-auditor`
5. Wait for all results.

## Questions

- Is the method mathematically coherent?
- Are gyro compensation and residual bias state identifiable under the declared role split?
- Are magnetometer model, frame, innovation, and Jacobian compatible with current MEKF conventions?
- Is the 6x3 gain and right injection correct?
- Is the causal feature routing free of truth/oracle/future leakage?
- Is weak-axis observability represented honestly?
- Is each proposed experiment necessary to decide G0-G4?
- Does every proposed PASS have a future red path and non-empty population?
- Is the side study isolated from main Phase 2 authorization?

## Write ownership

Write only:

- `experiments/side_gyro_mag_comp/audits/SC_DR0_INDEPENDENT_AUDIT.md`
- `experiments/side_gyro_mag_comp/audits/SC_DR0_INDEPENDENT_AUDIT.json`
- `experiments/side_gyro_mag_comp/handoffs/claude/CLAUDE_TO_CODEX.json`

Allowed decisions:

- `PASS_DR0_AUTONOMOUS_ADVANCE`
- `FAIL_REVISE_MATH`
- `FAIL_REVISE_SCOPE`
- `FAIL_INFORMATION_BOUNDARY`
- `UNKNOWN_INSUFFICIENT_CANONICAL_EVIDENCE`
- `STALE_CHECKPOINT`

Every blocker must include one minimum counterproposal covering the full failure class.
Do not edit Codex files.
