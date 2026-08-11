# Start Here — Final-Only Autonomous Operation

The user runs only two entry prompts:

- Codex App: `10_CODEX_AUTONOMOUS_PRIMARY.md`
- Claude Code: `11_CLAUDE_AUTONOMOUS_AUDITOR.md`

Prompts `01` through `09` are internal stage specifications read and executed by the two primary agents. Do not ask the user to run them one by one.

The apps coordinate through:

- `agent_system/side_gyro_mag_comp/state/STAGE_STATE.json`
- sealed Codex handoffs
- Claude audit handoffs
- one automatic repair round per audited stage

The user checks only:

- `experiments/side_gyro_mag_comp/final/FINAL_RESULT.md`
- `experiments/side_gyro_mag_comp/final/FINAL_DECISION.json`

Do not request intermediate research approval. External blockers are recorded in the final result.
