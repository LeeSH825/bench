# CLAUDE — Independent Evidence and Statistics Audit

Do not rerun training unless machine artifacts are inconsistent and the state machine marks a machine-artifact inconsistency requiring the single automatic repair round.
Recompute statistics from sealed per-trajectory records.

Spawn in parallel:

- `sc-evidence-statistician`
- `sc-code-contract-auditor`
- `sc-lean-scope-auditor`
- `sc-governance-auditor`

Verify only:

1. Was compensation headroom demonstrated by N1 versus N0?
2. Did learned correction improve sensor and downstream metrics?
3. Did N3 add incremental value over N2?
4. Did N3S remove the feature-related value using the same checkpoint?
5. Was nominal harmlessness satisfied?
6. Were populations non-empty, trajectories paired, seeds scoped, and CIs reproducible?
7. Were gates applied exactly without post-test relaxation?
8. Are weak-axis limitations and unsupported covariance claims handled honestly?

Write:

- `SC_EVIDENCE_AUDIT.md`
- `SC_EVIDENCE_AUDIT.json`
- Claude handoff JSON
- recomputation command log

Allowed decisions:

- `PASS_EVIDENCE_FOR_METHOD_LOCK_REVIEW`
- `PASS_COMPENSATION_REJECT_FEATURE`
- `FAIL_NO_HEADROOM`
- `FAIL_COMPENSATION`
- `FAIL_FEATURE_INCREMENT`
- `FAIL_FEATURE_SHORTCUT`
- `FAIL_NOMINAL_HARM`
- `FAIL_DATA_OR_STATISTICS`
- `UNKNOWN`
- `STALE_CHECKPOINT`

Do not request additional experiments merely to improve performance.
A new experiment is allowed only when a current G0-G4 predicate is otherwise UNKNOWN.
