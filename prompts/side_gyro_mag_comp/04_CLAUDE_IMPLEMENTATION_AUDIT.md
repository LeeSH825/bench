# CLAUDE — Independent Implementation Audit

Audit the sealed implementation checkpoint. Do not repair code.

Spawn in parallel:

- `sc-independent-math-auditor`
- `sc-code-contract-auditor`
- `sc-governance-auditor`
- `sc-lean-scope-auditor`

Verify:

- equations and code use the same frame, sign, dimensions, and state role;
- deterministic compensation and residual bias are not double counted;
- runtime is truth/oracle/future-free;
- same-realization and whole-trajectory checks fail closed;
- each red-path test actually becomes red under its counterexample;
- feature routing is branch-specific and no attention/extra path was added;
- N3S can reuse N3 checkpoint and alter only feature association;
- frozen paths and main Phase 2 boundary remain intact;
- target populations in smoke tests are non-empty.

Write only audit and Claude handoff paths.

Allowed decisions:

- `PASS_IMPLEMENTATION_AUTONOMOUS_ADVANCE`
- `FAIL_MATH_OR_FRAME`
- `FAIL_INFORMATION_LEAKAGE`
- `FAIL_RED_PATH`
- `FAIL_SCOPE_CREEP`
- `UNKNOWN`
- `STALE_CHECKPOINT`

One automatic repair round maximum; a second failure closes the workflow with a final blocked result.
