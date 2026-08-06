# Side Gyro–Mag Compensation v2 Final Result

Decision: `BLOCKED_CONTRACT_INCOMPLETE_AFTER_ONE_REPAIR`.

Method decision: `BLOCKED_IMPLEMENTATION_OR_INTEGRITY`.

The v2 study stops before implementation. G0–G4 are all `NOT_RUN`, no test or held-out payload was accessed, and there is no scientific performance conclusion.

## Why the study is blocked

The first red-capable local review found one repair class, `DECLARATIONS_NOT_FULLY_BOUND_TO_MACHINE_EXECUTION`, with five defects: missing split-aware metric identity and producer/evidence bindings; combined rather than separately addressable G1 metrics; incomplete exact gate arithmetic validation; non-exact R1–R13/R9 source projection binding; and incomplete changed-path, immutable-boundary, and command/access validation.

The single authorized CONTRACT repair produced canonical contract SHA-256 `50a1c4a...`, checkpoint SHA-256 `2aa9a457...`, validator result 7/7, and an in-memory mutation result of 485/485 rejected mutations.

The second red-capable review nevertheless found four decisive counterexamples:

1. Producer declarations identify a source `.py` file but not exact emitted JSON/CSV evidence paths, evidence schemas, and report-resolution rules.
2. The effective `repair_obligations` array is not derived from or deep-equal to the immutable source projection and can be weakened while validation remains green.
3. Wrong-namespace Claude-authored artifacts escaped the changed-path and v1-freeze validation boundary.
4. The declared producer path is outside the frozen implementation allowlist.

The 485/485 mutation result is therefore insufficient and is not a contract PASS. It establishes only that the encoded mutation set is rejected; it cannot cover missing predicates that were never encoded. Because the one repair round is exhausted, no further contract, validator, allowlist, implementation, or experiment work is authorized.

## Preservation cleanup

Three externally Claude-authored, untracked duplicate-v2 authority files were observed inside the v1 state namespace. Their pre-cleanup hashes are recorded in `FINAL_DECISION.json`. They were invalid as v2 authority or audit evidence and were deleted without relocation or replacement.

An independent full-filesystem comparison then found eight recreatable `__pycache__/*.pyc` files as the only remaining non-archive paths under the v1 code/test namespaces. Those exact generated cache paths are recorded in `FINAL_DECISION.json` and were removed; their hashes were not captured before independent removal. A second full comparison, now including ignored files, proves the complete v1 working-tree namespace matches the self-contained archive path-for-path and byte-for-byte: 85/85 paths present, zero missing, zero extra, zero byte mismatches, set digest `d9668c8976...`. The archive is base evidence commit `235619c...` plus self-contained tip `9cf80cc...`; the archived v1 final manifest remains 9/9 valid and `.codex/config.toml` remains `315ec7d2...`.

## Final audit

The Codex final bundle is sealed for an independent Claude audit. Codex has not created or fabricated `FINAL_AUDIT.md` or `FINAL_AUDIT.json`.
