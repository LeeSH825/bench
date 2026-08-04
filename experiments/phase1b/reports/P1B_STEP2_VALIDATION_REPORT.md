# Phase 1B Step 2 Validation Report

Final status: `PASS_P1B_STEP2_SENSOR_FUSION_C4`.

## Implementation and experiment evidence

- New schema/generator/metric/experiment source is confined to the exact
  allowlist. Frozen Phase 1A and Step 1 files were not modified.
- `validate` passed exact Phase 1A gyro+ST payload/replay reduction, all-one
  fixed/oracle equality, truth-free fixed signature, strict round-trip identity,
  invalid-sun state/covariance skip, and sensor-specific NIS.
- New tests: `38 passed` after implementation.
- Workload: 84 generated trajectories, 30 s at 10/5/2/1 Hz, 680 records,
  upper bound 367,200 filter-event steps.
- MAIN-FUSION, STRESS-MAG, and C4 each completed paired test N=50; four
  ablations each completed N=20. Initial pilot elapsed time was `251.61 s`.
- `pilot --resume` strictly loaded all three datasets and reused all 680
  checkpoints in `1.24 s` without rewriting physical streams.
- Physical dataset hashes were MAIN
  `46db727729f0bcbadca8368adf3b1bd5d0601fac2058ecff8dd7e7081f1a4d0b`,
  STRESS `d31ccf2dc71c50b51671e8ce2bc3fccd7eb68cec9c1da7441b94bee24e14ad84`,
  and C4 `24ac6fcc7de344657a41e2095b9a1981fa02a8804f1e86f9e9ab411394522c3c`.

## Post-implementation regression

| Suite | Result |
|---|---:|
| New Step 2 | 38 passed |
| Gate A | 55 passed |
| Gate B1 | 55 passed |
| Gate B2 | 67 passed |
| Gate C | 43 passed |
| D1 bridge | 24 passed |
| CP4 integration | 22 passed |
| Phase 1B Step 1 | 52 passed |
| Legacy | 18 passed, 5 subtests passed |

The real Phase 1A smoke ran twice after implementation with synthetic and
Basilisk producers at seeds 6101–6103. The first six results were
`fresh_generation`, the next six were `verified_cache_hit`, and every paired
dataset hash matched. Step 1 read-only validation passed; its 1,950-record
summary and C5 calculation remained exactly equal to the frozen report.

No test used tolerance relaxation, skip, xfail, jitter, inverse, pseudo-inverse,
clipping, or covariance repair.

## Dirty-tree integrity

The preflight snapshot contains the recoverable tracked/staged patches,
untracked source archive/path ledger, runtime manifest, allowlist existence,
and 2,171 frozen file hashes. Final re-hashing found `0` frozen mismatches.
Status comparison with the same normal-untracked mode found no removed initial
line and no unexpected new line outside the allowlist. Existing unrelated and
visualization paths were neither read for implementation nor modified.
