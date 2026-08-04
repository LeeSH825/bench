# Phase 1A Gate C Validation Report

## Decision

**PASS_GATE_C — Gate C GO.** The canonical right-local attitude/state error,
bias metrics, star-tracker NIS, six-dimensional NEES, strict-SPD diagnostics,
and consistency summary are implemented and validated. Gate D is authorized
by this result but was not started.

## Created files

Only the five source/test/doc/report targets in the exact allowlist were
created:

- `bench/metrics/mekf.py`
- `tests/test_mekf_metrics.py`
- `docs/research/phase1a/P1A_CANONICAL_MEKF_METRICS_CONTRACT.md`
- `docs/research/phase1a/P1A_GATE_C_TEST_MATRIX.md`
- `experiments/phase1a/reports/P1A_GATE_C_VALIDATION_REPORT.md`

The recoverable snapshot is
`experiments/phase1a/preflight_snapshots/04_20260801T143933Z/`; execution
evidence is under `experiments/phase1a/agent_logs/04_20260801T143933Z_*`.

No Gate A, Gate B1, Gate B2, legacy, runner, registry, configuration,
visualization, Package C, or model source/test was modified.

## Runtime and baseline/post-regression results

All Python commands used
`/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python` with
`PYTHONDONTWRITEBYTECODE=1`; pytest disabled its cache provider.

| Suite | Before implementation | Final after implementation | Status |
|---|---:|---:|---|
| Gate C | not present | 43 passed | PASS |
| Gate A | 55 passed | 55 passed | PASS |
| Gate B1 Amendment A1 | 55 passed | 55 passed | PASS |
| Gate B2 | 67 passed | 67 passed | PASS |
| Specified legacy | 18 passed, 5 subtests | 18 passed, 5 subtests | PASS |

Pass/fail is based on process exit status and contract preservation; counts are
recorded as provenance. The final logs are the
`04_20260801T143933Z_final_*.txt` files.

## Canonical metric evidence

The implementation imports Gate A quaternion normalization, inverse,
multiplication, exact-pi-aware log, SPD validation, and Cholesky solve. It does
not recreate those functions. The locked error is

```text
delta_theta = Log_q(inverse(q_hat_NB) otimes q_true_NB)
delta_b = b_true - b_hat
e = [delta_theta, delta_b]
```

Closed-form tests cover identity, each principal axis, an arbitrary axis,
near-zero, exact pi, near-pi, known right-local state recovery, bias RMSE,
diagonal/full-SPD NIS, diagonal/full-SPD NEES, independent quaternion signs,
and chi-square summaries. Sixteen closed-form logical cases passed in the
targeted run, and all 43 Gate C tests passed in the full run.

The 10-case deterministic analytic sweep produced:

| Quantity | Actual |
|---|---:|
| Maximum right-local state recovery absolute error | `3.5388358909926865e-16` |
| Maximum q/-q NEES difference | `0` |
| Maximum NIS independent-solve difference | `1.3010426069826053e-18` |
| Maximum NEES independent-solve difference | `1.1102230246251565e-16` |
| Minimum NIS | `3.699742742390987e-05` |
| Minimum NEES | `0.0010845666154957656` |

The pytest property reference is an independent SciPy Cholesky factorization
and two triangular solves. Inputs were snapshotted and remained exact; all
array-valued results were read-only.

## Basilisk Gate B2 direct-replay smoke

Seeds 801, 802, and 803 each generated a three-trajectory, 0.2-second typed
Basilisk UNIT-ST dataset and used the frozen Gate B1 direct replay. Truth was
paired to replay posterior samples by exact trajectory and timestamp; NIS used
only actual star-tracker residual/S evidence.

| Quantity across three seeds | Actual |
|---|---:|
| Maximum attitude error | `0.02731741267507712 rad` |
| Maximum bias-error vector norm | `0.01099084841880689 rad/s` |
| Maximum NIS | `0.8159568391115934` |
| Maximum NEES | `6.186514476849704` |
| Minimum posterior P eigenvalue | `2.2447900045270017e-06` |
| Minimum innovation S eigenvalue | `0.0010422454095772664` |
| Maximum q/-q attitude metric difference | `0` |
| Maximum q/-q NEES difference | `0` |

All metrics were finite/nonnegative and all P/S matrices passed strict
Cholesky. The targeted smoke result was 3 passed.

## Negative, pairing, and boundary validation

Tests reject float32 and non-array public inputs, empty arrays, nonfinite
values, shape/count/batch mismatches, asymmetric matrices, singular or
indefinite matrices, invalid consistency values/degrees of freedom, partial
pairing metadata, timestamp mismatches, and trajectory-ID mismatches. No
interpolation, event alignment, or posterior selection is inferred.

AST and isolated-process checks show that `bench.metrics.mekf` imports no
Basilisk, task/generator, runner, model, torch, or visualization module. A
source-level check confirms absence of explicit matrix inverse,
pseudo-inverse, least-squares fallback, diagonal perturbation, clipping, and
repair paths. The dedicated boundary check passed 2/2.

The first new-suite run was 42 passed and one failed assertion: a known bias
delta was formed by nonzero floating-point addition and then compared with
bitwise `array_equal`. The test input was corrected to a zero bias origin so
the closed-form exact assertion is valid. No implementation tolerance was
changed, and no skip or xfail was added. The corrected and final runs are
43/43.

## Dirty-tree integrity

The execution-start current tree was captured before implementation, including
tracked/staged patches, an untracked recovery archive, 1,463 dirty-path
status/content fingerprints, frozen-path hashes, allowlist existence, and
runtime versions.

The integrity comparison found:

| Check | Actual | Status |
|---|---:|---|
| Preexisting outside-allowlist path changes | 0 | PASS |
| Frozen Gate A/B1/B2/source/test/doc hash changes | 0 | PASS |
| Staged patch equality | exact, 0 bytes before/after | PASS |
| Concurrent external paths after snapshot | 0 | PASS |
| Modified-file whitespace diagnostics | 0 lines | PASS |

Existing dirty changes were not reset, restored, cleaned, stashed, staged,
committed, or pushed.

## Deferred scope and blocking issues

Blocking issues: none.

Deferred by contract: runner and artifact integration, registry/dispatch,
YAML, cache/sidecar integration, visualization, Package C experiments,
latency/OOSM, expanded sensors/faults, and neural/ANN/SNN/FPGA models. The
metrics contract records the exact timestamp, trajectory, posterior, truth,
star-tracker residual, and innovation-covariance evidence Gate D must preserve.

## Final gate

- Attitude geodesic/right-local error: PASS
- Bias error/RMSE: PASS
- Star-tracker NIS: PASS
- Right-local 6D NEES: PASS
- q/-q metric invariance: PASS
- SPD diagnostics: PASS
- Chi-square consistency summary: PASS
- Timestamp/posterior pairing: PASS
- Numerical fail-loud policy: PASS
- No inverse/pseudo-inverse/perturbation/clipping: PASS
- Import boundary: PASS
- B2 replay metric smoke: PASS
- Gate A/B1/B2/legacy regressions: PASS
- Dirty-tree integrity: PASS

**Gate C: GO**

**Gate D authorized: YES — not executed.**
