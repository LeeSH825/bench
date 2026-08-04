# Phase 1B Step 1 Identifiability Pilot Report

## Frozen A/B construction

Case A is C2 medium (`alpha_g=4`, `alpha_R=1`). Case B is C3 inlier reliability
with `alpha_g=1`. B was selected from the predeclared grid on the pilot
validation split (17 trajectories) and frozen before the independent test 50.

```text
selected B alpha_R:             1.08
validation innovation-RMS diff: 0.396%
independent-test F-BASE diff:    1.485%
required matching tolerance:     5%
result:                          PASS
```

No B parameter was changed after test observation.

## Matched evidence, independent test N=50

| Quantity, F-BASE | A process-side | B measurement-side | Interpretation |
|---|---:|---:|---|
| ST innovation RMS relative gap | reference | +1.485% | magnitude matched |
| Innovation norm P95 (rad) | 4.2626e-3 | 4.3451e-3 | close upper distribution |
| Innovation norm lag-1 mean | -0.2476 | -0.2674 | similar, not identical |
| Raw gyro measurement-increment RMS (rad/s) | 1.4396e-3 | 7.0927e-4 | mechanism remains distinguishable in raw gyro |
| Event attitude RMSE B−A (rad) | — | -2.758e-5 | 95% CI [-8.66e-5,+3.29e-5] |
| NIS normalized mean | 0.931 | 0.940 | similar |
| NEES normalized mean | 1.056 | 0.974 | both near one |

Thus scalar innovation RMS alone does not identify which covariance side
changed in this controlled pair. This is not a claim that the complete
innovation history or full raw sensor stream can never distinguish the cases:
the raw gyro increment statistic separates them clearly, and the autocorrelation
and P95 values are not exactly identical.

## Correct-side and wrong-side actions

For C5-A, applying the measurement-side wrong action increases attitude event
RMSE by `+2.146e-4 rad`, paired 95% CI `[+8.88e-5,+3.40e-4]`. The correct Qg
oracle changes attitude RMSE by only `+8.88e-7 rad` with a CI spanning zero but
restores NEES from 1.056 to 0.965.

For mild C5-B (`alpha_R=1.08`), the correct R oracle and wrong Qg action both
have attitude differences close to zero. The stronger C3 rows establish the
action separation: at alpha_R=8 the correct oracle improves RMSE by
`-1.318e-3 rad`, while wrong-side Qg inflation worsens it by `+4.487e-5 rad`.

Preliminary decisions:

```text
H3 process/measurement action separation: SUPPORTED by C2 wrong-side harm and
                                           severity-growing C3 oracle benefit
H4 scalar innovation RMS has a structural limit: SUPPORTED for this paired
                                                   UNIT-ST construction only
```

## Claim limits

The matched B event is mild, the pilot horizon is 10 s, and only gyro plus
inlier quaternion ST are present. Results do not prove an information-theoretic
impossibility, do not justify a neural model by themselves, and do not cover
C4, outliers, outages, latency, magnetometer or sun sensor. A future feature
contract should retain raw gyro evidence rather than treat scalar innovation
RMS as sufficient context.

Canonical evidence is
`experiments/phase1b/results/unit_st_classical_v1/pilot_summary.json`; the
simulation-only A/B sidecars remain outside estimator artifacts.
