# Phase 1B Step 1 Problem-Existence Report

## Scope and decision

The paired N=50 pilot supports a real time-varying measurement-reliability
problem in C3 and a smaller, monotonic process-consistency effect in C2. The
correct-side oracle is strongly useful in C3 and restores C2 NEES consistency,
but does not improve C2 event attitude RMSE at the tested normalized profile.

Preliminary decisions:

```text
H1 time-varying adaptation is needed: SUPPORTED, dominated by C3; C2 effect modest
H2 correct-side oracle context is useful: SUPPORTED for C3 and C2 consistency,
                                           not for C2 attitude RMSE
```

These are UNIT-ST pilot decisions, not general flight claims.

## Severity trends

F-BASE means:

| Condition | Event attitude RMSE (rad) | NIS norm. | NEES norm. | Divergence |
|---|---:|---:|---:|---:|
| C1 stationary | 1.6615e-3 | 0.924 | 0.967 | 0/50 |
| C2 alpha_g=2 | 1.6824e-3 | 0.927 | 0.993 | 0/50 |
| C2 alpha_g=4 | 1.7081e-3 | 0.931 | 1.056 | 0/50 |
| C2 alpha_g=8 | 1.7520e-3 | 0.939 | 1.187 | 0/50 |
| C3 alpha_R=2 | 2.0413e-3 | 1.114 | 1.080 | 0/50 |
| C3 alpha_R=4 | 2.6979e-3 | 1.492 | 1.316 | 0/50 |
| C3 alpha_R=8 | 3.6859e-3 | 2.245 | 1.791 | 0/50 |

C2 severe raises F-BASE event RMSE by about 5.4% from stationary and moves NEES
from 0.967 to 1.187. C3 severe raises event RMSE by about 122% and causes clear
NIS/NEES overconfidence.

## Oracle and wrong-side evidence

| Condition / comparison | Mean event RMSE difference vs F-BASE (rad) | Paired bootstrap 95% CI | Consistency evidence |
|---|---:|---|---|
| C2 medium oracle | +8.88e-7 | [-1.06e-6, +2.68e-6] | NEES 1.056→0.965 |
| C2 severe oracle | +1.10e-6 | [-3.52e-6, +5.38e-6] | NEES 1.187→0.968 |
| C2 medium wrong-side | +2.146e-4 | [+8.86e-5, +3.43e-4] | wrong R action harms |
| C2 severe wrong-side | +4.908e-4 | [+3.21e-4, +6.61e-4] | wrong R action harms |
| C3 mild oracle | -1.184e-4 | [-1.92e-4, -4.41e-5] | NIS 1.114→0.926 |
| C3 medium oracle | -5.238e-4 | [-6.89e-4, -3.65e-4] | NIS 1.492→0.927 |
| C3 severe oracle | -1.318e-3 | [-1.609e-3, -1.048e-3] | NIS 2.245→0.927; NEES 1.791→0.951 |
| C3 severe wrong-side | +4.487e-5 | [+3.45e-5, +5.46e-5] | NIS remains 2.202 |

Correct measurement-side R action has a repeated and severity-growing C3
benefit. In C2, Qg oracle mainly improves covariance consistency while the raw
state error change is below this pilot's resolution. Wrong-side R inflation is
actively harmful at C2 medium/severe, demonstrating that an arbitrary shared
action is not safe.

## F-TUNED trade-off

F-TUNED gives small negative paired attitude differences in all C2/C3 rows, but
its NIS/NEES normalized means remain approximately 0.15–0.32. Its improved bias
RMSE and small attitude benefit therefore do not replace the correct-side
oracle evidence; the filter is substantially over-conservative and has the
long-horizon penalty documented in the baseline report.

## Limitations

The process event changes gyro white noise only; alpha_b remains one. The truth
is spherical-inertia constant-rate Basilisk, latency is zero, and ST events are
all valid/inlier. Recovery is uniformly one gyro sample in this short profile,
so it does not discriminate policies. C4, sensor outages, false solutions and
combined sensor events remain deferred to an explicitly authorized later step.
