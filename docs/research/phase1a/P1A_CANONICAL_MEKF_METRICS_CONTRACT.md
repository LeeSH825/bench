# Phase 1A Canonical MEKF Metrics Contract

## Purpose and basis

This document locks the evaluation-only metric contract for the Phase 1A
six-dimensional kinematic MEKF. It builds on the approved Gate A math and the
Gate B1/B2 typed replay evidence without changing those gates. The canonical
implementation is `bench/metrics/mekf.py`; existing generic or visualization
metrics are not canonical for this MEKF.

The metric layer observes estimates, posterior evidence, and evaluation truth.
It never feeds a value back to an estimator, generator, replay, sensor model,
runner, or training process.

## Locked convention

`q_NB` is a scalar-first Hamilton quaternion representing the active rotation
from body coordinates to navigation coordinates. The local attitude error is
right multiplicative:

```text
q_true_NB = q_hat_NB otimes Exp_q(delta_theta)
delta_q = inverse(q_hat_NB) otimes q_true_NB
delta_theta = Log_q(delta_q)
```

The bias and state errors are:

```text
delta_b = b_true - b_hat
e = [delta_theta, delta_b] in R^6
```

The Gate A normalization, multiplication, inverse, and exact-pi-aware log map
are imported rather than duplicated. Consequently replacing either input
quaternion by its negative leaves every physical metric unchanged, including
the deterministic exact-pi case.

## Metrics

The attitude geodesic error is the primary log-map quantity

```text
attitude_error_rad = norm(delta_theta)
```

and is optionally converted to degrees. A component quaternion difference or
an acos-only roundoff path is not a valid substitute.

For bias samples, the implementation returns `delta_b`, its per-sample vector
norm, the RMSE of each axis over all samples, and

```text
bias_vector_RMSE = sqrt(mean(norm(delta_b)^2)).
```

For an actual star-tracker update residual `r` and its matching innovation
covariance `S`,

```text
NIS = transpose(r) @ solve(S, r).
```

No placeholder NIS is produced for gyro rows or for rows without residual
evidence. For a posterior estimate, its matching right-local covariance `P`,
and evaluation truth,

```text
NEES = transpose(e) @ solve(P, e).
```

Both solves use the frozen Gate A strict-SPD Cholesky/triangular-solve path.

## Shape, dtype, and unit contract

| Quantity | Shape | Dtype | Unit / meaning |
|---|---|---|---|
| `q_hat_NB`, `q_true_NB` | `(...,4)` | `float64` ndarray | unit quaternion representation |
| `b_hat`, `b_true` | `(...,3)` | `float64` ndarray | `rad/s` |
| `delta_theta` | `(...,3)` | read-only `float64` | `rad` |
| `delta_b` | `(...,3)` | read-only `float64` | `rad/s` |
| `e` | `(...,6)` | read-only `float64` | `[rad, rad/s]` local coordinates |
| attitude error | `(...)` | read-only `float64` | `rad`, or `deg` in the explicit helper |
| bias vector norm | `(...)` | read-only `float64` | `rad/s` |
| `r` | `(...,3)` | `float64` ndarray | `rad` |
| `S` | `(...,3,3)` | `float64` ndarray | `rad^2`, symmetric strict SPD |
| `P` | `(...,6,6)` | `float64` ndarray | right-local posterior covariance, strict SPD |
| NIS, NEES | `(...)` | read-only `float64` | dimensionless, finite, nonnegative |

All leading batch shapes must match exactly; broadcasting is not used to hide
count or alignment defects. Inputs must be NumPy arrays with exact locked
dtypes and finite values. All array-valued results are defensive read-only
copies, and inputs are not mutated.

## Timestamp, trajectory, and posterior pairing

Gate C performs no interpolation, nearest-time lookup, event alignment, or
prior/posterior guessing. When pairing metadata is supplied to NIS or NEES,
all members of that metadata group are mandatory and must be array-equal with
the metric batch shape.

For NEES, `q_hat`, `b_hat`, `P`, `q_true`, and `b_true` must identify the same
trajectory, the same physical timestamp, and the same posterior tangent. The
locked implementation can explicitly validate estimate/covariance/truth time
and trajectory-ID arrays. For NIS, `r` and `S` must identify the same valid
star-tracker update, timestamp, and trajectory. A count, timestamp, trajectory,
or metadata-presence mismatch fails loudly.

## Truth and use boundaries

NIS receives no truth. Attitude, bias, and NEES functions may receive truth
only after estimation for evaluation. Metrics do not expose future truth to
replay, infer event labels, create oracle `Q` or `R`, or alter state, truth,
events, residuals, or covariance evidence.

NIS is meaningful only for the residual and `S` emitted by an actual update
under the same innovation definition. NEES is meaningful only when `P` is the
posterior covariance of the supplied right-local six-dimensional error at the
same sample. Values computed from mismatched priors, posteriors, timestamps,
trajectories, or tangent conventions are invalid even if their shapes happen
to agree.

## SPD and numerical safety

The common SPD diagnostic returns relative Frobenius asymmetry, minimum
eigenvalue, Cholesky success, and matrix dimension. It does not modify or
return a repaired matrix. Asymmetric, nonfinite, singular, or indefinite `P`
and `S` fail loudly.

The canonical module contains no explicit matrix inverse, pseudo-inverse,
least-squares fallback, diagonal perturbation, eigenvalue clipping, covariance
repair, nonfinite masking, component quaternion MSE, or additive quaternion
subtraction. Strict Cholesky failure is an error, not a signal to switch
algorithms.

## Consistency summary and empty policy

For a nonempty one-dimensional NIS or NEES array, the summary reports count,
degrees of freedom per sample, sum, mean, and `mean/dof_per_sample`. At
confidence `c`, it also reports the central batch-sum interval from

```text
sum(values) ~ chi_square(count * dof_per_sample).
```

This is a diagnostic interpretation only under independent samples and a
correctly paired, matched Gaussian model. Correlation, data reuse, tuning on
the evaluation set, model mismatch, or incorrect pairing invalidate the
nominal confidence interpretation.

Empty attitude, state, bias, NIS, NEES, SPD, and consistency inputs fail
loudly. NaN/Inf, negative consistency values, invalid degrees of freedom, and
invalid confidence levels also fail. The module never returns NaN as an empty
sample sentinel.

## Public API

The source of truth exports:

- `right_local_state_error`
- `attitude_geodesic_error_rad` and `attitude_geodesic_error_deg`
- `bias_error_summary`
- `star_tracker_nis`
- `right_local_nees`
- `spd_diagnostics`
- `consistency_summary`

Their immutable result records are `RightLocalStateError`,
`BiasErrorSummary`, `SPDDiagnostics`, and `ConsistencySummary`.

## Gate D artifact requirements

Gate D must preserve, without lossy conversion or inferred alignment:

```text
timestamp
trajectory_id
q_hat_NB
b_hat
P
star-tracker update mask and update timestamp
star-tracker residual r
star-tracker innovation covariance S
q_true_NB
b_true
```

`q_hat`, `b_hat`, `P`, `q_true`, and `b_true` must share trajectory, physical
timestamp, and prior/posterior convention. `r` and `S` must share the same
star-tracker update and update timestamp. Gate D, not Gate C, is responsible
for artifact and runner integration.
