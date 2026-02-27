# Metrics Spec

## Scalar Metrics
- `mse`: mean squared error over all `(N,T,D)` elements.
- `rmse`: `sqrt(mse)`.
- `mse_db`: `10 * log10(max(mse, 1e-30))`.
- `timing_ms_per_step`: prediction wall-clock normalized by `(N*T)`.
- `nll`: `NA` when covariance is unavailable.

## Shift Metric
- `shift_recovery_k`: computed from per-step MSE after shift time `t0` with:
- window `W=20`
- tolerance `eps=0.05`
- failure policy `cap`.

## Budget/Op Metrics
- `train_updates_used` (alias of outer updates for MAML).
- `train_outer_updates_used` (explicit outer optimizer step count).
- `train_inner_updates_used` (explicit inner-loop optimizer step count).
- `adapt_updates_used`.
- `adapt_updates_per_step_max`.
- time breakdowns: `train_time_s`, `eval_time_s`, `adapt_time_s`, `total_time_s`.

## Severity Sweep Plot Fields
- Allowed x fields:
- `severity_key`
- `mse_mean`
- `mse_db_mean`
- `rmse_mean`
- `inv_r2_db`
- `severity_r_scale_mean`
- Allowed y fields:
- `mse_mean`
- `mse_db_mean`
- `rmse_mean`
- `severity_key`
- `severity_r_scale_mean`
- Default: `x=severity_key`, `y=mse_mean`.
