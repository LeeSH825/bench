# Phase 1B C4 Combined Event Report

Status: `PASS`, paired held-out `N=50` for every policy.

C4 used the predeclared slow gyro-bias random-walk multiplier
`alpha_b=100000` on `[0.2T,0.8T)` and fast magnetometer inlier covariance
multiplier `alpha_R_mag=16` on `[0.45T,0.6T)`. The large dimensionless process
multiplier is relative to the normalized `1e-12 rad^2/s^3` base PSD and is not
a hardware value. The magnetic event changed zero-mean Gaussian covariance
only; no mean bias, interference, outlier, or invalid packet was introduced.

All policies had zero divergence. Against F-BASE:

- process-only reduced slow-bias RMSE by `18.03%` (paired difference 95% CI
  `[-3.0205e-4,-1.4431e-4] rad/s`) and normalized NEES by `95.76%`, but worsened
  fast attitude peak by `24.27%`;
- measurement-only reduced mag normalized NIS by `39.26%` (CI
  `[-2.3358,-2.1629]`) but did not improve fast attitude peak or slow bias;
- full oracle reduced slow-bias RMSE by `28.56%` (CI
  `[-4.2043e-4,-2.8351e-4] rad/s`), fast attitude peak by `32.57%` (CI
  `[-1.6568e-3,-1.0359e-3] rad`), mag normalized NIS by `47.20%`, and normalized
  NEES by `96.32%`;
- wrong-process produced only `0.047%` slow-bias improvement and essentially
  zero fast-state change;
- wrong-measurement lowered mag NIS by overinflating the wrong measurement
  side, but worsened slow-bias RMSE by `4.50%` and fast peak by `4.48%`.

Thus process-only and measurement-only actions carry cause-specific evidence,
while the full action resolves their tradeoff in this combined regime. NIS
alone is explicitly insufficient: the wrong-measurement diagnostic obtains a
low mag NIS while state/bias behavior degrades. This supports separation for
this benchmark, not universal identifiability.

Trajectory IDs, attitude/rate truth, and C4-unaffected sun and star-tracker
payloads were exactly equal to MAIN-FUSION stationary. Fixed/tuned APIs received
no sidecar, window, event label, or truth. Oracle policies consumed only the
current sidecar event through a forward-only cursor.
