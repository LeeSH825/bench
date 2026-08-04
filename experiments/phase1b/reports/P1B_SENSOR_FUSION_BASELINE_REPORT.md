# Phase 1B Sensor Fusion Baseline Report

Status: `PASS`, paired held-out `N=50`, 30 s per trajectory.

The primary baseline remained `F-BASE`. Its MAIN-FUSION stationary result had
zero numerical divergence, mean attitude RMSE `0.0310341 rad`, mean slow-window
bias RMSE `3.33183e-4 rad/s`, and strict positive-definite posterior and sensor
innovation covariances. The all-one full-oracle path was exactly equal to
F-BASE for every q/b/P/residual/S artifact.

The whole-horizon normalized means were mag NIS `3.0354`, sun NIS `1.7585`, ST
NIS `1.3263`, and NEES `3.8033`. These include the deliberately randomized
initial attitude transient. The separately frozen 20%-horizon settled check
used 6,050 mag, 2,215 valid-sun, 1,250 ST, and 21,800 posterior samples. Settled
normalized means were mag `1.02290`, sun `1.00049`, ST `1.09189`, and NEES
`1.87302`. Sensor-specific NIS therefore passes its matched settled sanity
check; the remaining posterior overconfidence is a named classical limitation,
not hidden by covariance repair or retuning.

`F-TUNED` remained unchanged at `(0.125,0.125,8.0)` and was only a sensitivity
comparator. Relative to F-BASE it increased stationary mean attitude RMSE by
`0.185%`, fast-window peak by `54.5%`, slow-bias RMSE by `47.5%`, and normalized
NEES by `52.2%`. It is not promoted to primary baseline.

The N=20 paired ablation mean attitude RMSE values were gyro+ST `0.06657`,
gyro+mag+ST `0.04771`, gyro+sun+ST `0.05318`, and full fusion `0.03422` rad.
These are contribution sanity checks only and were not used for tuning.

Magnetic/sun references had valid separation approximately `86–89 deg`. This
is intentionally favorable deterministic benchmark geometry and is not an
orbit, WMM, eclipse, or flight-representativeness claim.
