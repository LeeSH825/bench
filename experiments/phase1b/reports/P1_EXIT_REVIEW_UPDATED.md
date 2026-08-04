# P1 Exit Review — Updated After Covariance Closure

Decision: **CONDITIONAL_GO**

Closure status: `PASS_P1_EXIT_CONDITION_CLOSURE`. The original condition was settled posterior NEES/DOF `1.873` with matched MAIN-FUSION sensor NIS.

## Diagnosed cause and calibrated policy

The transient, attitude marginal, bias marginal, full whitened-error, and attitude-bias cross-covariance evidence is frozen in `P1_EXIT_TRANSIENT_DIAGNOSTIC_REPORT.md`. Candidate selection used only the independent 30/20 calibration split.

Validation initial full NEES/DOF was `15.558045`. After the frozen 60% partition, attitude marginal NEES/DOF was `1.434813` and bias marginal was `2.744853`. The dominant settled source is therefore bias-side process/covariance understatement, with a separate large initial transient and material attitude-bias cross covariance; it is not a sensor-R mismatch.

`F-CALIBRATED-v1` is frozen at `{'s_P0_att': 2.0, 's_P0_bias': 4.0, 's_Qb': 8.0, 's_Qg': 2.0}` with all sensor R scales exactly one. Independent stationary and C4 confirmation each used N=50.

| Stationary settled policy | Full NEES/DOF | Attitude NEES/DOF | Bias NEES/DOF | Mag NIS/DOF | Sun NIS/DOF | ST NIS/DOF | Attitude RMSE (rad) | Bias RMSE (rad/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F-BASE | 1.418027 | 1.267383 | 1.832362 | 0.993057 | 0.975980 | 1.011273 | 0.000906979799 | 0.000113924368 |
| F-CALIBRATED-v1 | 1.020676 | 0.970833 | 1.311593 | 0.985289 | 0.970438 | 0.989938 | 0.000890025232 | 0.000121478792 |

The stationary paired mean change in absolute distance from NEES/DOF=1 was `-0.208037` with paired 95% bootstrap CI `[-0.30500210173950504, -0.11436263824653703]`. The mean relative attitude-bias P cross norm changed from `0.560534` to `0.467356`.

Stationary acceptance passed every predeclared guard, including N=50, strict P/S SPD, zero divergence, all three sensor NIS guards, accuracy, and the [0.8, 1.25] full-NEES target.

## Why the updated decision remains conditional

Stationary confirmation closes the named consistency target, but C4 does not pass every predeclared calibrated-policy guard. F-CALIBRATED-v1 whole-horizon bias RMSE was `0.010028002940591984` versus F-BASE `0.006343837012751137` (degradation `58.075%`). Its C4 settled normalized NIS was mag `1.7325503773256306`, sun `1.92057791368292`, and ST `4.0058636767354034`, outside the fixed stationary guard. These failures are retained rather than re-tuning Q/P0 or sensor R on confirmation data.

The C4 full-oracle cause-specific advantage, wrong-side ordering, zero divergence, and strict SPD evidence remain intact. Accordingly this is the contract's named `CONDITIONAL_GO` case: stationary covariance closure succeeds, while F-CALIBRATED-v1 must not replace F-BASE for C4.

## Frozen baseline matrix

- `F-BASE`: unchanged primary classical baseline.
- `F-TUNED=(0.125,0.125,8.0)`: unchanged sensitivity comparator only.
- `F-CALIBRATED-v1`: separate fixed P0/Q calibration; it does not replace   F-BASE or any oracle/wrong-side comparator.
- C4 process-only, measurement-only, full-oracle, and both wrong-side   diagnostics remain frozen comparators.

## Remaining limits and future scope

This decision covers only the deterministic representative-normalized classical benchmark. It does not establish orbit, WMM, eclipse, flight-sensor, universal calibration, learned-model, FPGA, or closed-loop performance.

A future Phase 2 design requires separate approval and must retain F-BASE, F-TUNED, F-CALIBRATED-v1, and the named classical/oracle/wrong-side matrix. No Phase 2 implementation was started in this study.
