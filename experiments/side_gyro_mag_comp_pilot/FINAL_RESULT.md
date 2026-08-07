# Final Gyro-Magnetometer Compensation Pilot Result

Final decision: `REJECT_LEARNED_COMPENSATION`

The pilot established real compensation headroom, but the learned compensation path failed its frozen sensor-level requirements. The dependent feature experiments were therefore not run.

## Executed sequence

- Canonical MEKF sanity: 98 tests passed.
- Six mandatory integrity checks: 6 strengthened production-path checks passed; each focused negative fixture was rejected for the invariant it exercises.
- Tiny smoke: passed for all five regimes and all declared variants; smoke values are not used as scientific evidence.
- G0: three N0/N1 training seeds and 30 paired R3 test trajectories per seed.
- G1: three N2 training seeds and the frozen R1, R2, and R3 paired comparisons.
- G2, G3, and G4: not run because G1 did not authorize the feature claim.

## Gate results

- **G0 PASS.** Mean R3 attitude RMSE fell from `0.0796075 rad` for N0 to `0.0623591 rad` for N1, a `21.67%` reduction. The trajectory-clustered paired 95% CI for N1 minus N0 was `[-0.0240839, -0.0108136] rad`; all three seed directions agreed.
- **G1 FAIL.** Mean R3 attitude RMSE fell from `0.0796075 rad` to `0.0700778 rad`, an `11.97%` reduction, with CI `[-0.0140758, -0.00493016] rad` and all three seed directions agreeing. Magnetometer angular error also improved from `0.0501634 rad` to `0.0181321 rad`. However, corrected gyro-rate RMSE worsened from `0.00102577` to `0.00472986 rad/s`, and integrated gyro-increment RMSE worsened from `0.000891390` to `0.00239241 rad`. The frozen G1 conjunction therefore failed.
- **G2 NOT_RUN.** Not authorized after G1 failure.
- **G3 NOT_RUN.** Not authorized after G1 failure.
- **G4 NOT_RUN.** Not authorized after G1 failure.

## Interpretation

Oracle gyro/magnetometer correction has meaningful downstream headroom in this frozen synthetic setting. The implemented learned path did not recover the gyro correction accurately enough to satisfy the preregistered compensation gate, despite downstream attitude and magnetometer improvements. The result therefore rejects learned compensation and makes no incremental compensation-feature claim.

This result is limited to the preregistered synthetic pilot. It does not establish calibrated covariance, flight performance, hardware efficiency, or generality beyond the declared regimes. Weak-axis and observable-plane metrics are descriptive only; every valid magnetometer sample and every declared test trajectory is retained in the machine records.

The trajectories contain 16 samples over a 1.6 s horizon. The preregistered R4 OOD primary endpoint remained ungated because G2 was not authorized after G1 failed. One calibration vector is shared by every trajectory within each regime, so sensor-parameter identity is not split and these results do not support generalization to new hardware. Checkpoint selection minimizes validation attitude RMSE, whereas G1 also judges sensor-level gyro accuracy; that objective/selection distinction is material to the observed G1 failure.
