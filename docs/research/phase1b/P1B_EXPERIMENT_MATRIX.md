# Phase 1B Step 1 Experiment Matrix

Locked common pilot profile: Basilisk spherical-inertia/zero-torque truth,
`T=10 s`, gyro 10 Hz, ST 2 Hz, event `[0.4T,0.6T)`, paired test `N=50`,
all ST valid/inlier. `alpha` is a covariance multiplier.

| ID | Changed variable | Fixed/paired variables | Estimators | Test N | Duration | Primary interpretation |
|---|---|---|---|---:|---:|---|
| C1 | none; all alpha=1 | all streams | base, tuned, four mismatch, all-one oracle | 50 | 10 s | matched stability and mismatch sensitivity |
| C2-MILD | alpha_g=2 | truth/base noise/ST/timing | base, tuned, oracle, wrong-side | 50 | 10 s | mild process step |
| C2-MED | alpha_g=4 | same | same | 50 | 10 s | medium process step and C5-A |
| C2-SEV | alpha_g=8 | same | same | 50 | 10 s | process trend and consistency |
| C3-MILD | alpha_R=2 | truth/base noise/gyro/timing | base, tuned, oracle, wrong-side | 50 | 10 s | mild inlier reliability step |
| C3-MED | alpha_R=4 | same | same | 50 | 10 s | medium reliability step |
| C3-SEV | alpha_R=8 | same | same | 50 | 10 s | measurement over-trust and oracle bound |
| C5-A | alpha_g=4, alpha_R=1 | paired truth/timing/unaffected streams | base, tuned, oracle, wrong-side | 50 | 10 s | process-origin member of RMS pair |
| C5-B | alpha_g=1, alpha_R=1.08 | paired truth/timing/unaffected streams | base, tuned, oracle, wrong-side | 50 | 10 s | measurement-origin member, alpha frozen on val N=17 |
| C1-LONG | none | stationary profile | base, tuned | 10 | 600 s | long-horizon stability/penalty |

All rows report attitude geodesic RMSE/P95/peak, bias vector RMSE, recovery,
divergence, NIS, NEES and P/S SPD. C5 additionally reports innovation RMS/norm
P95/autocorrelation and raw gyro measurement/increment RMS. Paired comparisons
use deterministic 2,000-resample bootstrap confidence intervals.

The pilot generator creates 84 whole trajectories with a 17/17/50
train/validation/test split. Tuning uses a separate stationary 9/3/3 split;
C5 RMS selection uses the pilot validation 17 only. Test 50 is untouched until
both the fixed policy and C5-B alpha are frozen.
