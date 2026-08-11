# SC-01 Minimal Hypothesis and Gate Contract

Status: frozen before test access.

## Regimes and split

- R0: nominal.
- R1: gyro constant deterministic offset plus diagonal scale, with residual stochastic bias retained.
- R2: magnetometer hard iron plus positive-definite near-identity soft iron.
- R3: combined R1+R2 using in-support parameter ranges.
- R4: combined held-out parameter ranges disjoint from R1-R3 support.

All component magnitudes below use independent fixed signs drawn before generation; intervals exclude zero. Gyro offsets are in rad/s. Magnetometer hard iron is normalized by `||m_N||`; scale quantities are dimensionless.

| Regime | `|c_g,i|` | `|diag(A_g)-1|` | `|b_m,i|/||m_N||` | soft-iron eigenvalue deviation `|lambda(A_m)-1|` |
|---|---:|---:|---:|---:|
| R0 | 0 | 0 | 0 | 0 |
| R1 | `[2e-4,8e-4]` | `[0.005,0.015]` | 0 | 0 |
| R2 | 0 | 0 | `[0.02,0.06]` | `[0.02,0.06]` |
| R3 | `[2e-4,8e-4]` | `[0.005,0.015]` | `[0.02,0.06]` | `[0.02,0.06]` |
| R4 | `[1.2e-3,1.8e-3]` | `[0.025,0.040]` | `[0.10,0.16]` | `[0.10,0.16]` |

The nearest R4 boundary is separated from in-support calibration magnitudes by at least `4e-4 rad/s` for offset, `0.010` for gyro scale, and `0.04` for magnetic quantities. `A_m` is generated as `U diag(lambda) U^T`, remains SPD, and obeys SC-00 conditioning bounds.

### Gauge fixing and identifiability

Within a regime, `(A_g,c_g)` is drawn once and shared by every trajectory and split containing that regime. Residual initial bias is drawn independently per trajectory as `b_r(0)~N(0,(2e-5 rad/s)^2 I)` and follows a zero-mean random walk. Thus `E[b_r]=0`; the minimum R1/R3 offset magnitude exceeds five residual-bias standard deviations. Every trajectory must satisfy the excitation certificate `lambda_min(sum_t (omega_true_t-mean)(.)^T)/T) >= 1e-5 (rad/s)^2`, with all three axes nonzero. Without these anchors, `(c_g,b_r)->(c_g+A_g C_SgB v,b_r-v)` is an exact observational symmetry and the target is defined only up to a constant. Since R4 calibration lies outside training support, R4 gyro corrected-rate and residual-bias metrics are reported jointly as diagnostics with no per-component calibration claim; they are not G1 sensor predicates.

Whole trajectory IDs are disjoint. Training/validation/normalization use R0-R3 only. R4 is test-only and may not affect training, early stopping, normalization, thresholds, or selection. R4 initial-state, orbit-phase, and raw-noise RNG namespaces are disjoint from R0-R3 train/validation namespaces and are recorded per trajectory. Raw realizations and trajectory IDs are paired across compared variants.

Smoke populations are train 4 and validation 2 per R0-R3, test 4 per R0-R4. Method-lock populations are train 40 and validation 10 per R0-R3, test 30 per R0-R4, with fixed training seeds `[31001, 31002, 31003]`.

## Models

- C0: classical right-error MEKF with raw gyro/mag and no oracle compensation.
- C1: the same classical MEKF with diagnostic oracle deterministic compensation.
- N0: unconditioned right-error Split backbone, raw values, FiLM exactly off.
- N1: N0 with oracle-corrected values, FiLM off.
- N2: learned corrected values, FiLM off.
- N3: N2 corrected values plus 8D branch-specific FiLM features.
- N3S: evaluation-only N3 using the identical checkpoint and corrected values. One fixed-point-free permutation is created per `(regime, training_seed)` and applied identically at every time index, reassigning whole feature sequences only. A stratum with fewer than two trajectories is a hard error; every stratum records fixed-point count exactly zero. Corrected values, raw realization, timestamps, states, and recurrent histories are not shuffled.

C0/C1 are diagnostic shell references. G0-G4 require only N0-N3S.

## Metrics and inference

The primary endpoint is trajectory-level mean attitude geodesic RMSE. G0 evaluates this same metric on R3; the declared primary method-lock endpoint is its R4 value. Required secondary outputs are gyro corrected-rate RMSE against the residual-bias-retaining target, gyro integrated-increment error, magnetometer corrected-vector angular error, residual gyro-bias RMSE, p95 attitude error, divergence count, magnetic-axis weak error, and observable-plane error. Populations must be non-empty and finite.

Paired bootstrap uses the unique test `trajectory_id` as its cluster: a sampled ID carries all three per-seed values as a block. The statistic first averages the paired contrast over seeds within ID and then over sampled IDs. It uses 10,000 resamples, percentile 95% intervals, and seed `45173`. A CI is below zero only when its upper endpoint is `<0`.

Every record key is `{experiment,regime,model,window,metric,seed,trajectory_id}`. Point estimates for G0-G4 are seed means. For G1/G2, the two-of-three rule is the sign of each seed's mean paired difference over its 30 R4 IDs. G4's 3% ratio is evaluated on the seed-mean aggregate and its no-added-divergence condition must hold in every seed. A trajectory diverges if any estimate/metric is non-finite or if maximum geodesic attitude error exceeds `1.0 rad`.

All contrasts are `candidate-reference`, so improvement is negative. N0-N3 use the same maximum epochs and the same validation-only early-stopping rule: select the earliest epoch attaining the minimum R0-R3 validation attitude RMSE, breaking ties by lower epoch; test/R4 is never consulted.

## Frozen gates

- G0, oracle headroom: on R3, N1 reduces primary RMSE versus N0 by at least 10%, and the paired CI for `N1-N0` is below zero.
- G1, learned compensation: versus N0, N2 strictly reduces both R1 gyro corrected-rate RMSE and integrated-increment error, strictly reduces R2 magnetometer angular error, and reduces R4 primary RMSE by at least 5% with CI below zero; the R4 improvement direction must hold for at least two of three seeds.
- G2, feature increment: on R4, N3 reduces primary RMSE versus N2 by at least 5% with CI below zero, with the same direction in at least two of three seeds.
- G3, association falsification: define per-trajectory `T = RMSE_N3S - 0.5 RMSE_N2 - 0.5 RMSE_N3`. Pass only when the paired 95% CI lower endpoint for `T` is `>0`. This is exactly the frozen `L>=0.5D` margin with uncertainty, where `D=RMSE_N2-RMSE_N3` and `L=RMSE_N3S-RMSE_N3`; there is no disjunct. A CI crossing zero is `INCONCLUSIVE_UNDERPOWERED`, a terminal blocked evidence outcome distinct from substantive `REVISE_FEATURE_INTERFACE`.
- G4, nominal harmlessness: on R0, `(RMSE_N3-RMSE_N0)/RMSE_N0 <= 0.03` and N3 adds no divergence relative to N0.

Gate order is G0, G1, G2, G3, G4. A failed gate stops unnecessary later work and uses the charter's frozen decision mapping. No threshold, architecture, feature dimension, support, split, learning rate, epoch count, or loss weight changes after test access; no rescue experiment is permitted.

## Hard exclusions

SNN, SoW, reliability gating, attention, Transformer, learned Q/R, uncertainty heads, temperature, vibration, outlier/saturation/MTQ regimes, extra runtime sensors, closed-loop, FPGA, automated sweeps, broad comparisons, test-driven tuning, and publication optimization remain excluded.
