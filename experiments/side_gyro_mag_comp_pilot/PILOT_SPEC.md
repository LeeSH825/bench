# Gyro-Magnetometer Compensation-Conditioned Split-MEKF-KalmanNet Pilot

Status: preregistered before any pilot test, smoke, training, or held-out evaluation.  
Base: `phase0-1/classical-foundation` at `052d2f7217b964b1fa4e80bd643716b433780f08`.  
Study namespace: `side_gyro_mag_comp_pilot`.

## Questions and sequential stops

- **G0:** Does oracle gyro/magnetometer compensation provide downstream headroom?
- **G1:** Does learned compensation improve sensor and state-estimation performance?
- **G2:** Do compensation features improve over corrected values alone?
- **G3:** Does trajectory-shuffling the features remove that additional benefit?
- **G4:** Is the proposed model harmless under nominal sensor conditions?

Execution is strictly `sanity -> six integrity checks -> smoke -> G0 -> G1 -> G2 -> G3 -> G4`. G0 failure stops all learned experiments. G1 failure stops the feature claim. No held-out result may change the architecture, features, optimizer, losses, epochs, learning rate, populations, metrics, or thresholds below.

## Regimes, populations, and pairing

Regimes are `R0_NOMINAL`, `R1_GYRO_BIAS_SCALE`, `R2_MAG_HARD_SOFT_IRON`, `R3_COMBINED`, and `R4_COMBINED_OOD`. R0-R3 supply training and validation trajectories; all five supply test trajectories. R4 parameter support is disjoint from R0-R3 training and validation support.

- Smoke: train 4, validation 2, and test 4 trajectories per regime; seed `31001`.
- Pilot: train 40, validation 10, and test 30 trajectories per regime; seeds `31001`, `31002`, `31003`.
- Generation seed: `271828`; split seed: `314159`; step `0.1 s`; sequence length `16`.
- Splits are whole-trajectory, unique, nonempty, and disjoint.
- Every compared variant uses identical raw gyro/magnetometer packets, timestamps, validity, truth trajectory, initial state, and realization digest for a given trajectory.
- Normalization and checkpoint selection use only R0-R3 training/validation trajectories. Test trajectories and R4 never influence training, normalization, stopping, or checkpoint selection.

## Variants

- `C0_RAW_MEKF`
- `C1_ORACLE_COMP_MEKF`
- `N0_RAW_SPLIT_KNET`
- `N1_ORACLE_COMP_SPLIT_KNET`
- `N2_LEARNED_COMP_ONLY_SPLIT_KNET`
- `N3_LEARNED_COMP_FEATURE_SPLIT_KNET`
- `N3S`: the exact N3 checkpoint and corrected values evaluated with one fixed-point-free whole-trajectory feature-sequence derangement per regime and seed. Only feature association changes.

C0/C1 are descriptive system references. Gates use the N variants specified below.

## Frozen model and training

The nominal state is scalar-first Hamilton active `q_NB` plus residual body-frame gyro bias. The error state is right-local 6D. Gyro is a causal propagation input; magnetometer is the vector measurement. The magnetometer gain is `6x3`, correction is right-multiplicative, and deterministic compensation remains distinct from the residual bias state.

Gyro and magnetometer use separate causal GRU encoders. Each emits one corrected 3-vector and one 8D feature. Gyro features FiLM only the prior/propagation Split branch (`G1`); magnetometer features FiLM only the innovation/measurement branch (`G2`). `K = G1 H^T G2` has shape `6x3`. Feature-off is exact identity. Attention and Transformer mechanisms are excluded.

Frozen sizes and training settings: float64; encoder hidden width 16; Split prior/measurement widths 32/32; Adam; learning rate `0.001`; weight decay `0`; trajectory batch 8; maximum 20 epochs; smoke 2 epochs; patience 4; gradient clip 1.0. Loss weights are corrected gyro `1.0`, corrected magnetometer `1.0`, downstream attitude `1.0`, residual bias `0.25`. The checkpoint is the earliest epoch attaining the minimum R0-R3 validation attitude RMSE.

## Metrics

All metrics are per whole trajectory before population aggregation.

- Attitude RMSE: `sqrt(mean(phi_t^2))`, where `phi_t` is shortest quaternion geodesic error.
- Corrected gyro-rate RMSE: `sqrt(mean(||omega_corrected_t - omega_oracle_target_t||^2))`; the oracle target retains residual bias.
- Integrated gyro attitude-increment error: with `S_t = sum_{u<=t}(omega_corrected_u - omega_oracle_target_u) dt_u`, use `sqrt(mean(||S_t||^2))`.
- Corrected magnetometer angular error: `mean(atan2(||u_t x v_t||, u_t dot v_t))` for normalized corrected and oracle-target vectors.
- Residual gyro-bias RMSE: `sqrt(mean(||b_hat_t - b_true_t||^2))`.
- Attitude p95: linear 95th percentile of `phi_t`.
- Divergence: any non-finite state/required metric or `max(phi_t) > 1.0 rad`.

For every valid magnetometer update, using the posterior estimate at that update,

```text
e_theta = Log(q_hat_NB^{-1} ⊗ q_true_NB)
u_m = m_true_B / ||m_true_B||
e_weak = u_m^T e_theta
e_plane = (I - u_m u_m^T)e_theta
```

The per-trajectory weak-axis RMSE is `sqrt(mean(e_weak^2))`; observable-plane RMSE is `sqrt(mean(||e_plane||_2^2))`. Every valid sample contributes to both. Every declared test trajectory is included. Zero valid magnetometer samples, a missing/duplicate update, zero/non-finite true-field norm, or a non-finite result invalidates the dataset. These two metrics are descriptive only and never control training, gate entry, or stopping.

The primary endpoint is trajectory attitude RMSE on `R4_COMBINED_OOD`. Required secondary metrics are all remaining metrics above.

## Paired inference

Gate comparisons use candidate-minus-reference contrasts. For pilot gates, each of the 30 trajectory IDs carries all three seed values as one bootstrap cluster. Use 10,000 trajectory-clustered percentile bootstrap resamples with seed `45173`; the 95% interval is the 2.5/97.5 percentile interval. Missing, duplicate, unpaired, non-finite, or non-positive reference-denominator populations invalidate the gate rather than being dropped or imputed.

## Frozen gates and final mapping

- **G0:** On R3, N1 reduces mean attitude RMSE versus N0 by at least 10%, and the paired CI for `N1-N0` has upper endpoint `< 0`. Failure: `STOP_NO_COMPENSATION_HEADROOM`.
- **G1:** N2 strictly improves corrected gyro-rate RMSE and integrated increment error versus N0 on R1, strictly improves magnetometer angular error on R2, and reduces R3 attitude RMSE versus N0 by at least 5%; the R3 paired CI upper endpoint is `< 0`, with negative mean contrast in at least two of three seeds. Failure: `REJECT_LEARNED_COMPENSATION`.
- **G2:** On the R4 primary endpoint, N3 reduces mean attitude RMSE versus N2 by at least 5%; the paired CI upper endpoint is `< 0`, with negative mean contrast in at least two of three seeds. Failure: `LOCK_COMPENSATION_ONLY_REJECT_FEATURE_PATH`.
- **G3:** Let feature gain be `mean(N2)-mean(N3)` and shuffled loss be `mean(N3S)-mean(N3)` on R4 attitude RMSE. Pass if shuffled loss is at least half the positive G2 feature gain, **or** the paired 95% CI for `N3S-N2` includes zero. Failure: `LOCK_COMPENSATION_ONLY_REJECT_FEATURE_PATH`.
- **G4:** On R0, `(mean(N3)-mean(N0))/mean(N0) <= 0.03`, and N3 adds no divergence for any seed. Failure: `LOCK_COMPENSATION_ONLY_REJECT_FEATURE_PATH`.

All-pass decision: `LOCK_COMPENSATION_CONDITIONED_SPLIT_MEKF_KALMANNET`. Real implementation, data, or execution failure maps to `BLOCKED_REAL_IMPLEMENTATION_OR_DATA_ERROR`.

## Mandatory integrity checks

Exactly six direct integrity checks are required, each with one failing negative fixture: whole-trajectory split disjointness; identical raw realization; runtime leakage/future-sample exclusion; right-error/6x3 gain/multiplicative injection; N3S exact checkpoint with feature-association-only intervention; and feature-off exact corrected-only equivalence. No general contract-mutation framework is permitted.

## Exclusions

No SNN, SoW, reliability gate, attention, Transformer, learned Q/R, uncertainty head, extra estimator sensor, temperature/vibration/outlier/saturation/MTQ regime, broad KalmanNet comparison, hyperparameter sweep, FPGA study, or closed-loop control.
