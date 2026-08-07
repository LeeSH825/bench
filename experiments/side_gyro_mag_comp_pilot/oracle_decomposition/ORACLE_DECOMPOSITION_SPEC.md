# Oracle Compensation Decomposition Diagnostic Specification

Status: frozen before any new oracle-decomposition arm was evaluated. This is a diagnostic extension of `side_gyro_mag_comp_pilot`, not a method-lock study and not authorization for Step 2.

Reviewed pilot commit: `ba312f10559935f7f60782771f8de64ab77787af`.  
Step 0 repair commit: `e622ca0e247b8a8e41d9f670eeb93f30a87d84a1`.  
Frozen pilot specification: `experiments/side_gyro_mag_comp_pilot/PILOT_SPEC.md` with SHA-256 `a10d1de963b27606e924657565716afbd7d85aea516f919a26875dd472c15410`.

## Questions and fixed scope

1. Reconstruct the committed four-arm C0/C1/N0/N1 comparison on R0-R3 without rerunning those arms.
2. Decompose combined oracle headroom into gyro-only, magnetometer-only, combined, and interaction effects.
3. Keep fixed classical-filter sensor interventions, fixed-N0-checkpoint neural sensor interventions, and separately trained N1 performance distinct.

Only the existing 30 test trajectories in each of R0-R3 are used. R4, N2, N3, N3S, learned-compensator training, and all Step 2 work are excluded. Dataset generation, split, timestamps, initial filter state, normalization, onboard magnetic reference, and raw realization digests remain those of the committed pilot.

## Checkpoint and normalization provenance frozen before evaluation

N0 and N1 do **not** use the same checkpoint. For each training seed they were trained separately with different sensor inputs: N0 used raw gyro/magnetometer packets; N1 used combined oracle gyro/magnetometer targets in the diagnostic training path. Their selected epochs are N0 `{31001: 12, 31002: 6, 31003: 14}` and N1 `{31001: 4, 31002: 4, 31003: 8}`. The checkpoint file SHA-256 values differ for every paired seed.

N0 and N1 use the same frozen normalization statistics and source trajectories. Every N0/N1 training record has normalization SHA-256 `8e1aa489985fb4089c9cf255e2eed8084e4daca2718aaf3d6f18002213fab501` and the same complete R0-R3 training-ID set. N1 is therefore a separately trained oracle-input arm, not a fixed-N0 intervention.

## Existing four-arm reconstruction

The committed `PER_TRAJECTORY_RECORDS.jsonl` is the only source for C0, C1, N0, and N1 values. Comparisons are:

- `C1 - C0` (combined oracle classical versus raw classical);
- `N1 - N0` (separately trained combined-oracle neural versus raw neural);
- `N0 - C0` (raw neural versus raw classical);
- `N1 - C1` (separately trained oracle neural versus oracle classical).

For N0 and N1, the three training-seed values are retained and averaged within trajectory before trajectory bootstrap; the direction of each seed-specific mean contrast is also reported. C0 and C1 use their deterministic trajectory values directly. Cross-backend comparisons average the neural seeds within trajectory and pair them with the deterministic classical value.

## New diagnostic intervention arms

Classical MEKF:

- `CG_ORACLE_GYRO_ONLY_MEKF`: oracle gyro target, raw magnetometer;
- `CM_ORACLE_MAG_ONLY_MEKF`: raw gyro, oracle magnetometer;
- `C1_ORACLE_GYRO_MAG_MEKF`: existing committed C1 combined-oracle semantics and records.

Fixed N0 checkpoint, separately for seeds 31001-31003:

- `NG_INT_ORACLE_GYRO_ONLY_SPLIT_KNET`: oracle gyro target, raw magnetometer;
- `NM_INT_ORACLE_MAG_ONLY_SPLIT_KNET`: raw gyro, oracle magnetometer;
- `NGM_INT_ORACLE_GYRO_MAG_SPLIT_KNET`: oracle gyro and magnetometer targets.

NG/NM/NGM load the exact committed N0 checkpoint for their seed. Only selected sensor values change. Normalization, model parameters, reset state, recurrent initialization, event order, timestamps, onboard constants, and target raw-realization provenance remain fixed. These are fixed-checkpoint causal sensor interventions, not oracle-trained models.

On R0, raw and oracle sensor vectors must be bit-identical. C0/CG/CM/C1 sensor interventions and N0/NG/NM/NGM within each N0 checkpoint must consequently be exact no-ops. Failure invalidates the diagnostic.

## Metrics and inference

For R0-R3, report:

- `attitude_geodesic_rmse_rad`;
- `residual_gyro_bias_rmse` (canonical record field `residual_gyro_bias_rmse_rad_s`);
- `corrected_gyro_rate_rmse_rad_s`;
- `integrated_gyro_increment_rmse_rad`;
- `corrected_magnetometer_angular_error_rad`;
- `weak_axis_rmse` (canonical record field `weak_axis_rmse_rad`);
- `observable_plane_rmse` (canonical record field `observable_plane_rmse_rad`);
- `divergence_count`.

All contrasts are paired by trajectory. Use 10,000 trajectory-clustered percentile bootstrap resamples with the existing seed `45173`. Comparison intervals use candidate minus reference, so negative means lower error. Effect intervals use the effect definitions below, so positive means improvement. No percentage or minimum-effect threshold is introduced.

For backend `B` and metric loss `L`, compute per trajectory before aggregation:

```text
E_G  = L_raw - L_gyro_oracle
E_M  = L_raw - L_mag_oracle
E_GM = L_raw - L_combined_oracle
I    = E_GM - E_G - E_M
```

Neural effects are computed within seed first, then averaged across the three seeds within trajectory for bootstrap. Each seed-specific mean effect direction is also reported. Classical effects use deterministic trajectories.

## Scoped diagnostic conclusion rule

The diagnostic status is determined only from R3 `attitude_geodesic_rmse_rad`. An effect is *resolved positive* when its paired 95% bootstrap interval has lower endpoint above zero.

- `MAG_DOMINANT_HEADROOM`: `E_M` is resolved positive for both backends and the interval for `E_M-E_G` is above zero for both.
- `GYRO_DOMINANT_HEADROOM`: `E_G` is resolved positive for both backends and the interval for `E_G-E_M` is above zero for both.
- `GYRO_AND_MAG_HEADROOM`: `E_G` and `E_M` are each resolved positive for both backends and neither dominance rule applies.
- `COMBINED_INTERACTION_ONLY`: neither isolated effect is resolved positive on either backend, while `E_GM` and `I` are resolved positive for both backends.
- `NO_RESOLVED_SENSOR_SPECIFIC_HEADROOM`: none of `E_G`, `E_M`, or `E_GM` is resolved positive on either backend.
- `INCONCLUSIVE_OR_IMPLEMENTATION_BLOCKED`: all other mixed-backend outcomes or any integrity/provenance failure.

Interaction is described as positive or negative overlap from `I`. It is not called causal synergy unless its paired interval excludes zero and the fixed intervention/checkpoint assumptions are stated. The final status cannot authorize Step 2.
