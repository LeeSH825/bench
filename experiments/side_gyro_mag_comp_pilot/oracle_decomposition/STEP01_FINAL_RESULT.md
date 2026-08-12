# Step 0-1 Final Result

Step 0 verdict: `PASS`. Six strengthened integrity checks and all 98 canonical MEKF regression tests passed. G0 and G1 were not rerun.

Step 1 diagnostic conclusion: `MAG_DOMINANT_HEADROOM`. Step 2 is not authorized or started.

## Existing four-arm evidence

The complete 128-cell C0/C1/N0/N1 comparison is in `EXISTING_FOUR_ARM_COMPARISON.json` and `ORACLE_DECOMPOSITION_REPORT.md`; it was reconstructed from committed records without rerunning any old arm. On R3 attitude RMSE:

- C0 raw MEKF: `0.106510527 rad`; C1 combined-oracle MEKF: `0.0624108450 rad`.
- N0 raw Split-KalmanNet: `0.0796075044 rad`; N1 separately trained combined-oracle Split-KalmanNet: `0.0623590856 rad`.

N0 and N1 use different per-seed checkpoint files and state dictionaries, selected at different epochs after training on raw versus combined-oracle sensor inputs. They use the same normalization digest and source trajectory IDs. N1 is not a fixed-N0 sensor intervention.

## R3 attitude decomposition

Classical fixed MEKF:

- Gyro-only effect `E_G = +0.000256637 rad`, 95% CI `[+0.0000304706, +0.000484572]`.
- Magnetometer-only effect `E_M = +0.0439310941 rad`, CI `[+0.0390950002, +0.0488187348]`.
- Combined effect `E_GM = +0.0440996825 rad`, CI `[+0.0393007303, +0.0489572665]`.
- Interaction `I = -0.0000880487 rad`, CI `[-0.000235119, +0.0000521439]`; unresolved and consistent with near-additivity/slight overlap.

Fixed N0 checkpoints, averaged within trajectory across seeds before bootstrap:

- Gyro-only effect `E_G = +0.000202705 rad`, CI `[-0.00000891146, +0.000415122]`; positive mean in 3/3 seeds but not resolved.
- Magnetometer-only effect `E_M = +0.0114127598 rad`, CI `[+0.00576630716, +0.0167328192]`; positive in 3/3 seeds.
- Combined effect `E_GM = +0.0115063702 rad`, CI `[+0.00585337752, +0.0168090999]`; positive in 3/3 seeds.
- Interaction `I = -0.000109094 rad`, CI `[-0.000180380, -0.0000421727]`; resolved overlap/redundancy, not synergy.

All R0 intervention arms were exact no-ops. Full R0-R3 effects for all eight required metrics, paired intervals, and seed directions are in `ORACLE_DECOMPOSITION_SUMMARY.json`. The conclusion is diagnostic only and does not reverse the pilot's `REJECT_LEARNED_COMPENSATION` decision or authorize learned-compensator redesign.
