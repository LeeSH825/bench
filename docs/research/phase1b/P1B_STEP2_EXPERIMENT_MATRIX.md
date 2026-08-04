# Phase 1B Step 2 Experiment Matrix

| ID | Sensors | Event | Policies | Held-out N | Primary interpretation |
|---|---|---|---|---:|---|
| MAIN-STATIONARY | gyro+mag+sun+ST | none | F-BASE, F-TUNED, all-one oracle | 50 | matched multirate stability and consistency |
| STRESS-MAG | gyro+mag | none | F-BASE-MAG, matched oracle | 50 | single-vector weak-axis characterization |
| MAIN-ABLATIONS | fixed paired full stream subsets | none | F-BASE | 20 | sensor contribution sanity, not tuning |
| C4 | gyro+mag+sun+ST | slow `alpha_b`, fast `alpha_R_mag` | F-BASE, F-TUNED, process-only, measurement-only, full, two wrong-side diagnostics | 50 | cause-specific action and combined-event evidence |

All primary trajectories are 30 s at gyro/mag/sun/ST rates 10/5/2/1 Hz.
Trajectory splits are whole and fixed at 20/20/60 percent; no test trajectory is
used for tuning. Every policy consumes the same pre-generated raw stream and
the same initial state.
