# Phase 1B Step 1 UNIT-ST Baseline Report

## Result

C1 matched stationary baseline, fixed tuning procedure, deliberate Q/R mismatch
and 600 s long-horizon subset all completed without divergence or SPD failure.
All-one ORACLE-QR is bit-exact to F-BASE and the frozen Phase 1A direct replay.

## Locked base profile

| Item | Value |
|---|---:|
| Truth | Basilisk spherical-inertia, zero torque |
| Pilot | 84 generated, 17/17/50 split, T=10 s |
| Gyro / ST | 10 Hz / 2 Hz, zero latency |
| Gyro base sigma | `5.0e-4 rad/s` |
| ST base R | `2.25e-6 I rad²` |
| Base Qg PSD | `2.5e-8 rad²/s` |
| Base Qb PSD | `1.0e-12` |
| Initial P diagonal | attitude `0.25²`, bias `0.01²` |
| Event evaluation window | `[0.4T,0.6T)` |

The profile is representative-normalized UNIT-ST and does not claim flight
sensor representativeness.

## Tuning

The staged train/validation-only budget was exactly 42 candidates: 5 Qg, 5 Qb,
5 R, then a 27-point local grid. The locked candidate grid was
`{0.25,0.5,1,2,4}`; all candidate objectives and trajectory IDs are in
`experiments/phase1b/results/unit_st_classical_v1/tuning.json`.

Selected fixed policy:

```text
F-TUNED: s_Qg=0.125, s_Qb=0.125, s_R=8.0
test split accessed during selection: false
```

The hierarchy prioritizes attitude RMSE before consistency. Consequently the
selected filter is deliberately reported as strongly conservative by NIS/NEES;
it is not described as consistency-optimal.

## C1 stationary test, N=50

Values are means across paired test trajectories.

| Policy | Event attitude RMSE (rad) | Bias vector RMSE (rad/s) | NIS norm. | NEES norm. | Divergence |
|---|---:|---:|---:|---:|---:|
| F-BASE | 1.6615e-3 | 2.0476e-3 | 0.924 | 0.967 | 0/50 |
| F-TUNED | 1.6469e-3 | 1.5752e-3 | 0.150 | 0.152 | 0/50 |
| F-MIS-Q-LOW | 1.6608e-3 | 2.0480e-3 | 0.929 | 1.070 | 0/50 |
| F-MIS-Q-HIGH | 1.6649e-3 | 2.0460e-3 | 0.904 | 0.784 | 0/50 |
| F-MIS-R-LOW | 1.6667e-3 | 2.1875e-3 | 3.499 | 3.105 | 0/50 |
| F-MIS-R-HIGH | 1.6544e-3 | 1.7394e-3 | 0.261 | 0.277 | 0/50 |
| ORACLE-QR all-one | 1.6615e-3 | 2.0476e-3 | 0.924 | 0.967 | 0/50 |

The R mismatch produces the expected strong consistency sensitivity. Qg
mismatch is weaker in this short stationary constant-rate profile. F-TUNED
slightly lowers short-horizon attitude and bias error but its very low NIS/NEES
is a material stationary consistency penalty.

## Long horizon, N=10, T=600 s

Each trajectory has 3,600 typed events and 600 ST updates.

| Policy | Event attitude RMSE mean (rad) | NIS norm. | NEES norm. | Divergence |
|---|---:|---:|---:|---:|
| F-BASE | 1.0600e-3 | 0.988 | 0.854 | 0/10 |
| F-TUNED | 1.9628e-3 | 0.186 | 2.252 | 0/10 |

F-BASE remained SPD and stable for 600 s. F-TUNED also did not diverge, but its
attitude RMSE was about 85% higher and its consistency diagnostics were poor.
This is a frozen-test finding; the policy was not retuned after observing it.

Evidence: `pilot_summary.json`, `long_horizon.json`, `tuning.json` under
`experiments/phase1b/results/unit_st_classical_v1`.
