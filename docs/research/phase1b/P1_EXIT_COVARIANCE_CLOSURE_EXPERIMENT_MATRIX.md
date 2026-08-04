# P1 Exit Covariance Closure Experiment Matrix

| Stage | Data visible | Policies/candidates | N | Purpose |
|---|---|---|---:|---|
| Diagnose | independent calibration train | F-BASE | 30 | time-resolved transient, block, whitened, and cross-covariance diagnosis |
| Diagnose | independent calibration validation | F-BASE | 20 | independent baseline and predeclared settling diagnostic |
| Search 1 | validation only | 5 P0-att then 5 P0-bias coordinates | 20 per candidate | transient objective |
| Search 2 | validation only | 5 Qg then 5 Qb coordinates | 20 per candidate | settled objective and guards |
| Search 3 | validation only | locked 3^4 local grid | 20 per candidate | final deterministic selection |
| Confirmation | new stationary confirmation | F-BASE, F-TUNED, F-CALIBRATED-v1 | 50 paired | held-out state/sensor consistency and accuracy |
| C4 confirmation | paired new C4 confirmation | frozen matrix plus F-CALIBRATED-v1 | 50 paired | accuracy, oracle advantage, and wrong-side ordering |

The confirmation master seed is not instantiated by `diagnose` or `search`.
`confirm` refuses to run without an immutable freeze record linked to the
complete search manifest. The C4 physical stream is generated only after that
freeze and reuses the stationary confirmation rigid-body/base-sensor
realization so that intervention pairing is exact.
