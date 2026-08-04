# P1 Exit Review

Decision: **CONDITIONAL_GO**

Phase 1A foundation, UNIT-ST Step 1, MAIN-FUSION stationary, STRESS-MAG, C4,
N=50 paired primary pilots, sensor-specific consistency, same-realization and
information-boundary checks, regressions, and dirty-tree integrity are
complete. This review authorizes only a future separately approved Phase 2
design step; no Phase 2 implementation was started here.

## Evidence supporting progress

- F-BASE had zero divergence in all three N=50 primary conditions.
- MAIN-FUSION all-one oracle reduced exactly to F-BASE; four-sensor ordering,
  invalid sun skips, and q/-q invariance passed.
- Settled MAIN-FUSION normalized sensor NIS was mag `1.023`, sun `1.000`, and ST
  `1.092`. Strict SPD evidence exists for every actual update.
- STRESS-MAG exposed the expected magnetic-axis weak direction rather than
  hiding it.
- C4 full oracle improved both slow process and fast measurement outcomes with
  paired 95% intervals excluding zero. Partial and wrong-side results supplied
  cause-specific separation evidence and important limitations.
- Raw physical streams and simulation-only oracle labels have separate schemas,
  files, manifests, hashes, and APIs. Fixed/tuned estimators never see oracle
  context, event windows, hidden labels, future events, or truth.
- All frozen regressions and exact Step 1 result recomputation passed.

## Why the decision is conditional

F-BASE remains overconfident at the posterior-state level after the 20% initial
transient: settled NEES/DOF is `1.873`, although sensor NIS is close to one.
Whole-horizon NIS/NEES is further inflated by randomized initial convergence.
F-TUNED makes stationary fast peak, slow bias, and consistency worse and must
remain a sensitivity comparator. Measurement-only oracle improves mag NIS but
not the chosen state-accuracy metric, and process-only worsens the fast peak;
only the full action resolves both sides in this regime. These named classical
limitations must remain mandatory future baselines/ablations.

## Frozen baselines and remaining classical work

- Primary: `F-BASE`.
- Frozen comparator: `F-TUNED=(0.125,0.125,8.0)`; never retune on these test
  results.
- Frozen C5: `alpha_R_ST=1.08` and all Step 1 streams/results/manifests.
- Retain stationary covariance-calibration analysis, initialization/transient
  separation, F-TUNED sensitivity, process-only/measurement-only/full oracle,
  and both wrong-side diagnostics in any future comparison.

## Unsupported and forbidden claims

The work does not establish flight-orbit, WMM, eclipse, or product-sensor
fidelity; full attitude observability from one magnetic vector; universal
process/measurement identifiability; learned-context usefulness; neural
superiority; FPGA suitability; or closed-loop performance. Favorable
mag/sun geometry is disclosed.

## Future scope requiring separate approval

A later Phase 2 may design classical-matched neural baselines while preserving
the frozen streams, boundaries, and comparator matrix above. KalmanNet,
Split-KalmanNet, ANN, SNN, FPGA, and closed-loop code were not created,
imported, trained, or executed in this step.
