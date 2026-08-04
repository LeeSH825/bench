# Phase 1 Exit Criteria

Locked before Phase 1B Step 2 scientific runs on 2026-08-02.

## Required evidence

- The approved Phase 1A foundation and Phase 1B Step 1 UNIT-ST C1/C2/C3/C5
  results remain byte-preserved and their regressions pass.
- `F-BASE` is the primary classical baseline. `F-TUNED` remains only the frozen
  sensitivity comparator with `(s_Qg,s_Qb,s_R_ST)=(0.125,0.125,8.0)`.
- MAIN-FUSION stationary, STRESS-MAG, and C4 each complete 50 paired held-out
  trajectories. Sensor ablations complete at least 20 paired trajectories.
- MAIN-FUSION stationary has no numerical divergence under `F-BASE`, has strict
  SPD posterior/innovation covariance, and the all-one oracle is exactly equal
  to the fixed path.
- STRESS-MAG reports both observable-plane and magnetic-axis attitude error; it
  does not claim full instantaneous attitude observability.
- Magnetometer and valid-sun NIS are available with 3 and 2 degrees of freedom;
  invalid sun rows are counted as skips and are excluded from NIS.
- C4 changes gyro-bias random-walk intensity in `[0.2T,0.8T)` and only
  magnetometer inlier covariance in `[0.45T,0.6T)`. Same-realization, unaffected
  stream equality, truth-free fixed APIs, and forward-only oracle access pass.
- Process-only, measurement-only, full-oracle, and both wrong-side mappings are
  evaluated. Oracle benefit and limitations are reported whether or not every
  directional hypothesis is supported.
- A practical oracle effect is predeclared as at least a 2% paired improvement
  on its cause-specific primary metric, with the paired 95% bootstrap interval
  also reported. This threshold is interpretive and is never used to alter data,
  tolerances, scales, or policy definitions after seeing test results.
- All Phase 1A, Phase 1B Step 1, smoke/cache, and legacy regressions pass, and
  allowlist-external dirty-tree content is unchanged.

## Decision rule

- `GO`: all mandatory evidence is complete and the classical baseline plus
  process/measurement separation evidence is sufficient for a later Phase 2
  design step.
- `CONDITIONAL_GO`: infrastructure and primary evidence are sufficient, but a
  named classical limitation must remain a mandatory future baseline/ablation.
- `STOP`: classical instability, no useful oracle effect in the primary
  problems, information leakage, same-realization failure, incomplete N=50
  primary pilots, regression failure, or a material sensor/model defect.

This document authorizes no Phase 2 implementation. Flight-orbit, WMM,
high-fidelity eclipse, product-level sensor accuracy, general identifiability,
and single-vector full-observability claims are outside this decision.
