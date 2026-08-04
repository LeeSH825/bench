# P1 Exit Covariance Closure Test Matrix

The closure tests cover:

- independent seed namespaces, closure-level 30/20 split, confirmation
  inaccessibility before freeze, and disjointness from frozen Phase 1 test IDs;
- fixed sensor R, truth/oracle-free calibrated replay, frozen F-BASE/F-TUNED,
  and immutable `F-CALIBRATED-v1` freeze identity;
- closed-form full, attitude-marginal, and bias-marginal NEES;
- strict-Cholesky whitened errors, per-coordinate/grouped energy,
  attitude-bias cross-correlation, and covariance cross-block metrics;
- exact initial/middle/settled partitions and a known settling-bin fixture;
- exact 20 coordinate plus 81 local-grid schedule, deterministic hierarchy and
  tie-break, validation-only selection, guard rejection, and candidate ledger;
- paired stationary/C4 N=50 completion, exact shared streams, sensor NIS and
  accuracy guards, full-oracle advantage, and wrong-side ordering;
- absence of inverse, pseudo-inverse, jitter, clipping, repair, sensor-R tuning,
  event-wise inflation, neural, FPGA, Phase 2, and closed-loop paths.

The final run also repeats every regression group required by Prompt 08 and
checks frozen artifact hashes plus entry/final dirty-tree fingerprints exactly.
