# P1 Exit Covariance Closure Contract

This study closes only the named posterior-consistency condition in the frozen
Phase 1 `CONDITIONAL_GO`. It does not alter any Phase 1 source, test, config,
result, manifest, baseline, sensor model, or sensor covariance.

`F-BASE` remains primary and `F-TUNED=(0.125,0.125,8.0)` remains a frozen
sensitivity comparator. The only candidate controls are scenario-wide fixed
`s_P0_att`, `s_P0_bias`, `s_Qg`, and `s_Qb`. Magnetometer, sun, and star-tracker
R scales are exactly one. No truth, oracle sidecar, event window, hidden label,
or future event is accepted by the calibrated fixed replay boundary.

The calibration pool uses seed namespace
`p1-exit-covariance-closure-independent-v1` and contains 30 whole-trajectory
train plus 20 whole-trajectory validation realizations. Confirmation data are
not generated or loaded before a candidate freeze record exists. After freeze,
a different master seed generates 50 stationary confirmation trajectories and
50 paired C4 trajectories. Every closure trajectory ID must be disjoint from
the frozen Phase 1 test IDs; calibration and confirmation IDs must be disjoint.

The time partitions are locked to initial `[0,0.2T)`, middle `[0.2T,0.6T)`, and
settled `[0.6T,T]`. Diagnostics use the Gate C right-local six-state error,
strict Cholesky solves, full/attitude/bias NEES, whitened-coordinate energy,
whitened cross-correlation, and the complete attitude-bias covariance block.
No inverse, pseudo-inverse, jitter, clipping, covariance repair, reported-P
rescaling, or event-dependent inflation is permitted.

Candidate selection uses validation summaries only after train/validation
diagnosis. The staged budget is 5+5 P0 coordinates, 5+5 Q coordinates, then an
81-point local combined grid. An interior coordinate winner uses its immediate
lower/current/upper values. If a winner is at a locked-grid endpoint, the axis
uses the nearest three values from the same locked grid; this deterministic
pre-search boundary rule preserves the 81-cell budget without introducing a
new scale. Hard guards require settled mag/sun/ST NIS in
`[0.8,1.25]`, no divergence or SPD failure, attitude RMSE degradation at most
5%, and bias RMSE degradation at most 10% relative to validation F-BASE.
Confirmation cannot change the candidate, objective, partitions, or guards.

The study ends with `P1_EXIT_REVIEW_UPDATED.md`. Phase 2, learned filters,
FPGA, and closed-loop implementation remain outside this contract.
