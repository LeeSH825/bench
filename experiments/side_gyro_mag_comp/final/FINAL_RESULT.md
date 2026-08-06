# Side Gyro-Magnetometer Compensation Study: Final Result

Final decision: `BLOCKED_IMPLEMENTATION_OR_INTEGRITY`

Terminal reason: `BLOCKED_CONTRACT_GAP_R8_WEAK_AXIS_POPULATION_DEFINITION`

The study did not advance beyond the single implementation-repair round. The
independent implementation audit found that 18 claimed invariants were not
connected to red-capable tests. Its one-round counterproposal required, among
other repairs, replacing the degenerate weak/observable population counters
with a **frozen-threshold membership definition**. No threshold or membership
formula exists in the canonical SC-00 contract, SC-01 gate contract, side-study
configuration, implementation audit, or repair counterproposal. Choosing one
inside the final repair round would invent evaluation semantics after the DR0
contract was frozen. The sole implementation writer therefore stopped without
editing the repaired implementation checkpoint.

The independent R9 clarification supplied a conservative pre-test lock for the
otherwise underspecified sensor-metric aggregations. It did not define or
authorize the separate R8 weak-axis population threshold. Because every item in
the implementation counterproposal is mandatory and the implementation repair
round is already consumed, the unresolved R8 contract gap is terminal. No
second implementation repair and no experiment execution is permitted.

## Decisive predicates

| Axis | Result | Decisive predicate and evidence |
|---|---|---|
| Mathematical validity | `UNKNOWN` | DR0 passed after one repair, and the implementation audit independently re-derived the frozen right-error equations as correct. However, the differentiable training operator was not red-path-equivalent to runtime, so the implemented math did not receive implementation PASS. |
| Code and information boundary | `FAIL` | The implementation audit classified CF-1 and CF-2 as not fully satisfied: deployable enforcement and executed split-ID provenance could be removed or redirected while the 42-test suite remained green. |
| Statistical evidence | `UNKNOWN` | The method-lock pilot was not run. There are no G0-G4 estimates, confidence intervals, or performance conclusions. |
| Scope proportionality | `PASS` | The audit found no scope creep and confirmed the excluded architectures, sensors, and comparisons were absent. |
| Governance and integrity | `FAIL` | The one authorized implementation repair cannot satisfy R8 without inventing a frozen population definition. Failing closed preserves the pre-test evidence boundary. |

## G0-G4 disposition

| Gate | Result | Scientific conclusion |
|---|---|---|
| G0 oracle compensation headroom | `NOT_RUN` | Unknown. |
| G1 learned compensation | `NOT_RUN` | Unknown. |
| G2 feature increment | `NOT_RUN` | Unknown. |
| G3 association falsification | `NOT_RUN` | Unknown. |
| G4 nominal harmlessness | `NOT_RUN` | Unknown. |

Accordingly, this result does **not** establish compensation headroom, learned
correction efficacy, feature-conditioned gain value, shuffle falsification, or
nominal harmlessness. It is a governance/contract blocker, not a negative
scientific result.

## What was completed

- A repaired DR0 bundle was independently audited as
  `PASS_DR0_AUTONOMOUS_ADVANCE`.
- A minimal isolated gyro/magnetometer compensation-conditioned right-error
  Split-KalmanNet pilot was implemented.
- The sealed implementation checkpoint digest is
  `68a08285878fa484291697c69e93ea855afe51521d90cc86061bc362b90a90a5`.
- Its recorded verification was 42 unit/red-path tests passing and a tiny smoke
  with 140 finite records: 7 variants (`C0`, `C1`, `N0`, `N1`, `N2`, `N3`,
  `N3S`) x 20 test trajectories, with 4 trajectories in each of R0-R4, one
  training seed (`31001`), whole-trajectory
  `attitude_geodesic_rmse_rad`, and 20 fixed-point-free N3S bridge records.
- Claude independently reproduced the 42-test baseline and tiny smoke
  bit-exactly, then rejected implementation advancement because mutation tests
  exposed 18 green-when-they-should-be-red invariants.
- The single implementation repair round was opened and stopped before edits
  when the R8 contract gap was confirmed.

The smoke is implementation evidence only. It carries
`performance_claim=false` and `covariance_claim_valid=false`; its numeric
values are not method-lock evidence.

## What remains incomplete

- The implementation counterproposal R1-R11 was not applied or resealed.
- CF-1/CF-2, train/runtime operator equivalence, reset binding, N3S checkpoint
  identity, exact gate red paths, and gate metric producers did not receive an
  independent implementation PASS.
- The full population (train 40/regime, validation 10/regime, test 30/regime,
  seeds 31001/31002/31003) was never generated or evaluated for method lock.
- No experiment result, paired bootstrap interval, or G0-G4 decision exists.

## Observability and claim limits

A single magnetic reference has instantaneous attitude sensitivity rank two;
rotation parallel to the body-frame magnetic direction is weak/unobservable.
No thresholded weak-axis population was lawfully frozen, which is the decisive
contract gap here. The pilot's neural gain factors are not physical covariance
matrices, so no NIS, NEES, calibrated-uncertainty, flight, closed-loop, energy,
or generality claim is supported.

## Minimum lawful next action

This run has no executable resume command because the repair limit is consumed.
A future, newly authorized study must first freeze an exact weak-axis and
observable-plane population membership formula and threshold in the canonical
math/gate contract, independently audit that contract before any test access,
and then begin a fresh implementation audit cycle. It must not reuse this run
to retroactively define or evaluate G0-G4.

Main Phase 2, SpikeRA-KalmanNet, frozen Phase 0-1 evidence, and
`.codex/config.toml` were not modified by this finalization.
