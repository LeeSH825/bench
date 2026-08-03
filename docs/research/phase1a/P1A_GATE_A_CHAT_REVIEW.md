# Phase 1A Gate A — Chat Independent Review

- Review date: 2026-07-31
- Reviewed artifact: `phase1a.zip` agent-only patch, Gate A tests, validation report, implementation contract, test matrix
- Review decision: **CONDITIONAL GO — Gate A Amendment A1 required before Gate B**

## 1. What was independently verified

The agent-only patch was applied to an isolated temporary repository and the submitted Gate A test command was rerun.

```text
42 passed in 1.41s
```

Additional independent property checks were executed against the submitted `bench/estimators/mekf.py` implementation:

| Check | Samples | Result |
|---|---:|---:|
| Quaternion → DCM → quaternion round trip, including near-pi cases | 10,000 | max DCM Frobenius error `9.7422e-16` |
| Exact reset Jacobian `J_r` vs central finite difference, corrections up to 0.5 rad | 1,000 | max relative error `6.2195e-10` |
| Van Loan `Q_d` vs independent 64-point Gauss–Legendre covariance integral | 50 | max relative error `7.3013e-16` |

These checks support the submitted propagation, Van Loan extraction, quaternion/DCM algebra, and reset Jacobian implementation.

## 2. Accepted Gate A results

- Scalar-first Hamilton, active B-to-N, right-error convention is implemented consistently.
- `F`, `G`, `Phi`, and `Q_d` signs and dimensions are consistent with the locked contract.
- Body-vector and sun tangent Jacobians agree with finite differences.
- Star-tracker update uses a three-dimensional local residual, Joseph covariance update, right injection, and exact `J_r` reset.
- Cholesky-based SPD solves are used without pseudo-inverse, silent jitter, or eigenvalue clipping.
- The core is independent of Basilisk, runner, model, task, metric, torch, and visualization imports.
- Existing selected legacy tests remained unchanged before and after the implementation.

## 3. Required Amendment A1 — exact-pi antipodal tie-break

The current implementation is antipodal-invariant for the tested ordinary and near-pi cases, but not for an exactly represented 180-degree relative quaternion.

Counterexample:

```text
q_hat = [1, 0, 0, 0]
q_z   = [0, 1, 0, 0]
-q_z  = [0,-1, 0, 0]

star_tracker_residual(q_hat, q_z)  = [+pi, 0, 0]
star_tracker_residual(q_hat,-q_z)  = [-pi, 0, 0]
```

Cause:

- `align_quaternion` flips only when the dot product is strictly negative.
- At an exact zero dot product, `q` and `-q` remain different representatives.
- The SO(3) logarithm at exactly pi is mathematically non-unique, so the implementation needs a deterministic software tie-break.

Required policy:

- Preserve ordinary shortest-arc behavior away from pi.
- Only inside a machine-roundoff-scale exact-pi tie region, select a deterministic axis sign, for example the first significant vector component is positive.
- Document that this removes representation dependence; it does not prove MEKF convergence from an exact 180-degree initial error.
- Add exact-pi tests for x/y/z and an arbitrary axis, including full update equivalence for `q_z` and `-q_z`.

## 4. Required Amendment A1 — state immutability hardening

`MEKFState` is a frozen dataclass and defensively copies inputs, but its NumPy arrays are currently writable. Direct mutation such as `state.P[0,0] = ...` succeeds.

Before event replay, the state arrays should be made non-writeable after copying, or the documentation must stop calling the state immutable. The recommended action is to enforce read-only `q_NB`, `b_g`, and `P` arrays and add tests for defensive copying and direct mutation failure.

## 5. Gate decision

| Item | Decision |
|---|---|
| B1 propagation/discretization | PASS |
| B3 body-vector Jacobian | PASS |
| B4 sun tangent Jacobian | PASS |
| B5 injection/reset | PASS |
| B6 ordinary/near-pi sign invariance | PASS |
| B6 exact-pi representation invariance | **AMENDMENT REQUIRED** |
| Numerical safety | PASS |
| Import/information boundary | PASS |
| Gate A | **CONDITIONAL GO** |

Gate B must not start until Amendment A1 passes. After A1, the next scope is Gate B1 only: typed zero-latency event schema, deterministic serialization/hash, trajectory-level split, synthetic UNIT-ST generator, and direct-core replay. Basilisk frame adaptation, canonical metrics, runner integration, and delayed/OOSM handling remain later gates.
