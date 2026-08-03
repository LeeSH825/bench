# Phase 1A Basilisk Frame Convention Proof

- Proof ID: `p1a-basilisk-frame-convention-proof-v1`
- Runtime: Python 3.10.13, NumPy 2.2.6, SciPy 1.15.3, Basilisk/bsk 2.10.2
- Interpreter: `/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python`
- Scope: spherical-inertia, zero-torque, attitude-only Basilisk truth

## Purpose and locked convention

Gate A defines scalar-first Hamilton `q_NB` as the active rotation from body
coordinates to navigation coordinates. Thus `R_NB = quat_to_dcm(q_NB)` maps a
body-frame coordinate column into the navigation frame. The Basilisk recorder
field name `sigma_BN` was not used as evidence by itself; the direction was
resolved with recorder-backed closed-form basis mappings.

The proven adapter is

```text
q_NB = normalize(Basilisk.RigidBodyKinematics.MRP2EP(sigma_BN))
R_NB = GateA.quat_to_dcm(q_NB)
C_BN = Basilisk.RigidBodyKinematics.MRP2C(sigma_BN)
R_NB = C_BN.T
```

`C_BN` maps navigation coordinates into body coordinates. Its transpose is the
locked active body-to-navigation rotation. No conjugation or inverse is applied
to the Euler parameter returned by `MRP2EP`.

## Candidate conventions

Two candidates were executed against independent closed-form Rodrigues basis
mappings:

1. `R_NB = quat_to_dcm(normalize(MRP2EP(sigma_BN)))`.
2. The inverse candidate `R_NB = MRP2C(sigma_BN)`.

Candidate 1 had maximum basis error `4.440892098500626e-16`. Candidate 2 had
maximum basis error `2.0` and is rejected. The selected candidate also satisfied
`R_NB = MRP2C(sigma_BN).T` with maximum element error
`4.440892098500626e-16`.

## Closed-form recorder proof

The test set contained identity; each axis at `+90 deg` and `-90 deg`; and ten
deterministic arbitrary axis-angle attitudes. The ±90-degree input MRPs were
constructed independently as `axis * tan(±pi/8)`. For each case a Basilisk
spacecraft with zero rate and torque recorded its initial `sigma_BN`; all three
body basis columns were then mapped by the selected `R_NB` and compared with a
closed-form Rodrigues rotation.

| Quantity | Result |
|---|---:|
| Cases | 17 |
| Arbitrary attitudes | 10 |
| Max recorder MRP error | `5.551115123125783e-17` |
| Max body-basis mapping error | `4.440892098500626e-16` |
| Max `R_NB - C_BN.T` element error | `4.440892098500626e-16` |

In particular, locked `+90 deg` about body `z` maps body `+x` to navigation
`+y`. The other two body basis vectors and all ±axis cases match their
closed-form mappings at float64 roundoff.

## MRP shadow-set proof

For each nonidentity proof attitude, the alternate representation

```text
sigma_shadow = -sigma / (sigma.T @ sigma)
```

was passed directly through the public adapter. Equality was assessed by DCM,
not raw quaternion components. Maximum physical DCM error was
`4.85722573273506e-16`, and the minimum `abs(q_primary dot q_shadow)` was `1.0`.
Basilisk may canonicalize a shadow MRP when it emits spacecraft state, but the
adapter itself accepts both finite representations. Nonfinite or malformed MRP
input fails loudly.

## Time-series representation

Every recorder sample is converted independently with the proven adapter. The
first sample receives a deterministic scalar/lexicographic representative; a
later sample is negated only if its dot product with the preceding represented
sample is negative. This adjacent-sign alignment changes no DCM and does not
impose a global `q[0] >= 0` rule on the trajectory.

## `omega_BN_B` frame, sign, and unit proof

The dynamic proof used a single rigid hub with representative mass `10 kg`,
spherical inertia `7 kg m^2`, body-origin center of mass, zero external torque,
and no orbit, gravity, or environment. It covered zero rate; each axis at
`+0.2 rad/s` and `-0.2 rad/s`; and ten deterministic arbitrary body rates with
norm at most `0.2 rad/s`. Duration was `1 s`.

The primary attitude error was

```text
delta_q = q_reference^-1 otimes q_basilisk
error_rad = norm(Log_q(delta_q))
q_reference(t) = q_initial otimes Exp_q(omega_BN_B * t)
```

| Grid | Step | Max attitude log error | Max local rate-increment error | Max recorded rate error |
|---|---:|---:|---:|---:|
| Coarse | `0.01 s` | `4.998209174226825e-16 rad` | `1.5855372570428017e-14 rad/s` | `0 rad/s` |
| Fine | `0.005 s` | `4.872566201647101e-16 rad` | `3.219646771412954e-14 rad/s` | `0 rad/s` |

The fine-grid attitude error did not increase and is far below the predeclared
`1e-8 rad` target. Rate-norm drift was zero on both grids. Therefore recorder
`omega_BN_B` is the body-frame angular rate in radians per second, with the same
sign and right-local propagation meaning used by Gate A:

```text
q_NB(t + dt) = q_NB(t) otimes Exp_q(omega_BN_B * dt)
```

## Decision

The executable proof confirms active body-to-navigation `q_NB`. The passive
wording previously attached to two Gate B1 field descriptions is a
documentation defect, not a different array convention. The adapter formula
above is locked for Gate B2.
