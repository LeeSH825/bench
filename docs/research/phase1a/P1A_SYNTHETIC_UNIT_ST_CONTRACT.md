# Phase 1A synthetic UNIT-ST contract

## Locked analytic model

Each whole trajectory samples one initial attitude `q_NB(0)`, constant
body-frame angular rate `ω`, and constant gyro bias `b`. On the exact float64
gyro grid `t_k = k / f_g`, including truth at `k=0`, the passive attitude is

```text
q_NB(t_k) = normalize(q_NB(0) ⊗ Exp(ω t_k)).
```

The first sensor event is gyro at `k=1`. Gyro measurements use the locked sign
and unit equation

```text
ω_m,k = ω + b + n_g,k                  [rad/s].
```

Star-tracker times are the gyro-grid subset defined by the exact integer ratio
`f_g / f_ST`. The right-local measurement is

```text
q_ST,k = q_NB(t_k) ⊗ Exp(n_ST,k),
n_ST,k ~ N(0, σ_ST² I),                [rad].
```

Its payload covariance is the configured positive diagonal `R_ST` in rad².
An independent sign stream may multiply any raw ST quaternion by `-1`; this is
representation only, leaves the physical rotation unchanged, and must leave
Gate A replay residual/correction/posterior evidence unchanged.

## Representative configuration and status

The immutable default representative configuration uses 8 trajectories, 2 s,
20 Hz gyro, 5 Hz ST, master seed `20260731`, maximum initial rotation 0.8 rad,
maximum angular rate 0.25 rad/s, maximum gyro bias 0.015 rad/s, gyro noise
standard deviation `8e-4` rad/s, ST noise standard deviation `1.5e-3` rad, and
`diag(R_ST)=(2.25e-6,2.25e-6,2.25e-6)` rad². Split fractions are 0.6/0.2/0.2.

These values are a representative deterministic validation configuration, not
a mission-fidelity, calibrated sensor, or performance claim. The locked B1
assumptions are constant rate, constant bias, analytic truth, white independent
measurement noise, integer cadence ratio, zero latency, and all-valid nominal
events.

## Seed isolation

The versioned policy is `p1a-separated-streams-v1`. SHA-derived NumPy generator
seeds use distinct named namespaces for:

```text
truth
gyro-noise
star-tracker-noise
star-tracker-sign
trajectory-split
```

Truth, gyro noise, ST noise, and raw sign have per-trajectory derived seeds;
split has a dataset-level derived seed. All derived values are recorded in the
manifest. Changing a sensor namespace preserves `truth_hash`; changing the
truth namespace changes it. A sign-only change may change raw sensor hash but
the quaternion dot-product magnitude stays one and replay is physically
identical. A split-only change preserves all data hashes and changes only split
metadata/manifest identity.

## Trajectory identity and split

Trajectory IDs are stable unique signed `int64` values derived from generator
ID, master seed, and trajectory index; sensor and split seeds cannot relabel
them. Train/validation/test assignment hashes `(split_seed, trajectory_id)`, so
it is deterministic and independent of input ordering. Fractions must be finite,
strictly positive, sum to one, and produce three nonempty groups. Duplicates and
too few IDs fail loudly. Selection copies complete truth spans and all associated
events, while compacting typed payload indices; an individual trajectory can
never cross split boundaries.

## Deferred work

Gate B1 intentionally does not implement Basilisk generation/adaptation,
nonzero arrival latency, event outages/dropouts, invalid-sensor campaign policy,
UNIT-ST performance metrics, runner or registry integration, cache migration,
visualization, or neural models. Those remain Gate B2/C/D work and require a
separate execution contract.
