# Phase 1A Basilisk UNIT-ST Contract

## Scope

This contract defines the first Basilisk-backed `UNIT-ST` dataset for Phase 1A
Gate B2. It adds a truth adapter and project-owned sensor-output layer while
leaving the validated Gate A MEKF and Gate B1 typed event, serialization, hash,
split, and replay contracts unchanged.

Locked identities are:

```text
generator_id              = basilisk-unit-st-v1
simulator_adapter_version = basilisk-sigmaBN-to-qNB-v1
sensor_model_version      = parameterized-gyro-st-v1
schema_version            = p1a-mekf-events-v1
convention_id             = qNB-scalar-first-hamilton-right-v1
seed_policy_version       = p1a-separated-streams-v1
```

## Tier-0 truth spacecraft

Each trajectory is an independent minimal Basilisk simulation:

- one rigid `spacecraft.Spacecraft` hub;
- representative normalized mass and spherical inertia;
- center of mass at the body origin;
- zero external torque;
- no orbit, gravity, environment, actuator, or other dynamics;
- one process, one fixed-rate task, and one `scStateOutMsg` recorder.

Default mass `10 kg` and inertia `7 kg m^2` are representative test parameters,
not claims about a flight vehicle. Spherical inertia makes zero-torque body rate
constant and permits a direct Gate A propagation proof.

The task step is exactly `1 / gyro_rate_hz`. Duration times gyro rate must be an
integer. The recorder grid therefore equals the truth and gyro grid; star
tracker rate must be an integer divisor of gyro rate.

## Frame and truth arrays

Gate A's active scalar-first Hamilton convention is authoritative. The
recorder-backed proof locks:

```text
q_NB_true = normalize(MRP2EP(sigma_BN))
R_NB      = quat_to_dcm(q_NB_true)
C_BN      = MRP2C(sigma_BN)
R_NB      = C_BN.T
```

`R_NB` maps body coordinates into navigation coordinates. Recorder
`omega_BN_B` is the angular rate of B relative to N, expressed in B, in rad/s,
with the same sign as Gate A right propagation:

```text
q_NB(t+dt) = q_NB(t) otimes Exp_q(omega_BN_B * dt)
```

Per trajectory the typed truth table stores float64, finite arrays:

| Array | Meaning | Unit |
|---|---|---|
| `truth_time_s` | Basilisk recorder time | s |
| `q_true_NB` | active body-to-navigation attitude | unit quaternion |
| `omega_true_rad_s` | recorder `omega_BN_B` | rad/s |
| `gyro_bias_rad_s` | project sensor-layer constant bias truth | rad/s |

Quaternion representations have unit norm. The first sign is deterministic;
later signs are aligned only to the previous represented sample. This affects
representation continuity, not physical DCM.

## Project-owned sensor-output layer

No Basilisk built-in star tracker is used or claimed. Both outputs below are a
project-owned parameterized wrapper applied to Basilisk truth.

Gyro events use body-frame radians per second:

```text
omega_m = omega_true_B + b_g_true + n_g
```

`b_g_true` is constant per trajectory. `n_g` is independent white Gaussian
noise with configured standard deviation. Gate B2 has no random walk, missing
sample, saturation, or scale/misalignment term.

Quaternion star-tracker events use right-local tangent noise:

```text
n_ST ~ N(0, R_ST)
q_ST = q_NB_true otimes Exp_q(star_tracker_noise_scale * n_ST)
```

`R_ST` is an exactly symmetric, strictly SPD float64 `3x3` covariance. A zero
noise exactness run sets `star_tracker_noise_scale=0` while preserving the
strictly SPD nominal `R_ST` passed to Gate A. An independent optional sign
stream may emit either `q_ST` or `-q_ST`; physical measurement and replay are
identical.

## Immutable resolved configuration

`BasiliskUnitSTConfig` freezes trajectory count, duration, gyro/ST rates,
master seed, initial attitude/rate bounds, bias bound, gyro noise, ST covariance
and sampling scale, raw sign enable, representative mass/inertia, split
fractions, and all seed namespaces. Invalid time/rate alignment, negative
magnitude, rate above the Gate B2 bound, nonpositive mass/inertia, non-SPD ST
covariance, invalid fractions, or colliding/empty namespaces fails loudly.

## Seed isolation

The master seed and frozen seed-policy identity derive disjoint namespaces for:

```text
Basilisk truth initial attitude
Basilisk truth initial body rate
gyro bias truth
gyro white noise
star-tracker tangent noise
star-tracker representation sign
whole-trajectory split
```

Each physical stream is further separated by trajectory ID. Estimator or model
identity is never part of seed derivation. Changing gyro/ST noise preserves
Basilisk attitude/rate truth; changing bias preserves attitude/rate truth while
changing bias truth and gyro payload; changing initial-condition streams changes
truth; changing only split leaves physical truth/sensor/order hashes unchanged.

## Event schedule and Gate B1 reuse

Every gyro epoch after `t=0` emits one valid gyro event. Star-tracker epochs are
a subset of gyro epochs. All valid events have exact
`arrival_time_s == measurement_time_s`. At a shared timestamp the gyro event has
the lower `event_order`, so replay propagates before applying the star-tracker
update.

The generator constructs the existing `MEKFEventTable`, `MEKFTruthTable`, and
`MEKFDataset`. It calls Gate B1 `compute_semantic_hashes` and
`split_trajectory_ids`; persistence and verification call the unchanged
`save_event_dataset` and strict
`load_event_dataset(..., expected_generator_id="basilisk-unit-st-v1")`.
No duplicate schema, serializer, semantic hash, splitter, or replay engine is
implemented. Artifacts remain exactly:

```text
manifest.json
truth.npz
events.npz
```

NPZ keys, dtypes/ranks, byte representation, hash domains, event order, and
replay API are unchanged from Gate B1.

## Manifest and hash identity

The manifest records all locked IDs, complete resolved config, zero-latency and
order declarations, master/derived seeds, trajectory IDs and memberships,
runtime versions and Basilisk path, source fingerprints for Gate A, Gate B1,
this generator, and the frame-proof document. Generated artifacts and the
validation report are excluded from source fingerprints, avoiding a cycle.

The strict loader validates the recorded generator ID and all manifest and
array semantic hashes. Identity tampering, wrong expected ID, corrupt NPZ,
noncanonical JSON, or extra/missing files fails loudly.

## Truth/sensor/estimator boundary

Basilisk state creates truth. The sensor wrapper reads truth to create payloads.
The replay public API receives only typed events, trajectory ID, initial filter
state/time, and nominal process noise. It cannot receive `q_true_NB`, true bias,
true rate, future values, oracle, label, model ID, metric, runner, or
visualization state. Truth arrays and Gate A state arrays remain read-only.

## Deferred features

Nonzero latency, outages, false solutions, magnetometer, sun sensor,
orbit/environment truth, canonical metrics, runner/registry integration,
visualization, and neural models are outside Gate B2. No Gate C implementation
is authorized by this contract.

