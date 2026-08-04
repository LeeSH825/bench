# Phase 1A Gate B1 typed MEKF event contract

## Decision

This document locks the Gate B1 in-memory and on-disk boundary for analytic
UNIT-ST data. Its inputs are the Phase 0A convention/math decisions, the
approved Gate A MEKF API, and the Gate B1 execution contract. The decision is
implemented for the synthetic zero-latency path. Latency, invalid/outage policy
beyond fail-loud nominal handling, and a Basilisk adapter remain TBD for a later
gate. The next eligible step is Gate B2 only after separate approval.

Frozen schema and convention constants are:

- schema identity (`schema_version`): `p1a-mekf-events-v1`
- seed policy: `p1a-separated-streams-v1`
- convention: `qNB-scalar-first-hamilton-right-v1`
- sensor codes: gyro `1`, star tracker `2`, stored as `int16`

Dataset-generator identity is a separate manifest field. The analytic generator
records `synthetic-unit-st-v1`; a Gate B2 dataset will record
`basilisk-unit-st-v1`. Supporting that identity in the common serializer does
not implement or approve the Basilisk generator.

## Event and payload fields

`E`, `G`, and `S` denote event, gyro-payload, and star-tracker-payload counts.
All numeric arrays are C-contiguous defensive copies and are read-only after
construction.

| Field | dtype | Shape | Unit | Frame / meaning |
|---|---:|---:|---|---|
| `trajectory_id` | `int64` | `[E]` | none | Stable whole-trajectory identity |
| `sensor_code` | `int16` | `[E]` | none | `1=GYRO`, `2=STAR_TRACKER` |
| `measurement_time_s` | `float64` | `[E]` | s | Physical measurement time |
| `arrival_time_s` | `float64` | `[E]` | s | Filter arrival time; identical to measurement time in B1 |
| `event_order` | `int64` | `[E]` | none | Unique deterministic tie-break within a trajectory |
| `valid` | `bool` | `[E]` | none | Payload-valid flag; nominal B1 data are all true |
| `payload_index` | `int64` | `[E]` | none | Index into the payload table selected by `sensor_code` |
| `gyro_omega_rad_s` | `float64` | `[G,3]` | rad/s | Body-frame measured angular rate |
| `star_tracker_q_NB` | `float64` | `[S,4]` | none | Scalar-first Hamilton active body-to-navigation `q_NB` |
| `star_tracker_R_rad2` | `float64` | `[S,3,3]` | rad² | Right-local rotation-vector covariance |

Payload ownership is one-to-one: every gyro or ST payload index appears exactly
once in an event of its sensor type, with no cross-type reinterpretation.

## Separate truth fields

`N` is the number of trajectories and `T_total` is the flattened truth sample
count. Truth is not an event payload and is not accepted by replay.

| Field | dtype | Shape | Unit | Frame / meaning |
|---|---:|---:|---|---|
| `trajectory_id` | `int64` | `[N]` | none | Unique truth trajectory identities |
| `truth_offsets` | `int64` | `[N+1]` | samples | Ragged offsets, starting at zero |
| `truth_time_s` | `float64` | `[T_total]` | s | Strictly increasing within a trajectory |
| `q_true_NB` | `float64` | `[T_total,4]` | none | Analytic scalar-first Hamilton active body-to-navigation `q_NB` |
| `gyro_bias_rad_s` | `float64` | `[T_total,3]` | rad/s | Constant body-frame gyro bias |
| `omega_true_rad_s` | `float64` | `[T_total,3]` | rad/s | Constant true body angular rate |

## Ordering and replay semantics

The canonical per-trajectory key is `(arrival_time_s, event_order)`. Tables must
already be in that order; replay never silently sorts malformed input. At a
shared timestamp, the gyro row has the lower `event_order` and is processed
first. It advances the current filter time using `propagate_state`; the ST row
then calls `star_tracker_update` at that exact current time and reads `R_ST`
from its payload. The first gyro is at `t1 > t0`, never at the initial time.

Gate B1 is strictly zero latency: `arrival_time_s == measurement_time_s` by
exact float64 equality for every accepted event. Nonzero latency is rejected.
Invalid gyro input is rejected. Invalid ST input may be skipped, but the nominal
generator emits valid events only.

Replay accepts only an event table, trajectory ID, initial `MEKFState`, initial
time, and fixed nominal `Q_c`. It returns read-only time, sensor, posterior
state/covariance, residual, innovation-covariance evidence, and a final Gate A
state. No truth, oracle, label, future sample, model ID, metric, or runner state
crosses this API.

## Validation policy

Construction fails loudly on a wrong dtype, rank, shape, non-finite value,
negative time/index, unknown sensor code, non-normalized ST quaternion,
nonsymmetric/non-SPD ST covariance, duplicate or out-of-order event tie-break,
nonzero latency, payload ownership error, duplicate truth ID, malformed ragged
offset, or non-increasing truth time. No coercion from the legacy float32 format,
pseudo-inverse, jitter, eigenvalue clipping, skip, or xfail is used.

## Serialization and semantic identity

A complete artifact is exactly one directory containing:

```text
manifest.json
truth.npz
events.npz
```

NPZ loading always uses `allow_pickle=False`; object arrays, missing/extra files,
missing/extra fields, partial artifacts, noncanonical manifests, and hash
mismatches are rejected. `manifest.json` is ASCII JSON with sorted keys, compact
separators, and no NaN/Infinity.

Arrays are hashed as canonical little-endian, C-order bytes. Each digest domain
includes, in order, the field name, canonical dtype string, exact shape, payload
length, and bytes. SHA-256 identities are:

- `truth_hash`: all truth fields;
- `sensor_payload_hash`: gyro and ST payload fields;
- `event_order_hash`: all event routing/order/validity fields;
- `manifest_hash`: canonical manifest without its `semantic_hashes` block;
- `dataset_hash`: a domain-separated composition of truth, sensor-payload, and
  event-order hashes. Split/config metadata therefore changes `manifest_hash`,
  while a split-seed-only change does not relabel the physical dataset hash.

The manifest identity includes every frozen version ID, complete generator
configuration, master and derived named seeds, trajectory IDs and split,
Python/NumPy/SciPy versions, zero-latency/order declarations, and SHA-256 source
fingerprints for Gate A core, event schema, and generator. Source fingerprints
are manifest data; the `semantic_hashes` block itself is excluded to avoid a
circular hash.

### Schema identity and dataset-generator identity

The existing `schema_version` field is the schema identity. It identifies the
event/truth fields, dtype and rank rules, timing/order semantics, serialization,
and replay boundary. No redundant `schema_id` field is added. A strict loader
accepts only the supported `p1a-mekf-events-v1` schema.

The independent `generator_id` field identifies which deterministic generator
family and version produced a dataset. It must be a string with no surrounding
whitespace and must match:

```text
<lowercase-family>-v<positive-integer>
```

The family is one or more lowercase alphanumeric hyphen-separated tokens; the
version has no sign, zero value, or leading zero. Consequently both
`synthetic-unit-st-v1` and `basilisk-unit-st-v1` are valid and distinct, while
empty, whitespace-only, unversioned, uppercase, underscore, `v0`, and `v01`
identities are rejected.

`load_event_dataset(path, expected_generator_id=...)` optionally locks a caller
to one generator. An exact match loads; a mismatch raises `ValueError`. Omitting
the argument preserves existing callers but still validates recorded schema and
generator identities, canonical JSON, and all semantic hashes. Both identity
fields are part of `manifest_hash`; `generator_id` is intentionally not part of
the physical `truth_hash`, `sensor_payload_hash`, `event_order_hash`, or their
composed `dataset_hash`.

Synthetic and future Basilisk UNIT-ST producers must call this same serializer
and loader in `bench/tasks/generator/mekf_events.py`. A producer-specific copy
of the serializer is forbidden.

## Boundary status

Truth exists for generation validation and test-only analytic comparisons. The
estimator-facing replay path cannot receive it. Legacy sequence adapters,
Basilisk truth/sensor adapters, delayed arrival, outages, metrics, runners,
registries, visualization, and neural-model inputs are outside Gate B1 and have
not been implemented here. Gate B1 Amendment A1 only made the shared manifest
identity contract compatible with a distinct future Basilisk generator ID.

## Gate B2 convention erratum

The former `passive q_NB` wording in the `star_tracker_q_NB` and `q_true_NB`
field descriptions was a documentation defect. Gate A and the Gate B1
generator/replay already use scalar-first Hamilton active body-to-navigation
`q_NB`; no code behavior or physical dataset meaning changed.

Gate B2 recorder-backed identity, each-axis ±90-degree, ten-arbitrary-attitude,
and MRP shadow-set basis proofs confirmed that `R_NB = quat_to_dcm(q_NB)` maps
body coordinates into navigation coordinates. For Basilisk `sigma_BN`, the
verified relation is `MRP2C(sigma_BN) = C_BN = R_NB.T`, with
`q_NB = normalize(MRP2EP(sigma_BN))`.

This correction is not a schema migration. Array keys, dtype, rank, byte
representation, serializer, schema/generator versions, event ordering, replay
API, and every semantic-hash domain remain unchanged.
