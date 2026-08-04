# Phase 1A MEKF Adapter and Artifact Contract

- Gate: D1
- Adapter ID: `mekf-event-replay-bridge`
- Adapter version: `p1a-gate-d1-v1`
- Status: verified for unregistered direct use

## Purpose and input basis

The Gate D1 bridge exposes the already verified Gate B1 typed-event replay as a
truth-free estimator boundary. It does not own quaternion, propagation, update,
or event-ordering math. The only numerical estimator call is the frozen
`bench.tasks.generator.mekf_events.replay_trajectory` function, which in turn
uses the frozen Gate A core.

The estimator-facing call is:

```python
MEKFEventReplayBridge.replay_events(
    event_table,
    trajectory_id,
    initial_state,
    initial_time_s,
    Q_c,
    dataset_identity,
) -> MEKFReplayArtifact
```

`event_table` is a `MEKFEventTable`; `initial_state` is an explicitly supplied
`MEKFState`; `initial_time_s` is in seconds; and `Q_c` is the caller's float64
`[6,6]` continuous-time process-noise configuration. The bridge never derives
initial state or filter configuration from truth.

## Existing base API decision

`bench.models.base.ModelAdapter` and the present runner lifecycle are dense
sequence APIs. `ModelAdapter.predict(y_seq, ...)` accepts a regular sequence,
while `run_suite._load_split_npz` converts `x`, `y`, and row-shaped numeric
extras to float32 before `_SeqDataset` and batched `predict`. That path cannot
losslessly carry the disjoint float64/int64 typed payload tables, exact event
ordering, ragged per-trajectory replay, or compact star-tracker innovation
evidence.

Gate D1 therefore selects an explicit, registry-unregistered bridge rather than
a `ModelAdapter` subclass. It has no `setup`, `train`, `load`, `predict`,
`adapt`, or `save` lifecycle and declares `is_frozen=True` and
`supports_training=False`. This is the smallest implementation that satisfies
the typed-event contract without changing the frozen base API or runner.

## Truth-free boundary

No truth attitude, truth bias, truth angular rate, oracle covariance, event
label, future event, or metric result is accepted by `replay_events`. The event
table contains sensor packets only. `truth_hash` is carried solely as an opaque
dataset-provenance digest; it is not truth data and is never dereferenced by the
bridge.

Truth remains in `MEKFTruthTable` outside this module. Evaluation may join it
only after estimation, using exact `(trajectory_id, timestamp_s)` keys.

## Dataset identity

`DatasetIdentity` contains exactly:

| Field | Contract |
|---|---|
| `schema_version` | exact Gate B1 schema version |
| `generator_id` | strict versioned generator identity |
| `convention_id` | exact locked active B-to-N quaternion convention |
| `truth_hash` | lowercase SHA-256 hex |
| `sensor_payload_hash` | lowercase SHA-256 hex |
| `event_order_hash` | lowercase SHA-256 hex |
| `manifest_hash` | lowercase SHA-256 hex |
| `dataset_hash` | lowercase SHA-256 hex |

`DatasetIdentity.from_verified` copies values returned by a Gate B generator or
strict `load_event_dataset` call. It does not compute or rewrite any hash. If a
serialized manifest contains a `semantic_hashes` block, it must equal the
supplied verified hashes. A bridge may be constructed with
`expected_dataset_identity`; any non-exact identity then fails before replay.

The artifact repeats these fields in `ArtifactProvenance` and adds only
`adapter_id` and `adapter_version`. Adapter identity is outside the dataset hash
domain and cannot change sensor values or MEKF numerics.

## Artifact schema

`E` is the processed-event count and `S` is the count of valid star-tracker
updates. Every array is C-contiguous, defensively copied, and non-writeable.

| Field | dtype / shape | Unit and meaning |
|---|---|---|
| `trajectory_id` | Python `int` | whole-trajectory identity |
| `event_index` | `int64 [E]` | row in the input `MEKFEventTable` |
| `event_order` | `int64 [E]` | frozen within-trajectory order |
| `timestamp_s` | `float64 [E]` | measurement/filter time, seconds |
| `sensor_code` | `int16 [E]` | Gate B gyro or star-tracker code |
| `q_hat_NB` | `float64 [E,4]` | scalar-first active B-to-N posterior quaternion |
| `b_hat_rad_s` | `float64 [E,3]` | posterior body-frame gyro bias, rad/s |
| `P` | `float64 [E,6,6]` | posterior covariance of `[rad, rad/s]` right-local error |
| `st_event_index` | `int64 [S]` | input-table row for each valid ST update |
| `st_event_order` | `int64 [S]` | event order for each valid ST update |
| `st_timestamp_s` | `float64 [S]` | ST update time, seconds |
| `st_residual` | `float64 [S,3]` | right-local ST residual, rad |
| `st_S` | `float64 [S,3,3]` | ST innovation covariance, rad² |
| `final_state` | `MEKFState` | read-only final posterior `q/b/P` |
| `processed_event_count` | Python `int` | `E` |
| `gyro_event_count` | Python `int` | processed gyro count |
| `star_tracker_update_count` | Python `int` | `S` |
| `provenance` | `ArtifactProvenance` | dataset and adapter identity |

The event row is the posterior immediately after that event. At a shared
timestamp, the gyro row is the propagated posterior and the subsequent ST row
is the updated posterior. `final_state` must be bitwise equal to the last
`q_hat_NB`, `b_hat_rad_s`, and `P` row.

## Compact star-tracker evidence

The ST artifact contains one row only for an actual valid ST update. It has no
gyro placeholder rows, invalid-ST placeholder rows, sentinel values, or padded
records. `st_event_index`, `st_event_order`, and `st_timestamp_s` must exactly
select the ST rows from the processed event artifact. `st_residual` and `st_S`
are copied unchanged from `ReplayResult`.

## Replay dependency and integrity checks

The bridge calls `replay_trajectory` once per public call. It does not import or
call `propagate_state` or `star_tracker_update` and does not duplicate their
loop. Packaging maps the replay's unique `event_order` values back to input
table indices, checks timestamps and sensor codes exactly, and verifies that
the processed count equals gyro events plus valid ST events. P and S stacks
must remain exactly symmetric and Cholesky-SPD; no inverse, pseudo-inverse,
jitter, or tolerance repair is performed.

## Metric pairing boundary

The bridge does not import `bench.metrics.mekf`. After replay, evaluation may
join truth by exact trajectory/time and call the Gate C functions as follows:

- `q_hat_NB` plus joined `q_true_NB` for geodesic error;
- `b_hat_rad_s` plus joined true bias for bias error;
- `q_hat_NB`, `b_hat_rad_s`, `P`, and exactly paired truth for NEES;
- compact `st_residual` and `st_S` for NIS.

The posterior covariance and state estimate are paired on the same artifact
row. NIS uses only matching compact ST rows.

## Immutability

Artifact construction requires exact dtypes and ranks, makes independent
copies, sets every NumPy array non-writeable, and validates all counters and
index relationships. `final_state` is an immutable `MEKFState`, whose arrays
are independently owned and non-writeable. Mutating a source table, replay
result, caller state, or caller `Q_c` cannot mutate an existing artifact.

## Gate D2 minimum extension

Gate D2 must use the identifiers:

- model ID: `mekf_event_replay_v1`
- task-family ID: `mekf_unit_st_v1`

The required integration is append-only:

1. `bench/tasks/bench_generated.py`: add `mekf_unit_st_v1` dispatch that creates
   or locates the existing Gate B typed dataset sidecar. A cache hit must use
   strict `load_event_dataset` verification, preserve whole-trajectory split
   membership, and expose a sidecar reference without converting events to the
   legacy observation sequence.
2. `bench/models/registry.py`: add a separate typed-event bridge registry entry
   for `mekf_event_replay_v1`. Do not force the bridge into the
   `Type[ModelAdapter]` legacy registry contract.
3. `bench/runners/run_suite.py`: branch for the exact task-family/model pair
   after dataset path/cache resolution and before `_load_split_npz`,
   `_SeqDataset`, or `_predict_batches`. Strictly load the event sidecar,
   construct `DatasetIdentity` from the verified manifest/hashes, and call
   `replay_events` per whole test trajectory with task-configured initial state
   and `Q_c` that are not derived from truth.

Truth is joined only after all estimator artifacts return and immediately
before canonical metric evaluation. The exact join keys are
`(trajectory_id, timestamp_s)`; truth must never be put in bridge context.

Runner artifact storage must be:

```text
runs/.../artifacts/mekf_replay/manifest.json
runs/.../artifacts/mekf_replay/trajectory_<trajectory_id>.npz
```

Each NPZ uses `allow_pickle=False` compatible arrays with the exact `q/b/P/r/S`
schema above; `manifest.json` is canonical sorted compact JSON containing the
trajectory index, counters, dataset identity, and adapter identity. No generic
`preds_test.npz` coercion is permitted for this path.

On generation, load, and cache hit, Gate D2 must compare all of
`schema_version`, `generator_id`, `convention_id`, `truth_hash`,
`sensor_payload_hash`, `event_order_hash`, `manifest_hash`, and `dataset_hash`.
Existence-only cache acceptance is forbidden.

The direct/bridge/runner equivalence test must load one serialized event table,
use the same explicit initial state, initial time, and `Q_c`, then compare
`event_index`, order, timestamp, sensor code, q, b, P, ST residual, ST S, final
state, counters, and dataset identity with `np.array_equal`. It must cover a
fresh run and a verified cache hit for both synthetic and Basilisk producers.

The exact existing-file shortlist for Gate D2 is therefore:

```text
bench/tasks/bench_generated.py
bench/models/registry.py
bench/runners/run_suite.py
```

`bench/tasks/data_format.py` and `bench/tasks/generator/contract.py` remain out
of scope. If the sidecar cannot be delivered without changing either file,
Gate D2 must stop for an explicit schema-migration scope extension.

## Deferred scope

Registry/runner/task changes, cache integration, suite YAML, visualization,
nonzero latency/OOSM, additional sensors, Package C, neural models, training,
and hardware targets are not part of Gate D1.
