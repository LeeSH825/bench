# P1A-CP4 Integration Contract

Status: implemented and validated in Phase 1A CP4 Step 2.

## 1. Purpose and fixed identities

This contract integrates the already validated Gate A MEKF, Gate B typed event
datasets, Gate C metrics, and the D1 replay bridge into the existing suite
runner. It does not define a new research gate and does not authorize a neural
model.

The only integrated pair is:

```text
task_family = mekf_unit_st_v1
model_id    = mekf_event_replay_v1
```

A partial match fails before the legacy data or model lifecycle. The estimator
input is the typed `MEKFEventTable`; a dense `float32 y`, regular-grid proxy, or
zero-filled event sequence is not an admissible representation.

## 2. Dispatch and registry kind

`bench.tasks.bench_generated.prepare_mekf_unit_st_v1` is a separate typed
sidecar dispatcher. It selects exactly one of:

- `synthetic-unit-st-v1`
- `basilisk-unit-st-v1`

The producer is resolved from `task.typed_event_dataset.producer_id`. The suite
seed becomes the generator `master_seed`; a conflicting embedded seed is an
error. Dataset identity excludes model ID, model metadata, training metadata,
and estimator configuration.

`mekf_event_replay_v1` lives in the separate typed-event bridge registry. It is
not inserted into the legacy `Type[ModelAdapter]` registry because that API
requires dense sequence prediction. Existing legacy registry lookup and list
behavior remain unchanged.

## 3. Runner branch and lifecycle

The runner recognizes either member of the reserved pair and validates the
complete pair before the legacy `_load_split_npz`, `_SeqDataset`, adapter
training/evaluation, or `_predict_batches` lifecycle can run. The exact-pair
order is:

1. validate the explicit filter state, time, covariance, and process noise;
2. generate or strictly load the typed three-file sidecar;
3. select one whole-trajectory train/validation/test split;
4. call the frozen D1 bridge once per selected trajectory;
5. only after every replay completes, exact-join truth by trajectory ID and
   timestamp;
6. call only the frozen Gate C metric functions;
7. atomically publish runner artifacts and metrics.

Only the `untrained:frozen` plan is valid. Training and adaptation are disabled
and their update ledgers remain zero.

## 4. Typed cache contract

The cache namespace is versioned by `p1a-cp4-typed-event-cache-v1`, producer,
the complete generator configuration hash, and suite seed. A valid dataset
directory contains exactly:

```text
manifest.json
truth.npz
events.npz
```

Fresh generation writes to an unrecognized temporary directory, runs the
strict loader with `expected_generator_id`, validates the result, and only then
atomically publishes `dataset/`. A cache hit repeats the strict loader and
validates:

- schema, generator, convention, zero-latency, and same-time order IDs;
- complete resolved generator configuration and master seed;
- seed-policy version, deterministic trajectory IDs, all derived stream seeds,
  and whole-trajectory split membership;
- current runtime identity and current source/proof fingerprints;
- all five recorded semantic hashes.

A stale, incomplete, corrupted, wrong-producer, wrong-config, wrong-runtime, or
wrong-source cache fails loudly. It is neither overwritten nor silently
regenerated in place.

## 5. Initial state and process noise

Every replay uses caller-owned, truth-independent values from
`task.mekf_replay`:

- scalar-first active `q_NB`;
- gyroscope bias estimate in rad/s;
- positive six-element posterior covariance diagonal;
- nonnegative six-element continuous process-noise diagonal;
- initial time in seconds;
- whole-trajectory evaluation split and metric confidence level.

The runner records the resolved full `P` and `Q_c` matrices, units, time, and
convention. No initial state or process-noise term may be derived from truth.

## 6. Replay artifact contract

Each successful run publishes:

```text
artifacts/mekf_replay/manifest.json
artifacts/mekf_replay/trajectory_<trajectory_id>.npz
```

The NPZ is loaded with `allow_pickle=False` and contains only:

```text
event_index, event_order, timestamp_s, sensor_code,
q_hat_NB, b_hat_rad_s, P,
st_event_index, st_event_order, st_timestamp_s, st_residual, st_S
```

Star-tracker evidence is compact: it has one row per valid star-tracker update,
not one row per event. Truth, oracle values, labels, and future values are
forbidden in the trajectory NPZ.

The manifest is ASCII canonical JSON with sorted keys and compact separators.
It records the task/model pair, D1 adapter identity, all eight dataset identity
fields, producer and cache state, dataset and estimator configuration hashes,
whole split and trajectory filenames, event/update counters, metric contract,
and the explicit no-truth assertion.

Artifact publication uses a temporary sibling and atomic directory replacement.
A replay or serialization failure cannot leave a newly valid-looking partial
`mekf_replay` directory.

## 7. Truth join and metrics

Truth is inaccessible to the bridge and first appears after all estimation is
complete. The join requires both `trajectory_id` and the float64 timestamp to
match exactly. Interpolation, nearest-neighbor selection, tolerance-based
matching, and truth-derived initialization are forbidden.

The runner calls the frozen `bench.metrics.mekf` implementations for:

- scalar-first quaternion-sign-invariant attitude geodesic error;
- gyroscope-bias error and RMSE;
- star-tracker NIS from compact residual and `S` evidence;
- right-local six-dimensional NEES from matched posterior `P`;
- strict SPD diagnostics for `P` and `S`;
- chi-square consistency summaries.

Reported minimum fields are attitude RMSE in rad/deg, P95 and maximum;
per-axis/vector bias RMSE; NIS/NEES count, mean, normalized mean, and interval;
and `P`/`S` SPD counts, minimum eigenvalue, asymmetry, and Cholesky status.

## 8. Same-realization and legacy isolation

Changing model display metadata, training metadata, or other nonsemantic task
metadata does not change dataset configuration hash, semantic identity, cache
path, event stream, or replay arrays. Every estimator consumer of one dataset
uses the same strict sidecar identity.

The legacy dense cache, `ModelAdapter` registry, prediction artifact, metrics,
training, and visualization paths retain their prior behavior. This integration
does not modify `bench/tasks/data_format.py` or
`bench/tasks/generator/contract.py`.

## 9. Scope boundary

This contract completes the Phase 1A foundation only. Phase 1B classical MEKF
benchmark completion is the next authorized stage. Package C, neural models,
and visualization work are outside this execution and are not started here.
