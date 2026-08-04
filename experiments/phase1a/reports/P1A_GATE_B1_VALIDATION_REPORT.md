# Phase 1A Gate B1 validation report

## Decision

```text
Status: PASS_GATE_B1

Schema: PASS
Zero latency/order: PASS
Serialization/hash: PASS
Seed isolation: PASS
Trajectory split: PASS
Synthetic UNIT-ST: PASS
Direct replay: PASS
Truth boundary: PASS
Numerical/replay safety: PASS
Gate A regression: PASS
Legacy regression: PASS
Dirty-tree integrity: PASS
Gate B1: GO
```

No blocking issue was found. This decision authorizes no automatic Gate B2
work; execution stopped at Gate B1.

## Generated paths

Only the nine exact allowlist targets were created:

```text
bench/tasks/generator/mekf_events.py
bench/tasks/generator/unit_st_synthetic.py
tests/test_mekf_events.py
tests/test_unit_st_synthetic.py
tests/test_mekf_replay.py
docs/research/phase1a/P1A_EVENT_SCHEMA_CONTRACT.md
docs/research/phase1a/P1A_SYNTHETIC_UNIT_ST_CONTRACT.md
docs/research/phase1a/P1A_GATE_B1_TEST_MATRIX.md
experiments/phase1a/reports/P1A_GATE_B1_VALIDATION_REPORT.md
```

The required `03A_*` execution evidence and
`preflight_snapshots/03A_20260731T093301Z/**` were also created under the
contract-authorized provenance prefixes. No pre-existing file was modified.

## Commands and results

All Python commands used the explicit interpreter
`/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python`. Pytest commands set
`PYTHONDONTWRITEBYTECODE=1` and disabled its cache plugin.

Before implementation, Gate A was reconfirmed with:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_mekf_conventions.py tests/test_mekf_core.py
```

Result: `55 passed in 0.83s`.

Before implementation, the designated legacy subset was run with:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_basilisk_imu_generator.py tests/test_basilisk_mrp_ekf.py \
  bench/tests/test_generator_contract_tg0.py bench/tests/test_adcs_event_metrics.py
```

Result: `18 passed, 5 subtests passed in 4.40s`.

The complete new Gate B1 suite was run with:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_mekf_events.py tests/test_unit_st_synthetic.py tests/test_mekf_replay.py
```

Final result: `39 passed in 0.78s`. An initial diagnostic run found two test
setup cases that swapped event order across different timestamps and therefore
did not violate the sort key. The cases were corrected to reverse the same-time
gyro/ST tie; no production behavior, tolerance, expected numerical value, skip,
or xfail was changed.

After implementation, Gate A produced `55 passed in 0.58s`; the legacy subset
produced `18 passed, 5 subtests passed in 4.43s`. Both exit codes were zero.

The standalone property command generated seeds 4100 through 4109, four whole
trajectories per seed, regenerated each dataset, serialized and strictly loaded
it, replayed nominal and all-ST-sign-negated streams, checked split disjointness,
and applied finite/unit-quaternion/Cholesky checks. Result:

```text
PROPERTY_SWEEP datasets=10 trajectories_per_dataset=4
regeneration=PASS roundtrip=PASS sign_pair=PASS split=PASS replay_safety=PASS
```

Complete command output is in the required `03A_*` evidence logs.

## Schema, ordering, and serialization evidence

Event metadata use exact `int64/int16/float64/float64/int64/bool/int64`
dtypes for trajectory, sensor, measurement time, arrival time, event order,
valid flag, and payload index. Payloads are float64 `[G,3]`, `[S,4]`, and
`[S,3,3]`. Truth is separate float64/int64 ragged storage. Wrong dtype/rank,
payload ownership, quaternion norm, covariance SPD, latency, and ordering
counterexamples all rejected during the 39-test suite.

Every generated event satisfied exact
`arrival_time_s == measurement_time_s`. At shared timestamps, the gyro had the
lower event order and replay propagated it before applying the ST update.

The artifact format was exactly `manifest.json`, `truth.npz`, and `events.npz`.
Canonical JSON and strict `allow_pickle=False` NPZ loading round-tripped every
array and all five hashes exactly. Object arrays, corrupt JSON, and a forged
recorded hash were rejected.

A representative seed-731 semantic identity was:

```text
truth_hash          9b9545b069cdf3c0feb5e636e45213a1a17bf49dd18cfb1d7ef0c53a8152a71d
sensor_payload_hash 2fe16d091f43d3c0c24cde6044ecbba043ff7f1f8b8bf20ffb1edbe19e6da38a
event_order_hash    02bdea51896c359f66dd489f363aecd5d779cd0b64fa29c612d30f62f65ef125
manifest_hash       cfa52f175fa349745496a24d541fd2ffc4ee7559c6c756d867a1827df6822696
dataset_hash        60607c5f078fd170392ec58846b44e8c3e43157509e1a7f74628d1ba9fa798e7
```

## Determinism, seeds, and split evidence

Repeated identical configuration produced identical semantic hashes and stable
unique int64 trajectory IDs. With only the gyro-noise namespace changed, the
truth hash stayed equal and sensor hash changed. With only the truth namespace
changed, truth hash changed. With only the split namespace changed, the split
changed while truth, sensor, event-order, and dataset hashes stayed equal. With
only the ST sign namespace changed, raw ST representation changed while each
physical quaternion retained `abs(q·q′)=1`.

For the representative 8-trajectory data, train/validation/test counts were
`5/2/1`; all three pairwise intersection counts were zero and the union count
was 8. Reversing input ID order produced identical memberships. Selecting the
validation split retained only its complete truth spans and event/payload rows.

## Direct replay and truth boundary

Replay directly composed the frozen Gate A `propagate_state`,
`star_tracker_update`, and `quat_geodesic_angle` APIs. The inspected public
signature was exactly:

```text
(event_table, trajectory_id, initial_state, initial_time_s, Q_c)
```

There is no truth, oracle, label, future sample, model, metric, runner, or
registry input. Truth arrays remained unchanged and were not passed to replay.

For a representative trajectory, replay processed 20 events including 4 ST
updates; maximum quaternion-norm deviation was `0.0`, minimum posterior
covariance eigenvalue was `4.7605859379652635e-07`, and all covariance entries
were finite. Same-stream and serialization-round-trip replay evidence was
array-identical. Negating every raw ST quaternion produced exact-equal posterior
q/b/P, residuals, and innovation covariances. A 10-second sequence maintained
unit quaternions and Cholesky-positive posterior covariance throughout. Direct
mutation of frozen Gate A q/b/P arrays continued to raise `ValueError`.

## Dirty-tree integrity

The current working tree at `2026-07-31T09:33:01Z` was captured before any target
creation. Branch and HEAD were recorded only as provenance. The final comparison
covered 1,260 pre-existing dirty paths:

```text
pre-existing status/content mismatches: 0
unexpected new paths outside authorized prefixes: 0
staged changes: 0
concurrent external artifact additions: 0
frozen Gate A fingerprint changes: 0
```

No reset, restore, clean, stash, stage, commit, push, merge, or rebase operation
was performed. The external artifact ledger was path/status-only and empty; no
new `artifacts/benchmark_write_control/**` path was read or modified.

## Deferred scope

Still deferred are Gate B2 Basilisk event generation/adaptation and nonzero
latency/outage work, Gate C metric/runner/registry integration, and Gate D
visualization or neural-model work. None of those features was implemented or
integrated. The contract-mandated read-only reference inspection and legacy
regression exercised existing Basilisk-related code only; they did not begin
Gate B2 development.

## Amendment A1 — generic manifest generator identity (2026-08-01)

### Decision

Gate B1 Amendment A1 is approved. The common serializer now distinguishes the
fixed `schema_version=p1a-mekf-events-v1` from a strict versioned
`generator_id`. Both `synthetic-unit-st-v1` and `basilisk-unit-st-v1` serialize
and strictly load without changing the event/truth NPZ contract. This amendment
did not create, import, or run a Basilisk Gate B2 implementation.

### Implementation and API

`bench/tasks/generator/mekf_events.py` now validates generator IDs against
`<lowercase-family>-v<positive-integer>` rather than comparing every producer
to the synthetic constant. The loader API is backward compatible and adds the
keyword-only lock:

```python
load_event_dataset(path, expected_generator_id="basilisk-unit-st-v1")
```

A match succeeds; a mismatch raises `ValueError`. Omitting the argument still
validates the supported schema, recorded generator-ID syntax, canonical
manifest, and semantic hashes.

### Regression and extension evidence

The pre-implementation results were `55 passed` for Gate A, `39 passed` for the
original Gate B1 suite, and `18 passed, 5 subtests passed` for the designated
legacy subset. After A1, Gate A remained `55 passed`, Gate B1 increased to
`55 passed` without deleting or weakening its original 39 cases, and legacy
remained `18 passed, 5 subtests passed`.

New tests cover both required generator identities, strict expected-ID
match/mismatch, empty/whitespace/malformed/unversioned rejection, generator-ID
tamper detection, unsupported schema rejection, exact three-file and NPZ
key/dtype/rank invariance, and frozen representative synthetic data hashes. A
separate two-ID by five-seed sweep passed all 10 cases, including exact array
and direct-replay equality and object-array rejection.

The previous Gate B2 compatibility probe failed because the serializer required
`synthetic-unit-st-v1`. The A1 probe successfully saved and strictly loaded the
same deterministic fixture with recorded ID `basilisk-unit-st-v1`. Expected-ID
mismatch, whitespace ID, unsupported schema, and hash-inconsistent valid-ID
tamper probes all raised `ValueError`.

### Hash impact

For the representative seed-731 configuration, these physical semantic hashes
were identical before and after A1:

```text
truth_hash          9b9545b069cdf3c0feb5e636e45213a1a17bf49dd18cfb1d7ef0c53a8152a71d
sensor_payload_hash 2fe16d091f43d3c0c24cde6044ecbba043ff7f1f8b8bf20ffb1edbe19e6da38a
event_order_hash    02bdea51896c359f66dd489f363aecd5d779cd0b64fa29c612d30f62f65ef125
dataset_hash        60607c5f078fd170392ec58846b44e8c3e43157509e1a7f74628d1ba9fa798e7
```

The event-schema source fingerprint changed from
`f6bb1af5d458f7657bdc770b56601d66763d1d20c85c20277a28c81df284361f`
to `7ec71749c5ac3b1b65a0cba87c4a6e494c50a90d4930325b5664241af615b449`.
Because source fingerprints are manifest identity, the synthetic
`manifest_hash` changed from
`cfa52f175fa349745496a24d541fd2ffc4ee7559c6c756d867a1827df6822696`
to `0db994ee8ded23bc8c085a482431ba0e551626b58588f918bed19a4b00fef1d4`.
No physical dataset semantic hash changed.

### Dirty-tree integrity and scope

The approved current-tree baseline was preserved in
`preflight_snapshots/03A1_20260801T022617Z/`. Comparison of all pre-existing
dirty paths outside the exact amendment allowlist found zero status or content
mismatches and zero concurrent external additions. Staged paths remained zero;
Gate A fingerprints and noneditable Gate B1 fingerprints were unchanged.
`unit_st_synthetic.py` was unchanged, and the Gate B2 source/test targets
remained absent.

Gate B1 reapproval: **GO**. Gate B2 may be retried under a separate execution
contract using the exact generator ID `basilisk-unit-st-v1`.
