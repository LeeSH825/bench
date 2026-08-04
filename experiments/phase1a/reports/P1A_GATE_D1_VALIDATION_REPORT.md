# Phase 1A Gate D1 Validation Report

- Date: 2026-08-02 (Asia/Seoul)
- Status: `PASS_GATE_D1`
- Decision: `Gate D1 GO`
- Gate D2 authorization recommendation: `YES`, with no Gate D2 work started

## Created files

```text
bench/models/mekf.py
tests/test_mekf_adapter.py
docs/research/phase1a/P1A_MEKF_ADAPTER_ARTIFACT_CONTRACT.md
docs/research/phase1a/P1A_GATE_D1_TEST_MATRIX.md
experiments/phase1a/reports/P1A_GATE_D1_VALIDATION_REPORT.md
```

The recoverable baseline provenance is under:

```text
experiments/phase1a/preflight_snapshots/05D1_20260802T000000Z/
```

It contains the pre-execution porcelain status, binary tracked patches,
untracked archive, 1,513 pre-existing dirty-path fingerprints, frozen-path
hashes, allowlist collision state, and snapshot checksums. The first snapshot
command named the B2 test incorrectly; the provenance manifest was immediately
corrected to `tests/test_basilisk_unit_st_generator.py` using its measured
SHA-256, then all snapshot checksums were regenerated and verified.

## Base API compatibility decision

`ModelAdapter.predict(y_seq, ...)` and the current runner's
`_load_split_npz`/`_SeqDataset` path cannot preserve the Gate B typed event
schema: the runner converts legacy sequence observations and numeric extras to
float32, assumes a regular batch/time grid, and returns a generic prediction
without compact ST residual/S evidence.

Decision: implement `MEKFEventReplayBridge`, an explicit unregistered bridge,
not a `ModelAdapter` subclass. Its public method accepts only
`event_table`, `trajectory_id`, caller-supplied `initial_state`,
`initial_time_s`, caller-supplied `Q_c`, and `dataset_identity`.

## Implementation result

The bridge calls the frozen Gate B1 `replay_trajectory` exactly once. It copies
the returned posterior q/b/P and compact ST residual/S without numerical
conversion, maps replay event order to exact input-table row indices, validates
timestamps/codes/counts, and packages the result in `MEKFReplayArtifact`.

All artifact arrays preserve exact float64/int64/int16 dtypes, shapes, and
values; own defensive copies; and have `writeable=False`. P and S validation is
strict Cholesky-SPD with no repair. `final_state` is an immutable copy and is
bitwise equal to the final posterior row.

The source does not import Gate C metrics, Basilisk, torch, runner, registry, or
visualization code. It does not import propagation or ST update functions and
contains no inverse, pseudo-inverse, jitter, float32 conversion, training path,
or truth-table access.

## Direct/bridge equivalence

For every checked trajectory, the bridge artifact and frozen direct replay are
exactly equal for:

```text
trajectory_id
processed event count
timestamp / event_order / sensor_code
q_hat_NB / b_hat_rad_s / P
ST event order / residual / S
final q / b / P
```

The comparison uses `np.array_equal`; no numerical tolerance is applied.
Artifact arrays do not share memory with the monkeypatched direct
`ReplayResult`, proving defensive packaging.

## Synthetic, Basilisk, and serialization validation

- Analytic synthetic UNIT-ST: exact direct/bridge equivalence.
- Basilisk UNIT-ST: exact direct/bridge equivalence.
- Synthetic seed sweep: seeds 601, 602, 603, 604, 605, all exact.
- Basilisk seed sweep: seeds 701, 702, 703, all exact.
- Strict Gate B save/load round trip: direct and bridge replay remain exact.
- Strict expected generator ID is used on serialization load.

## Identity and same-realization validation

`DatasetIdentity` preserves the exact schema, generator, convention, truth,
sensor payload, event order, manifest, and dataset hash values supplied by the
verified generator/loader. It performs format validation but no semantic hash
recomputation. An expected identity mismatch fails before replay.

Bound and unbound bridge instances return identical dataset hash and numeric
artifact. Adapter ID/version are provenance-only and remain outside the data
semantic hash domain. Original and globally sign-negated raw ST quaternion
streams produce identical q/b/P/r/S and physical metrics; the separately
computed sensor/dataset identities remain distinct as required by the bytewise
semantic domain.

## Truth boundary and Gate C metric smoke

No truth array, oracle field, event label, future value, or metric enters the
estimator-facing method. The test joins truth only after artifact creation by
exact `(trajectory_id, timestamp_s)` lookup.

With that separate join, artifact-based and direct-evidence values are exactly
equal for geodesic attitude error, bias error, right-local NEES, and compact ST
NIS. P and S Cholesky checks pass for every artifact row.

## Negative and boundary validation

The test suite confirms fail-loud behavior for:

- wrong schema version and malformed SHA-256 identity;
- expected dataset identity mismatch;
- missing trajectory ID;
- initial filter time later than the first gyro event;
- invalid same-timestamp event order;
- inconsistent artifact gyro count;
- inconsistent compact ST event index;
- attempted mutation of every artifact array and final-state array.

Subprocess import inspection confirms that importing `bench.models.mekf` does
not import Basilisk, torch, runner, registry, or visualization modules. Registry
AST inspection confirms the bridge ID is not registered.

## Test results

### New Gate D1 tests

```text
tests/test_mekf_adapter.py: 24 passed in 1.59s
```

An earlier development run produced 23 passed and one test-only failure because
the new test referenced a nonexistent `BiasErrorSummary.error_rad_s` field. The
new allowlisted test was corrected to the frozen public field
`per_axis_error_rad_s`; implementation source and frozen Gate C source were not
changed. Two subsequent runs completed 24/24.

### Pre-implementation baseline

```text
Gate A:  55 passed in 1.20s
Gate B1: 55 passed in 1.90s
Gate B2: 67 passed in 3.76s
Gate C:  43 passed in 2.55s
Legacy:  18 passed, 5 subtests passed in 4.50s
```

### Post-implementation regression

```text
Gate A:  55 passed in 1.02s
Gate B1: 55 passed in 1.75s
Gate B2: 67 passed in 3.04s
Gate C:  43 passed in 2.12s
Legacy:  18 passed, 5 subtests passed in 3.21s
```

All commands used the required interpreter, `PYTHONDONTWRITEBYTECODE=1`, and
`pytest -p no:cacheprovider`. No skip, xfail, tolerance relaxation, expected
value change, or numerical fallback was used.

## Verification commands

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -m pytest -q -p no:cacheprovider tests/test_mekf_adapter.py

PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -m pytest -q -p no:cacheprovider tests/test_mekf_conventions.py tests/test_mekf_core.py
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -m pytest -q -p no:cacheprovider tests/test_mekf_events.py tests/test_unit_st_synthetic.py tests/test_mekf_replay.py
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -m pytest -q -p no:cacheprovider tests/test_basilisk_unit_st_generator.py
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -m pytest -q -p no:cacheprovider tests/test_mekf_metrics.py
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -m pytest -q -p no:cacheprovider tests/test_basilisk_imu_generator.py tests/test_basilisk_mrp_ekf.py bench/tests/test_generator_contract_tg0.py bench/tests/test_adcs_event_metrics.py
```

Read-only inspection used `pwd`, `git rev-parse --show-toplevel`, `rg`, `sed`,
`wc`, `find`, `sha256sum`, `git status`, `git diff` for the recoverable snapshot,
and Python AST parsing. No branch, HEAD, commit history, or past commit delta was
inspected. No Git mutation command was run.

## Dirty-tree integrity

- Frozen path SHA-256 check: all 13 recorded files `OK`.
- Pre-existing dirty fingerprint comparison: 0 mismatches across 1,513 paths.
- D1 source/test/doc/report additions: exact allowlist only.
- Snapshot additions: exact `05D1_*` provenance only.
- Staging area: unchanged and empty in the snapshot (`STAGED_BEFORE.patch` is
  zero bytes); no stage operation was performed.
- AST parse: PASS.
- D1 file trailing-whitespace scan: no findings.

Two unrelated visualization-owned untracked paths appeared after the approved
baseline snapshot:

```text
"\275\272\305\251\270\260\274\246 2026-07-28 180236.png"
docs/benchmark_visualization/benchmark_visualization_tool_docs/22_config_gui_launch_implementation_prompt.md
```

They were neither read nor modified and are recorded as external unrelated
non-source artifacts per the dirty-tree policy. They do not overlap any frozen,
shared-critical, MEKF target, D1 allowlist, or provenance path.

Dirty-tree integrity result: `PASS`.

### Git status and diff summary

The final whole-tree porcelain summary is:

```text
264 modified
681 deleted
584 untracked
staged diff empty
```

The ordinary tracked working-tree diff reports `945 files changed, 4526
insertions(+), 17952 deletions(-)`. The fingerprint comparison proves that this
tracked diff is the unchanged approved baseline, not Gate D1 output. Gate D1's
source, test, and two contract documents are new untracked files and therefore
do not appear in ordinary `git diff`; the validation report exists under the
contract's approved report path but is hidden from porcelain by the repository
rule `.gitignore:9:reports/`. No file was staged.

Relative to the preflight untracked list, 17 visible paths appeared: 15 are the
four visible D1 allowlist files plus `05D1_*` snapshot provenance, and two are
the external visualization-owned ledger entries above. The ignored validation
report is the fifth required D1 deliverable.

## Gate D2 exact required changes

Append these identifiers:

```text
model_id: mekf_event_replay_v1
task_family: mekf_unit_st_v1
```

Exact existing-file shortlist:

```text
bench/tasks/bench_generated.py
bench/models/registry.py
bench/runners/run_suite.py
```

Required changes are:

1. typed sidecar dispatch plus strict cache/manifest identity checks in
   `bench_generated.py`;
2. a separate typed-event bridge registry entry in `registry.py`, preserving
   the legacy `Type[ModelAdapter]` registry;
3. a `run_suite.py` branch after cache/path resolution and before
   `_load_split_npz`, `_SeqDataset`, and `_predict_batches`;
4. post-estimation exact truth join before Gate C metrics;
5. `run_dir/artifacts/mekf_replay/manifest.json` plus one strict NPZ per
   trajectory for q/b/P/r/S;
6. equality checks for all eight identity fields on generation, load, and cache
   hit;
7. direct/bridge/runner `np.array_equal` tests on one serialized realization,
   including fresh and cache-hit paths for synthetic and Basilisk producers.

`bench/tasks/data_format.py` and `bench/tasks/generator/contract.py` are not on
the shortlist. If they become necessary, Gate D2 must stop for scope extension.

Gate D2 was not started.

## Final status

```text
Status: PASS_GATE_D1

Base API compatibility decision: PASS
Direct replay reuse: PASS
Direct/adapter exact equivalence: PASS
Synthetic generator adapter: PASS
Basilisk generator adapter: PASS
Serialization round-trip equivalence: PASS
Dataset identity preservation: PASS
Same-realization independence: PASS
Truth-free estimator boundary: PASS
Lossless q/b/P artifact: PASS
Compact ST r/S artifact: PASS
q/-q invariance: PASS
Gate C metric pairing smoke: PASS
Artifact immutability: PASS
No math/replay duplication: PASS
Import/source boundary: PASS
Gate A regression: PASS
Gate B1 regression: PASS
Gate B2 regression: PASS
Gate C regression: PASS
Legacy regression: PASS
Dirty-tree integrity: PASS

Gate D1: GO
Gate D2 authorized: YES
```
