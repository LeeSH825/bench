# Phase 1A Gate B1 Amendment A1 report

## Status

```text
Status: PASS_GATE_B1_AMENDMENT_A1

Manifest identity separation: PASS
Synthetic generator regression: PASS
Second generator identity extension: PASS
Expected-ID validation: PASS
Malformed/empty ID rejection: PASS
Schema mismatch rejection: PASS
Identity corruption detection: PASS
NPZ/schema invariance: PASS
Serialization/hash: PASS
Synthetic replay invariance: PASS
Gate A regression: PASS
Gate B1 regression: PASS
Legacy regression: PASS
Dirty-tree integrity: PASS

Gate B1 reapproval: GO
Gate B2 retry authorized: YES
```

Authorization is limited to retrying Gate B2 under its own execution contract.
No Basilisk implementation, frame proof, sensor generation, metric, runner,
registry, visualization, or neural-model work was started here.

## Implementation

The blocker was a synthetic-only equality check in the common manifest
serializer. Amendment A1 retained `schema_version=p1a-mekf-events-v1` as the
schema identity and generalized only `generator_id` to a strict deterministic
version form. The accepted syntax is
`<lowercase-family>-v<positive-integer>` with no surrounding whitespace, zero
version, leading-zero version, uppercase, or underscore.

`load_event_dataset` now accepts optional keyword-only
`expected_generator_id`. Existing calls without that argument remain valid and
still receive recorded-ID, supported-schema, canonical-JSON, NPZ, and semantic
hash validation.

Modified or created target files were:

```text
bench/tasks/generator/mekf_events.py
tests/test_mekf_events.py
docs/research/phase1a/P1A_EVENT_SCHEMA_CONTRACT.md
docs/research/phase1a/P1A_GATE_B1_TEST_MATRIX.md
docs/research/phase1a/P1A_GATE_B1_AMENDMENT_A1_CONTRACT.md
experiments/phase1a/reports/P1A_GATE_B1_VALIDATION_REPORT.md
experiments/phase1a/reports/P1A_GATE_B1_AMENDMENT_A1_REPORT.md
```

The other editable test targets were not changed because the existing replay
and generator regressions already exercised the preserved contracts.

## Before and after test results

Every command used Python
`/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python`, set
`PYTHONDONTWRITEBYTECODE=1`, and disabled the pytest cache plugin.

| Suite | Before | After |
|---|---:|---:|
| Gate A (`test_mekf_conventions.py`, `test_mekf_core.py`) | 55 passed | 55 passed |
| Gate B1 (`test_mekf_events.py`, `test_unit_st_synthetic.py`, `test_mekf_replay.py`) | 39 passed | 55 passed |
| Designated legacy regression | 18 passed, 5 subtests passed | 18 passed, 5 subtests passed |

No tolerance was relaxed and no skip, xfail, jitter, pseudo-inverse, or expected
legacy value change was introduced.

## New identity tests

- Both `synthetic-unit-st-v1` and `basilisk-unit-st-v1` save, strict-load, and
  reproduce hashes using a deterministic in-memory fixture; no Basilisk runtime
  is involved.
- Expected generator ID match succeeds and mismatch raises explicit
  `ValueError`.
- `""`, `" "`, `"\t"`, unversioned, uppercase, underscore, `v0`, `v01`, and
  non-string IDs are rejected.
- A valid generator ID changed without updating its recorded manifest hash is
  detected as an artifact semantic hash mismatch.
- Unsupported schema identity is rejected on save and load.
- Artifacts remain exactly `manifest.json`, `truth.npz`, and `events.npz`; NPZ
  keys, dtype, rank, values, `allow_pickle=False`, and object-array rejection
  remain locked.
- The two-ID × five-seed property sweep passed all 10 combinations, including
  expected-ID mismatch, schema mismatch, identity corruption, exact event/truth
  arrays, and exact direct-replay equivalence.

## Synthetic semantic and replay regression

For the seed-731 representative configuration, the before/after values were
identical for:

```text
truth_hash          9b9545b069cdf3c0feb5e636e45213a1a17bf49dd18cfb1d7ef0c53a8152a71d
sensor_payload_hash 2fe16d091f43d3c0c24cde6044ecbba043ff7f1f8b8bf20ffb1edbe19e6da38a
event_order_hash    02bdea51896c359f66dd489f363aecd5d779cd0b64fa29c612d30f62f65ef125
dataset_hash        60607c5f078fd170392ec58846b44e8c3e43157509e1a7f74628d1ba9fa798e7
```

The `mekf_events.py` source fingerprint changed from
`f6bb1af5d458f7657bdc770b56601d66763d1d20c85c20277a28c81df284361f`
to `7ec71749c5ac3b1b65a0cba87c4a6e494c50a90d4930325b5664241af615b449`.
That fingerprint is intentionally part of manifest identity, so only the
representative synthetic `manifest_hash` changed, from
`cfa52f175fa349745496a24d541fd2ffc4ee7559c6c756d867a1827df6822696`
to `0db994ee8ded23bc8c085a482431ba0e551626b58588f918bed19a4b00fef1d4`.

Existing serialized round-trip replay and q/-q replay tests remained exact.
The property sweep also compared every replay evidence array and final state
before/after load for every ID/seed combination.

## Second identity and negative probes

The pre-amendment blocker evidence recorded:

```text
ValueError: manifest generator_id must equal 'synthetic-unit-st-v1'
```

The post-amendment probe recorded:

```text
SECOND_ID_SAVE_LOAD_PASS recorded=basilisk-unit-st-v1
EXPECTED_ID_MISMATCH_PASS ValueError: manifest generator_id mismatch
MALFORMED_ID_REJECTION_PASS ValueError
SCHEMA_MISMATCH_REJECTION_PASS ValueError
IDENTITY_CORRUPTION_REJECTION_PASS ValueError: artifact semantic hash mismatch
```

## Dirty-tree integrity

The snapshot is
`experiments/phase1a/preflight_snapshots/03A1_20260801T022617Z/`. It records the
repository root, 1,342 baseline status lines, tracked unstaged binary diff,
empty staged binary diff, 397 untracked paths and their recoverable archive,
pre-existing dirty fingerprints, and frozen Gate A/noneditable Gate B1 hashes.

Final comparison results:

```text
missing or changed pre-existing status outside allowlist: 0
pre-existing dirty content hash differences outside allowlist: 0
external artifact ledger entries: 0
Gate A frozen fingerprint differences: 0
Gate B1 noneditable fingerprint differences: 0
staged paths: 0
unit_st_synthetic.py differences: 0
Gate B2 source/test created: no/no
```

No reset, restore, clean, stash, stage, commit, push, merge, rebase, switch, or
checkout operation was performed.

## Evidence

Primary evidence files are:

```text
experiments/phase1a/agent_logs/03A1_baseline_gate_a.txt
experiments/phase1a/agent_logs/03A1_baseline_gate_b1.txt
experiments/phase1a/agent_logs/03A1_baseline_legacy.txt
experiments/phase1a/agent_logs/03A1_baseline_synthetic_hashes.txt
experiments/phase1a/agent_logs/03A1_identity_tests_first_run.txt
experiments/phase1a/agent_logs/03A1_generator_identity_property_sweep.txt
experiments/phase1a/agent_logs/03A1_manifest_compatibility_probe.txt
experiments/phase1a/agent_logs/03A1_post_gate_a.txt
experiments/phase1a/agent_logs/03A1_post_gate_b1.txt
experiments/phase1a/agent_logs/03A1_post_legacy.txt
experiments/phase1a/agent_logs/03A1_post_synthetic_hashes.txt
experiments/phase1a/agent_logs/03A1_dirty_tree_integrity_final.txt
```
