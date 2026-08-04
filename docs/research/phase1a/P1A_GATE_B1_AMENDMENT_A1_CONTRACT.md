# Phase 1A Gate B1 Amendment A1 contract

## Decision scope

Amendment A1 separates MEKF event-schema identity from dataset-generator
identity in the Gate B1 manifest serializer. It changes no event/truth NPZ
field, dtype, rank, timing, ordering, hash domain, trajectory split, or direct
replay behavior. It does not implement Gate B2 or execute Basilisk.

## Blocker cause

Before A1, `save_event_dataset` compared `manifest["generator_id"]` directly to
the analytic generator constant `synthetic-unit-st-v1`. A schema-compatible
dataset correctly identifying its producer as `basilisk-unit-st-v1` therefore
failed before serialization. The blocker was manifest identity coupling, not an
event, payload, frame, sensor, or replay incompatibility.

## Contract before Amendment A1

- `schema_version` was fixed to `p1a-mekf-events-v1`.
- `generator_id` was incorrectly fixed to `synthetic-unit-st-v1` by the common
  serializer.
- `load_event_dataset(path)` checked artifact structure and semantic hashes but
  offered no caller-specified expected generator identity.
- The exact three-file NPZ/JSON artifact and all five semantic hashes were
  already frozen.

## Contract after Amendment A1

- `schema_version` remains the sole schema identity and remains fixed to
  `p1a-mekf-events-v1`; no redundant `schema_id` field or NPZ migration exists.
- `generator_id` independently identifies the deterministic dataset-generator
  family and version.
- A valid ID matches `<lowercase-family>-v<positive-integer>`, with lowercase
  alphanumeric hyphen-separated family tokens, no surrounding whitespace, no
  zero version, and no leading-zero version.
- The common serializer and strict loader support at least
  `synthetic-unit-st-v1` and `basilisk-unit-st-v1`.
- Even without an expected-ID argument, strict loading validates the recorded
  generator ID, the supported schema identity, canonical JSON, and hashes.
- Generator-specific serializers remain forbidden.

## Public API change

The loader adds one backward-compatible keyword-only argument:

```python
load_event_dataset(path, expected_generator_id="synthetic-unit-st-v1")
```

An exact recorded/expected match succeeds. A mismatch raises `ValueError` and
is never ignored or reinterpreted. Existing `load_event_dataset(path)` callers
remain valid. `validate_generator_id(value)` exposes the one shared validation
rule for serializer/loader and future producer integration.

## Backward compatibility

The analytic generator continues to emit `synthetic-unit-st-v1` through the
existing `GENERATOR_ID` constant. Its public generation, serialization, split,
and replay calls are unchanged. Gate A source/tests and
`unit_st_synthetic.py` remain frozen. Existing event and truth arrays retain
their exact keys, dtypes, ranks, values, timing, and ordering.

## Hash impact

Both `schema_version` and `generator_id` are canonical manifest identity and
therefore affect `manifest_hash`. Changing only a valid generator ID changes
`manifest_hash` as intended. It does not change `truth_hash`,
`sensor_payload_hash`, `event_order_hash`, or their composed `dataset_hash`.

Regenerating a synthetic manifest after this source amendment also changes its
recorded `bench/tasks/generator/mekf_events.py` source fingerprint, so its
`manifest_hash` changes for that second legitimate reason. The representative
seed-731 physical data hashes remain frozen and are regression-tested.

## Gate B2 identity

The exact generator identity reserved for a future Gate B2 producer is:

```text
basilisk-unit-st-v1
```

This identity authorization is serializer compatibility only. It does not
claim that Basilisk truth, frame proof, or sensor output has been implemented or
validated.
