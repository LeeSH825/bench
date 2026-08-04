# Checkpoint Schema v2

Successor to `checkpoint_v1_schema.md`, which remains accurate for v1 packages
and is **not** superseded: v1 keeps exactly the meaning it always had.

## 1. Why v2 exists

Write-control child launch has to answer "which training loop produced this?".
A v1 manifest cannot answer it — the field did not exist. Rather than
retroactively reinterpret v1 packages (DND-CSR-001), v2 adds the proof.

## 2. What v2 adds

| Field | Meaning |
|---|---|
| `training_path_id` | The loop that produced the checkpoint. Required. |
| `training_path_contract_version` | Version of the training-path contract. |

Everything else is unchanged from v1: same layout, same atomic publication
order, same digest and inventory validation, same payload keys.

## 3. The version is derived, not imposed

```
save(..., training_path_id=<set>)   -> schema_version 2
save(..., training_path_id=None)    -> schema_version 1
```

A package is v2 exactly when it can prove its training path. This is why every
pre-existing v1 test still passes untouched, and why a v2 package can never
exist without the proof that makes it one.

## 4. Read policy

`SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS = (1, 2)`. Both are readable and
restorable. Anything outside the set — including a future version — is
**refused**, never migrated.

## 5. Valid is not the same as launch-eligible

Three separate questions:

| Question | v1 package | v2 (`control_resumable_v1`) |
|---|---|---|
| Valid artifact? | **yes** | yes |
| Restorable at adapter/service level? | **yes** | yes |
| May launch a write-control child? | **no** | yes |

A v1 package is refused with:

- `CHECKPOINT_SCHEMA_NOT_WRITE_CONTROL_CERTIFIED`
- `CHECKPOINT_TRAINING_PATH_UNPROVEN`

A v2 package on the legacy loop is refused with `TRAINING_PATH_NOT_RESUMABLE`.

Eligibility is decided from the **manifest and registry only**. The payload is
never unpickled to answer it, so an ineligible package is never loaded.

## 6. Consistency requirement

For a v2 package these must agree, and a mismatch is reported as
`TRAINING_PATH_MISMATCH`:

```
manifest.training_path_id
registry.runs.training_path_id (parent)
registry.checkpoints.training_path_id
certification row training_path_id
```

## 7. Certification

The certification key now includes both the checkpoint schema version and the
training path:

```
model|implementation|schema|boundary|precision|device|workers|mode|training_path
```

so the same model on the legacy loop is a different tuple and is not certified.
v1-era certification evidence from the checkpoint tranche is retained in that
tranche's report and matrix; it is not deleted.
