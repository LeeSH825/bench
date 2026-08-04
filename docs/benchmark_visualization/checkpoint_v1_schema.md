# Checkpoint v1 Schema

> **Successor:** schema v2 adds `training_path_id` for write-control child
> launch — see `checkpoint_v2_schema.md`. Nothing on this page changes: v1
> packages keep exactly the meaning described here, and remain valid and
> restorable. They are simply not eligible to launch a resumed child.

`CHECKPOINT_SCHEMA_VERSION = 1`. A package whose version does not match is
**refused**, never silently migrated (ADR-CSR-009).

## 1. Layout

```
<run_dir>/checkpoints/<checkpoint_id>/
    manifest.json      # typed description, readable without unpickling
    payload.pt         # torch.save package
```

One directory per checkpoint. Publication is therefore two renames into a
fresh directory, and an existing checkpoint is never overwritten — a repeat
save of the same id raises rather than clobbering.

Files ending in `.tmp-write` are incomplete writes by construction. They are
never listed, never validated as checkpoints, and never auto-deleted (a temp
file may belong to a live writer in another process).

## 2. Manifest

Everything needed to decide *"may I resume this, into this configuration?"*
lives in the manifest, so the decision never requires reading the payload.

| Field | Purpose |
|---|---|
| `schema_version` | Refuse-on-mismatch gate |
| `checkpoint_id`, `run_id`, `created_at` | Identity |
| `kind` | `periodic` / `best` / `interrupt` / `final` — a typed field, not a filename convention (ADR-CSR-007) |
| `model_id`, `implementation_id`, `variant_id` | Identity keys |
| `phase`, `subphase`, `resume_boundary` | Where in the lifecycle |
| `cursor` | `global_update`, `epoch`, `batch_plan_position`, `batch_plan_id` |
| `component_inventory` | Named slots present in the payload |
| `structural_config_hash`, `dataset_fingerprint` | Hard compatibility keys |
| `git_revision`, `submodule_revisions` | Provenance |
| `payload_uri`, `payload_bytes`, `payload_sha256` | Integrity |
| `parent_run_id`, `resumed_from_run_id`, `resumed_from_checkpoint_id` | Lineage |
| `certification`, `capabilities` | Declared envelope |

## 3. Payload

| Key | Contents |
|---|---|
| `schema_version` | Version gate |
| `cursor` | Training position |
| `model_slots` | `{"model": state_dict}` — named |
| `optimizer_slots` | `{"main": optimizer.state_dict()}` — named |
| `scheduler_slots` | `{}` (neither certified adapter has one) |
| `grad_scaler` | `None` (no AMP in the certified envelope) |
| `best_state` | Best weights, best step, best metric |
| `validation_state` | Cadence and history |
| `extra_state` | Adapter state outside `state_dict()`; also carries progress |
| `rng` | Python, NumPy, torch CPU (and CUDA when applicable) |
| `batch_plan` | Plan parameters + id |
| `resolved_run_spec` | Config snapshot |

### Why `weights_only=False`

The payload deliberately carries optimizer state, RNG tuples, and NumPy state
objects, which `weights_only=True` refuses. That is precisely why loading is
gated on (a) digest verification and (b) the path being inside the approved
control root. Checkpoints are **trusted local artifacts**; there is no upload
path and no remote load.

## 4. Publication order

```
1. payload temp write        6. payload atomic replace
2. flush + fsync             7. manifest atomic replace
3. sha256 compute            8. directory fsync
4. manifest temp write       9. registry transaction
5. flush + fsync            10. checkpoint event append
```

The digest is computed on the temp file *before* the rename, so what is
catalogued is exactly what was made durable.

Crash outcomes, all covered by tests:

| Crash point | Result | Reconciler action |
|---|---|---|
| Steps 1–5 | No manifest → not a checkpoint | Temp file reported, not deleted |
| Between 6 and 7 | No manifest → not a checkpoint | Same |
| Between 8 and 9 | Complete package, no row | Digest verified, then **adopted** |
| After 9, before 10 | Row + package, no event | Row is authoritative; nothing lost |
| Payload corrupted later | Digest mismatch | Marked `INVALID`; load refused |
| Package deleted | Row with no package | Marked `INVALID`; row kept as evidence |

## 5. Compatibility

Must match exactly to resume: `model_id`, `implementation_id`,
`structural_config_hash`, `dataset_fingerprint`, plus a payload that actually
declares model *and* optimizer slots. A payload with no optimizer slots is
rejected as a warm start, not accepted as a resume.

Permitted operational overrides: `log_level`, `telemetry_interval`,
`api_endpoint`, `poll_interval`, `run_dir`. **Learning rate is not** — changing
it is a forked continuation, not a resume, and would make "continued from
update N" a false claim.

## 6. Batch plan

The cursor is meaningful only because the batch schedule is explicit:

```python
BatchPlan(dataset_length, batch_size, seed, drop_last=False)
```

Each epoch's permutation is drawn from a generator seeded from `(seed, epoch)`,
so epochs are independent and any global position is seekable without
replaying earlier draws. `plan_id` is a content hash of the parameters; a
resume into a differently-shaped dataset is refused rather than silently
retrained on a different order.
