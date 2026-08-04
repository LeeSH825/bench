# Config GUI Contract

What the browser may configure, where the schema comes from, and what the
preview means. The launch surface itself is in `launch_api_contract.md`.

## 1. Presets are a tracked allowlist, not a file browser

```
bench/control/config/presets.py
```

The catalog lists exactly the files **Git tracks** under `bench/configs`:

```bash
git ls-files -- bench/configs
```

Consequences, all of them deliberate:

- an untracked file dropped into `bench/configs` is invisible and not
  addressable;
- a `preset_id` is `<stem>.<12 hex of the content digest>` — an opaque handle,
  **not** a path. A path-shaped id (`../../etc/passwd`, `/etc/passwd`) simply
  does not match anything, so traversal is not a case that has to be defended
  against, it is a case that cannot be expressed;
- a symlink whose target resolves outside the preset root is refused after
  resolution, not before.

The catalog is read-only. Nothing in the GUI path writes to `bench/configs`,
and a launch re-reads the file to confirm its digest still matches.

## 2. YAML is parsed as data, under bounds

`safe_load_preset_text` uses `yaml.safe_load` — no custom tags, so
`!!python/object/apply` is a parse error, not a code path — plus:

| Bound | Value | Why |
|---|---|---|
| bytes | 512 KiB | a suite config is kilobytes |
| depth | 30 | a nesting bomb costs no bytes |
| nodes after expansion | 20 000 | alias expansion ("billion laughs") is cheap to write and expensive to load |

A syntax error is reported with line and column and surfaces as
`valid: false` with code `YAML_PARSE_ERROR`, never as a 500.

## 3. One schema, derived from the config the CLI already uses

```
bench/control/config/descriptor.py
```

The descriptor **describes** the existing dataclasses; it does not restate
them. A field the descriptor does not declare cannot be overridden from the
GUI: `apply_overrides` refuses unknown paths rather than writing them, so the
form cannot become an injection channel into the runner.

Each field carries a classification:

| Classification | Meaning |
|---|---|
| `structural` | feeds `structural_config_hash`, therefore `variant_id` — editing it makes a **different experiment** |
| `operational` | does not change run identity |
| `identity` | part of the model/run identity itself |

The form only offers the certified envelope, because offering more would
advertise something the backend refuses:

```
device: cpu        precision: fp32
num_workers: 0     gradient_accumulation_steps: 1
```

Training fields are conditional (`visible_when: model_trainable`). A
model-based filter has no learning lifecycle, so it is not shown a budget.

## 4. Form and raw YAML are one draft

Both editors resolve through the same path:

```
preset YAML → safe parse → overrides → RunSpecDraft → resolve_run_spec
```

There is no GUI-specific parser, defaulting, or coercion. Editing a budget in
the form and editing the same value in the YAML tab produce the same
`structural_config_hash` and the same `variant_id` — pinned by
`test_form_yaml_round_trip_preserves_the_draft`.

Keys the schema does not model are **preserved verbatim** in the document and
reported as `UNSUPPORTED_FIELDS_PRESERVED` (severity `warning`). The GUI does
not manage them and does not silently drop them.

## 5. Validation and preview

```
POST /api/v1/config/validate
```

Registered **unconditionally**, including in the read-only build. A preview
allocates nothing, writes nothing, and touches no worker; refusing it in
read-only mode would only push operators toward launching blind. This is the
single exception to "no POST in the default build", and the read-only test
states it explicitly rather than asserting the absence of the verb.

The response carries:

```
resolved_run_spec, canonical_yaml,
structural_config_hash, operational_config_hash, variant_id,
training_path_id, implementation_id,
launch_eligibility{eligible, reason_code, reason, stop_resume_available},
diff{changed_fields[], structural_changed, variant_changed},
issues[], unsupported_fields[]
```

`diff` compares **resolved specs**, not raw YAML documents. Comparing raw
documents at descriptor paths reported "0 changed fields" while the structural
hash moved, which is the worst possible answer: an operator would approve a
review screen that said nothing had changed.

## 6. Launch eligibility

Derived, never hard-coded by model name:

| Case | Result |
|---|---|
| model in `RESUMABLE_MODEL_IDS`, `control_resumable_v1` | eligible, `stop_resume_available: true` |
| model-based baseline (not trainable), `not_applicable` | eligible, `stop_resume_available: false` |
| trainable but not GUI-certified (Adaptive / MAML / ME-Split) | **not** eligible, `ADAPTER_NOT_GUI_LAUNCH_CERTIFIED` |
| invalid config | not eligible, `INVALID_CONFIG` |

A preset containing only ineligible models is listed with
`launch_support: MODEL_NOT_LAUNCHABLE` and the reason, rather than hidden — an
operator should be able to see that a config exists and why it cannot be
started here.

## 7. Provenance capture

A preview does **not** capture Git provenance: it would shell out to `git` on
every keystroke. The launch path captures it fresh at request time, so an
allocated run records the same `git_commit`, `git_dirty`, submodule revisions
and environment fingerprint the CLI records. Provenance is not an input to the
structural or operational hash, so capturing it late does not move run
identity — asserted in `test_provenance_is_captured_on_the_launch_path`.

## 8. Parity with the CLI

`tests/test_control_gui_cli_parity.py` allocates the same preset through
`bench.control.cli launch --dry-run` and through the GUI service and compares
the resolved specs field by field. Only per-run identity may differ:

```
experiment.experiment_id, identity.run_id, hashes.resolved_spec_hash
```

`provenance.environment_fingerprint` is also excluded, and the reason is a
property of the existing fingerprint, not of this tranche: `environment_document`
reports torch facts only when torch happens to be imported in the *capturing*
process, so a CLI subprocess and an in-process API can fingerprint the same
machine differently. Recorded in `known_limitations.md`.

Numerical parity (same metrics from both launch paths, tiny KNet and Split) is
gated behind `BENCH_PARITY_E2E=1` because it trains for real.

## 9. Not in this tranche

Sweeps, batch launch, GPU queueing, warm-start, evaluate-checkpoint, a saved
draft library, config upload from arbitrary paths, authentication, and GUI
launch for Adaptive / MAML / ME-Split.
