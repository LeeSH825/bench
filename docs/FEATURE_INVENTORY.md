# Versioned feature inventory and portability boundary

Inventory basis: `origin/main` at `1f21790fc609449dfef6fc20e8ae383fcba93357`
plus the portable-core packaging changes on
`codex/portable-feature-integration`.

This inventory describes only tracked source. It deliberately does not infer
features from another working tree's untracked files.

## Versioned surfaces

| Surface | Primary source | Portable wheel contract |
|---|---|---|
| Deterministic data generation | `bench/tasks/`, `bench/tasks/generator/` | Included |
| Suite execution and provenance | `bench/runners/` | Included; `bench-run-suite` |
| KF/MEKF and neural adapters | `bench/models/`, `bench/estimators/` | Included; upstream model submodules remain external |
| Metrics and reports | `bench/metrics/`, `bench/reports/` | Included; `bench-make-report` |
| Run registry and supervision | `bench/control/` | Included; `bench-control` |
| Checkpoint/stop/resume | `bench/control/checkpoints/` | Included; exact resume is certification-gated |
| Configuration validation/launch | `bench/control/config/`, API config router | Included |
| HTTP write actions | `bench/control/api/routers/actions.py` | Included but not registered unless local write mode is enabled |
| Dash control UI | `bench/ui/` | Included; New Run and action controls follow API capability gates |
| Offline Run Inspector | `viz/` | Included |
| Suite definitions | `bench/configs/*.yaml` | Included as package data |

## Control-plane behavior

The current tracked implementation is not merely a read-only prototype.
These backends are present:

- checkpoint package/catalog, digest validation, reconciliation, and
  certification;
- durable, idempotent graceful-stop requests handled at the worker boundary;
- exact-resume validation and immutable child-run launch;
- restart reconciliation for open resume actions;
- preset schema/descriptor, YAML parsing, typed override validation, identity
  hashes, preview, and launch coordination;
- API and Dash launch/stop/resume actions.

The safety boundary is mode-dependent:

- Default: observation routes and non-durable config preview are registered;
  action POST routes are absent.
- `BENCH_CONTROL_ENABLE_WRITES=1`: action routes and Dash controls are enabled,
  only on a loopback bind.
- There is no authentication. Public write mode is refused.
- Exact resume is supported only for envelopes listed as certified by
  `/api/v1/capabilities/exact-resume`.

## Runtime data flow

```text
tracked suite YAML
  -> deterministic dataset cache
  -> task x scenario x seed x model x plan
  -> run provenance, metrics, checkpoints, predictions
  -> report tables/plots or offline visualization

tracked config preset
  -> typed validation and identity preview
  -> CLI or explicitly enabled local API launch
  -> supervised worker
  -> graceful stop/checkpoint
  -> certified exact-resume child
```

Generated paths are `bench_data_cache/`, `runs/`, and `reports/`. They are
runtime products rather than package source and are not tracked. Frozen or
canonical scientific evidence under `experiments/` is a separate authority
domain and is not touched by repository-hygiene or wheel packaging.

## Package discovery and clean-source guarantee

The former `packages = ["bench"]` declaration shipped only the root package.
The portable configuration uses namespace-aware discovery for `bench.*` and
`viz.*`, includes `bench/configs/*.yaml`, and publishes the six console
scripts documented in the README.

`scripts/verify_portable_wheel.py` checks the wheel for benchmark, checkpoint,
resume, write-action, config-GUI, and visualization members plus the console
entry points and license. It also rejects required members that exist locally
but are not Git-tracked.

`scripts/verify_clean_wheel.py` is the stronger release check: it exports
`HEAD` with `git archive`, builds that clean tree, installs the wheel into a
temporary environment, imports the critical modules outside the repository,
and executes every public CLI with `--help`. Untracked modules or configs
therefore cannot create a false PASS.

## External and deferred boundaries

- KalmanNet-family upstream implementations remain pinned submodules under
  `third_party/`; the wheel does not vendor them.
- PyTorch CPU/CUDA selection is environment-specific. The project no longer
  forces a CUDA 12.1 uv index for every platform.
- Basilisk, control, visualization, and research schema dependencies are
  optional extras.
- ADCS replay/Vizard integration, SpikeRA research integration, and Phase 2
  control-plane changes are not part of this tranche.
- No history rewrite is performed. Removing generated products from the branch
  stops future tracking but does not shrink existing Git history.

## Verification commands

```bash
python scripts/verify_clean_wheel.py
python -m pytest -q tests/test_control_write_api.py \
  tests/test_control_worker_resume_child.py \
  tests/test_control_gui_cli_parity.py
```
