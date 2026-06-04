# Official Plan Policy

This benchmark separates diagnostic runs from paper-style comparisons with
`init_id:track_id` plan identity.

## Current Fig5a Official Filter

`--fig5a-official-plans` keeps only the rows below:

| model_id | official plan | status |
| --- | --- | --- |
| `mb_kf_oracle`, `mb_kf_*`, `oracle_kf`, `nominal_kf`, `oracle_shift_kf` | `pretrained:frozen` | official baseline |
| `kalmannet_tsp` | `trained:frozen` | official NN comparison |
| `split_knet` | `trained:frozen` | official NN comparison |
| `adaptive_knet` | none | diagnostic only until the Adaptive/HKNet SoW path is integrated and validated |
| `maml_knet` | none | diagnostic only until Lorenz/checkpoint compatibility is resolved |

Default Fig5a plots still show all successful plans, split by
`model_id | init_id:track_id`, so high-dB diagnostic runs remain visible and
traceable.

## Why Adaptive-KNet Is Diagnostic-Only

The current bench adapter imports `mnets.KNet_mnet.KalmanNetNN` as a plain
trainable KalmanNet. The ICASSP24 Adaptive-KNet reference path uses
hypernetwork/context-modulated KalmanNet variants, mixed noise-ratio training
sets, and SoW/noise-ratio inputs or search. The current adapter does not pass
SoW into the model and its budgeted adaptation path is a bench-side
observation-reconstruction update, not the reference SoW search.

Until that paper path is implemented and revalidated, Adaptive-KNet results are
useful diagnostics but should not be included in official Fig5a comparisons.
The official adapter design is tracked in
`reports/adaptive_hknet_official_adapter_design.md`; any future official path
should use a new model id such as `adaptive_hknet`, not reinterpret historical
`adaptive_knet` runs.

## Why MAML-KalmanNet Is Diagnostic-Only

The bundled MAML checkpoints are compatible with the original 2D synthetic
rotation tasks, not the current 3D Lorenz partial-observation tasks. On Lorenz,
the adapter logs zero compatible tensors for the bundled checkpoint and trains
from initialized weights. Controlled Lorenz tests also show high sensitivity to
noise-grid scenarios and nonlinear learner assumptions.

Until checkpoint/task compatibility and the intended meta-test adaptation policy
are resolved, MAML-KalmanNet results are diagnostic and should not be included
in official Fig5a comparisons.
Each `maml_knet` run writes `maml_checkpoint_compatibility.json` in its run
directory. The detailed policy is tracked in
`reports/maml_checkpoint_compatibility_policy.md`.

## Usage

Use default report mode for debugging:

```bash
python -m bench.reports.make_report --suite-yaml <suite.yaml> --input-scope all_runs --fig5a-plot
```

Use official mode for paper-style comparisons:

```bash
python -m bench.reports.make_report --suite-yaml <suite.yaml> --input-scope all_runs --fig5a-plot --fig5a-official-plans
```
