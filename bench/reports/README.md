# Bench Reports (Step 7)

This folder contains **generated report artifacts** produced from `run_dir` outputs under `runs/`.

## What this generates
For each suite (e.g., `shift`, `basic`) it produces:

- `summary_<suite>.csv`
  - One row per run: (suite, task_id, scenario_id, model_id, track, seed)
  - Metrics from `metrics.json` if present; failures/missing runs are marked via `status` and empty metric cells.

- `aggregate_<suite>.csv`
  - Seed aggregation per (model_id, task_id, scenario_id, track)
  - Includes fail_count / fail_rate
  - Mean / std / sem / 95% CI for scalar metrics (computed over successful runs only)

- Plots (for `shift` suite):
  - `shift_recovery_<task_id>.png` : mean MSE(t) over seeds for each (model, scenario, track), with t0 marker if available.
  - `severity_sweep_<task_id>_R_scale.png` : mean MSE vs R_scale (if scenario_settings contains `shift.post_shift.R_scale`)

- Optional LaTeX:
  - `summary_<suite>.tex` : basic LaTeX tabular export of the aggregate table

## Where it reads from
- `runs/<suite>/.../<run_dir>/metrics.json`
- `runs/<suite>/.../<run_dir>/metrics_step.csv`
- `runs/<suite>/.../<run_dir>/failure.json` (if present)
- `runs/<suite>/.../<run_dir>/config_snapshot.yaml` (used to recover scenario_settings for sweep plots)

## How to generate
From your bench repo root:

### Shift suite
```bash
python -m bench.reports.make_report \
  --suite-yaml bench/configs/suite_shift.yaml \
  --runs-root runs \
  --out-dir reports
