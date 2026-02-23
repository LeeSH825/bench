# Bench Reports

This folder contains generated report artifacts produced from `run_dir` outputs under `runs/`.

## Baseline outputs (backward-compatible)

- `summary_<suite>.csv`
  - One row per run.
  - Primary key columns include `model_id`, `task_id`, `scenario_id`, `seed`, `init_id`, `track_id`.
  - Includes status/failure columns:
    - canonical `failure_type`
    - `failure_stage` (or phase equivalent)
  - Includes scalar metrics and ops/ledger/cache fields when present.

- `aggregate_<suite>.csv`
  - Seed aggregation per `(model_id, task_id, scenario_id, init_id, track_id)`.
  - Includes fail count/rate, aggregate `failure_type`, and scalar metric stats.
  - Includes ops/budget/cache summary columns (means/medians/rates).

- Shift-suite baseline plots (preserved):
  - `shift_recovery_<task_id>.png`
  - `severity_sweep_<task_id>_R_scale.png`

- Optional LaTeX:
  - `summary_<suite>.tex`

## S6 additive views (flag-gated)

Enable these with `bench.reports.make_report` flags:
- `--plan-views`
- `--include-ops`
- `--budget-curves`
- optional `--group-by init_id,track_id` (default is already this)

Generated artifacts:
- `plan_compare_<suite>.csv`
  - Wide plan-comparison table for the same `(model, task, scenario, seed)` across plans.
  - Includes plan-specific `status`, `failure_type`, `mse_db`, `recovery_k`, updates, cache hit, and deltas.

- `failure_by_plan_<suite>.csv`
  - Failure counts/rates by `failure_type` and plan grouping.

- `ops_by_plan_<suite>.csv`
  - Ops summary by plan grouping:
    - train/eval/adapt/total time stats
    - update usage stats
    - cache enabled/hit and train-skipped rates.

- New plots:
  - `track_compare_<task_id>_<metric>.png`
    - compares `trained,frozen` vs `trained,budgeted`.
  - `budget_curve_<task_id>_<metric>.png`
    - `adapt_updates_used` vs quality metric (`mse_db`/`recovery_k`), grouped by severity.
  - `ops_tradeoff_<suite>.png`
    - `total_time_s` vs `mse_db`, grouped by plan.

## Failure field compatibility

- Canonical field is `failure_type`.
- Legacy run directories with `failure.json` containing `category` are mapped to `failure_type` at read-time.
- Writers should not emit `category` in new runs.

## Input artifacts read by reports

- `runs/<suite>/.../metrics.json`
- `runs/<suite>/.../metrics_step.csv`
- `runs/<suite>/.../failure.json`
- `runs/<suite>/.../run_plan.json`
- `runs/<suite>/.../budget_ledger.json`
- `runs/<suite>/.../timing.csv`
- `runs/<suite>/.../config_snapshot.yaml`

## Example command

```bash
python -m bench.reports.make_report \
  --suite-yaml bench/configs/suite_plan_matrix_smoke.yaml \
  --runs-root runs \
  --out-dir reports \
  --plan-views \
  --include-ops \
  --budget-curves \
  --group-by init_id,track_id
```
