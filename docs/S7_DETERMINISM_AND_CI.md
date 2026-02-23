# S7 Determinism, Invariants, and CI Lanes

This document defines S7 guardrails for Route B without changing model behavior, FAIRNESS, METRICS, or D15.

## 1) Determinism policy

Reference: `DECISIONS.md` D19.

- Stability metrics:
  - `mse_db` (required)
  - `recovery_k` (shift tasks when present)
- CPU tolerance:
  - `mse_db` absolute tolerance `<= 1e-9`
  - `recovery_k` exact match (integer)
- GPU/accelerator tolerance (policy target):
  - `mse_db`: `abs<=1e-5` or `rel<=1e-5`
  - `recovery_k`: exact match preferred, fallback `abs<=1`
- Cache invariance:
  - Cache miss vs hit must not change performance metrics.
  - Only cache/ledger fields may differ (`cache_hit`, `train_skipped`, `train_updates_used`).

## 2) Artifact invariants

### run_dir invariants

- Successful runs must include:
  - `run_plan.json`
  - `budget_ledger.json`
  - `metrics.json`
  - `metrics_step.csv`
  - `timing.csv`
  - `checkpoints/model.pt` for trained plans
- Failed runs must include:
  - `failure.json` with canonical `failure_type`
  - stage info in `failure_stage` (or `phase` for compatibility)
  - no legacy `category` field in new outputs

### report invariants (S6 views)

For plan-matrix smoke reports:
- Required tables:
  - `plan_compare_<suite>.csv`
  - `failure_by_plan_<suite>.csv`
  - `ops_by_plan_<suite>.csv`
- Required plots:
  - `track_compare_*.png`
  - `budget_curve_*.png`
  - `ops_tradeoff_*.png`
- Required schema:
  - tables must expose `init_id`, `track_id`, status/failure fields, and budget/ops/cache fields.

## 3) Enforced tests

- `bench/tests/test_determinism_cpu_seed_stable.py`
- `bench/tests/test_determinism_gpu_seed_stable.py` (S8, GPU only; clean SKIP when CUDA unavailable)
- `bench/tests/test_cache_invariance.py`
- `bench/tests/test_report_schema_guardrails.py`
- `bench/tests/test_artifact_invariants_smoke.py`

Run:

```bash
python -m bench.tests.run_all --device cpu
```

GPU-only targeted run:

```bash
python -m bench.tests.run_gpu_checks
```

## 4) CI lane plan

### Fast lane (PR)

- Scope:
  - CPU smoke regression (`run_all`)
  - determinism CPU check
  - report schema guardrails
- Command:
  - `python -m bench.tests.run_all --device cpu`

### Full lane (nightly/optional)

- Scope:
  - CPU checks + representative suite/report generation
  - optional GPU determinism job on GPU runners
- Suggested commands:
  1. `python -m bench.runners.run_suite --suite-yaml bench/configs/suite_shift.yaml --device cpu`
  2. `python -m bench.reports.make_report --suite-yaml bench/configs/suite_shift.yaml --runs-root runs --out-dir reports --plan-views --include-ops --budget-curves`
  3. `python -m bench.tests.run_gpu_checks`

### Environment awareness

- If GPU is unavailable, skip GPU determinism checks and run CPU-only checks.
- Use `env.json`, `git_versions.txt`, and runner cache key digests to contextualize reproducibility drift.
- In CI, GPU job is guarded by `vars.HAS_GPU_RUNNER == 'true'` and uses GPU runner labels.
