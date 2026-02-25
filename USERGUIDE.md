 # AI-ADCS Benchmark User Guide

  ## 1. Quick Start

  ### 1.1 Prerequisites

  - Python >=3.9 is required (pyproject.toml), and this repo was validated with .venv/bin/python (Python 3.10.19).
  - Core runtime deps are declared in pyproject.toml: numpy, scipy, pandas, matplotlib, pyyaml, tqdm, torch.
  - GPU is optional in current runner behavior: requesting CUDA falls back to CPU if unavailable.
  - UNVERIFIED: one-command install flow for this exact environment.
  - How to verify: run python3 -m pip install -e . at repo root, then run .venv/bin/python -m bench.runners.run_suite --help.
  - Verified in: pyproject.toml, bench/runners/run_suite.py, README.md.

  ### 1.2 5-minute smoke run (verified commands)

  BENCH_DATA_CACHE=/tmp/bench_data_cache_userguide .venv/bin/python -m bench.tasks.smoke_data --suite-yaml bench/configs/suite_plan_matrix_smoke.yaml --task C_shift_plan_matrix_smoke_v0 --seed 0 --batch-
  size 4

  BENCH_DATA_CACHE=/tmp/bench_data_cache_userguide .venv/bin/python -m bench.runners.run_suite --suite-yaml bench/configs/suite_plan_matrix_smoke.yaml --tasks C_shift_plan_matrix_smoke_v0 --models
  adaptive_knet --seeds 0 --plans trained:frozen --device cpu

  .venv/bin/python -m bench.reports.make_report --suite-yaml bench/configs/suite_plan_matrix_smoke.yaml

  BENCH_DATA_CACHE=/tmp/bench_data_cache_userguide .venv/bin/python -m bench.runners.smoke_run --suite-yaml bench/configs/suite_plan_matrix_smoke.yaml --task-id C_shift_plan_matrix_smoke_v0 --model-id
  adaptive_knet --seed 0 --track frozen --init-id trained --device cpu

  - Verified in: bench/tasks/smoke_data.py, bench/runners/run_suite.py, bench/runners/smoke_run.py, bench/reports/make_report.py.

  ### 1.3 Where results go

  - run_suite writes per-run artifacts under runs/... using reporting.output_dir_template from suite YAML.
  - Example observed run_dir: runs/plan_matrix_smoke/C_shift_plan_matrix_smoke_v0/adaptive_knet/frozen/seed_0/scenario_3f778e0b9924.
  - When multiple plans are run in one call (--plans ...), run_suite appends init_<init_id> below scenario dir.
  - Suite summary CSV path comes from suite YAML reporting.tables.summary_csv (for example reports/summary_shift.csv).
  - Verified in: bench/runners/run_suite.py, bench/configs/suite_plan_matrix_smoke.yaml, bench/configs/suite_basic.yaml, bench/configs/suite_shift.yaml.

  ### Locked Specs (SoT)

  - Present:
  - DECISIONS.md
  - bench/configs/suite_basic.yaml
  - bench/configs/suite_shift.yaml
  - Missing in current repo state (UNVERIFIED):
  - FAIRNESS.md
  - METRICS.md
  - STEP0_CHECKLIST.md
  - How to verify missing files: run rg --files | rg 'FAIRNESS\\.md|METRICS\\.md|STEP0_CHECKLIST\\.md|DECISIONS\\.md|suite_basic\\.yaml|suite_shift\\.yaml'.
  - Verified in: repository file scan from repo root.

  ## 2. Mental Model (Concepts)

  - Task: one entry in suite tasks[], with task ID, dimensions, data sizes, noise/observation config, and optional sweep axes.
  - Suite: one YAML that bundles tasks, models, runner, metrics, and reporting.
  - Scenario: one concrete sweep point (dotted sweep keys expanded into nested scenario config).
  - Seed: top-level run seed (suite seeds or --seeds override), also used in deterministic split and generation hashes.
  - bench_generated: benchmark-owned dataset cache, canonical tensor layout [N,T,D] (NTD), stored as split NPZ files.
  - ModelAdapter: bench-facing contract (setup/train/eval/load/predict/adapt/save) used by runner.
  - Two-track fairness in implemented code:
  - Frozen: no adaptation updates allowed (adapt_updates_used must remain 0).
  - Budgeted: adaptation allowed with explicit budget and optional t0 gate.
  - “Shift” in current linear shift tasks: within-sequence change at t0 with post-shift R_scale (noise variance scaling).
  - Verified in: bench/tasks/bench_generated.py, bench/tasks/data_format.py, bench/models/base.py, bench/runners/run_suite.py, bench/tasks/generators/linear.py, bench/configs/suite_shift.yaml.

  ## 3. What This Benchmark Can Do (Capabilities Matrix)

  | Capability | Supported (Yes/No) | How to run | Outputs | Verified in |
  |---|---|---|---|---|
  | Data generation into bench_generated cache | Yes | .venv/bin/python -m bench.tasks.smoke_data --suite-yaml ... --task ... --seed ... | bench_data_cache/<suite>/<task>/scenario_<id>/seed_<seed>/
  {train,val,test}.npz | bench/tasks/smoke_data.py, bench/tasks/bench_generated.py |
  | Run full suite (all enabled tasks/models/seeds) | Yes | .venv/bin/python -m bench.runners.run_suite --suite-yaml <suite.yaml> | run_dir artifacts + suite summary CSV | bench/runners/run_suite.py |
  | Run filtered single task/model/seed | Yes | add --tasks ... --models ... --seeds ... to run_suite | one or few run_dir(s), depending on sweep size | bench/runners/run_suite.py |
  | Run one combination wrapper | Yes | .venv/bin/python -m bench.runners.smoke_run --suite-yaml ... --task-id ... --model-id ... | one run_dir and printed result block | bench/runners/smoke_run.py |
  | Deterministic split + scenario ID hashing | Yes | automatic during smoke_data / prepare_bench_generated_v0 | stable scenario_<hash12> + deterministic train/val/test splits | bench/tasks/
  bench_generated.py, bench/utils/seeding.py |
  | Shared split files across models | Yes | run same suite/task/scenario/seed with different models | all models read same NPZ split paths | bench/runners/run_suite.py (_npz_path has no model_id) |
  | Training cache (model_cache_dir) for trained plan | Yes | configure runner.model_cache_dir and run trained:* plans | cache hit/miss reflected in budget_ledger.json and run_plan.json | bench/runners/
  run_suite.py, bench/configs/suite_cache_smoke.yaml |
  | Metrics artifacts (metrics.json, per-step, timing) | Yes | produced by run_suite | metrics.json, metrics_step.csv, timing.csv | bench/runners/run_suite.py, bench/metrics/core.py |
  | Report tables/plots generation | Yes | .venv/bin/python -m bench.reports.make_report --suite-yaml ... | summary_<suite>.csv, aggregate_<suite>.csv, plots | bench/reports/make_report.py, bench/reports/
  plots.py |
  | Failures captured per run | Yes | automatic on exceptions or missing data | failure.json with failure_type, phase, context | bench/runners/run_suite.py |
  | Enable/disable policy (enabled: false) | Yes | suite runner.enabled_policy.skip_if_disabled=true | disabled tasks/models are skipped | bench/runners/run_suite.py, suite YAMLs |
  | Auto-generate missing data inside run_suite | No | N/A | run fails with status=missing_data + hint | bench/runners/run_suite.py |

  ## 4. Running the Benchmark

  ### 4.1 Running a suite

  - Base CLI:

  .venv/bin/python -m bench.runners.run_suite --suite-yaml bench/configs/suite_shift.yaml

  - Minimal filtered pattern:

  .venv/bin/python -m bench.runners.run_suite --suite-yaml <suite.yaml> --tasks <task_id> --models <model_id> --seeds <seed> --plans <init_id>:<track_id> --device cpu

  - Suite YAML controls task/model selection via tasks[], models[], and runner.enabled_policy.
  - With skip_if_disabled: true, enabled: false tasks/models are skipped; missing enabled uses policy defaults.
  - Current suite caveat: suite_basic.yaml and suite_shift.yaml include my_model, but my_model is not in adapter registry.
  - Verified in: bench/runners/run_suite.py, bench/configs/suite_basic.yaml, bench/configs/suite_shift.yaml, bench/models/registry.py.

  ### 4.2 Running a single task

  - Basic suite example (verified):

  BENCH_DATA_CACHE=/tmp/bench_data_cache_userguide .venv/bin/python -m bench.tasks.smoke_data --suite-yaml bench/configs/suite_basic.yaml --task A_linear_canonical_v0 --seed 0 --batch-size 8

  BENCH_DATA_CACHE=/tmp/bench_data_cache_userguide .venv/bin/python -m bench.runners.run_suite --suite-yaml bench/configs/suite_basic.yaml --tasks A_linear_canonical_v0 --models kalmannet_tsp --seeds 0
  --plans untrained:frozen --device cpu

  - Shift suite example (verified):

  BENCH_DATA_CACHE=/tmp/bench_data_cache_userguide .venv/bin/python -m bench.tasks.smoke_data --suite-yaml bench/configs/suite_shift.yaml --task C_shift_Rscale_v0 --seed 0 --batch-size 8

  BENCH_DATA_CACHE=/tmp/bench_data_cache_userguide .venv/bin/python -m bench.runners.run_suite --suite-yaml bench/configs/suite_shift.yaml --tasks C_shift_Rscale_v0 --models kalmannet_tsp --seeds 0 --plans
  untrained:frozen --device cpu

  - Note: for tasks with sweep axes, selecting one task_id still runs all sweep scenarios for that task.
  - Verified in: bench/runners/run_suite.py, bench/tasks/smoke_data.py, bench/utils/sweep.py, suite YAMLs.

  ### 4.3 Controlling randomness (seeds)

  - Default seeds come from suite seeds: [...].
  - Override seeds with --seeds ....
  - Dataset generation uses stable hashed seeds for system/noise/data/split paths (for deterministic cache and split behavior).
  - Runner sets deterministic global seeds via bench seeding utility.
  - Verified in: bench/configs/suite_basic.yaml, bench/configs/suite_shift.yaml, bench/runners/run_suite.py, bench/tasks/bench_generated.py, bench/utils/seeding.py.

  ### 4.4 GPU usage

  - Device selection order: --device override, else suite.runner.device.
  - If CUDA requested but unavailable, runner logs warning and uses CPU.
  - Deterministic CUDA path sets CUBLAS_WORKSPACE_CONFIG=:4096:8 when missing.
  - --precision accepts fp32/amp, but runtime AMP execution is UNVERIFIED in current runner.
  - How to verify AMP behavior: inspect bench/runners/run_suite.py for autocast/GradScaler usage.
  - Verified in: bench/runners/run_suite.py.

  ## 5. Data: bench_generated Format, Cache, and Splits

  ### 5.1 Data cache layout

  - Cache root:
  - BENCH_DATA_CACHE env var if set.
  - else <repo_root>/bench_data_cache.
  - Path format:
  - bench_data_cache/<suite_name>/<task_id>/scenario_<scenario_id>/seed_<seed>/{train,val,test}.npz.
  - scenario_id is first 12 chars of SHA1 over {task_id, scenario_cfg} canonical JSON.
  - Directory tree example:

  bench_data_cache/
    shift/
      C_shift_Rscale_v0/
        scenario_ff58a3daaa87/
          seed_0/
            train.npz
            val.npz
            test.npz

  - Verified in: bench/tasks/bench_generated.py.

  ### 5.2 Dataset schema

  - NPZ required keys: x, y, meta_json.
  - NPZ optional keys: u, F, H.
  - NPZ extras: any non-reserved key (for example q2_t, r2_t, SoW_t, SoW_dB_t, SoW_hat_t, task_key).
  - Shapes/dtypes:
  - x: [N,T,x_dim], float32
  - y: [N,T,y_dim], float32
  - F: [x_dim,x_dim], float32 (optional)
  - H: [y_dim,x_dim], float32 (optional)
  - meta_json: JSON string
  - Meta v1 required blocks (enforced/validated): schema_version, task_family, dims, splits, ssm.true, ssm.assumed, mismatch.*, noise_schedule.*, switching.*.
  - Shift-specific values observed in generated meta: noise.shift.t0, noise.shift.post_shift.R_scale.
  - Verified in: bench/tasks/data_format.py, bench/tasks/generator/schema.py, bench/tasks/generator/validate.py, bench/tasks/bench_generated.py.

  ### 5.3 Splits (train/val/test)

  - If generator returns one combined array, splits are created by deterministic permutation keyed by (split, suite, task, scenario, seed).
  - If generator provides internal split payloads, those are written directly.
  - run_suite consumes train.npz, val.npz, test.npz from cache path.
  - All models use identical split files for a fixed (suite, task, scenario, seed) because split path does not include model_id.
  - Verified in: bench/tasks/bench_generated.py, bench/runners/run_suite.py.

  ## 6. Tracks & Fairness Rules (Frozen vs Budgeted)

  ### 6.1 Frozen inference track

  - Allowed frozen plans in runner: pretrained:frozen, trained:frozen, untrained:frozen.
  - Enforced rule: if adapt_updates_used != 0, run fails with budget_overflow.
  - UNVERIFIED: additional fairness constraints (for example BN-stat updates or info-visibility rules) from FAIRNESS.md because file is missing.
  - How to verify: add FAIRNESS.md and search for Frozen, BN, parameter update.
  - Verified in: bench/runners/run_suite.py, DECISIONS.md.

  ### 6.2 Budgeted adaptation track

  - Allowed budgeted plan in runner: trained:budgeted (must have adaptation_enabled: true in track config).
  - Enforced limits in runner and adapter:
  - max_updates <= 200
  - max_updates_per_step <= 1
  - optional allowed_after_t0_only gate
  - If allowed_after_t0_only=true and per-step updates occur before t0, run fails.
  - Budget accounting is logged in budget_ledger.json; plan policy in run_plan.json.
  - Overflow/fairness violations produce failure.json with failure_type=budget_overflow.
  - Verified in: bench/runners/run_suite.py, bench/models/adaptive_knet.py, bench/tests/test_adapt_smoke_plan.py, DECISIONS.md.

  ## 7. Metrics & Reports

  ### 7.1 Scalar metrics

  - Implemented scalar metrics in run artifacts:
  - mse, rmse, mse_db
  - timing_ms_per_step, timing_std_ms_per_step
  - nll.value currently written as null with policy=NA_if_no_cov
  - shift_recovery when t0 exists
  - Shift recovery implementation currently uses fixed W=20, eps=0.05, failure_policy="cap" in runner.
  - UNVERIFIED: canonical metric policy in METRICS.md (file missing).
  - How to verify: provide METRICS.md and compare with bench/metrics/core.py.
  - Verified in: bench/runners/run_suite.py, bench/metrics/core.py.

  ### 7.2 Time-series metrics

  - metrics_step.csv columns: t,mse_t,rmse_t,mse_db_t.
  - timing.csv columns: batch_idx,batch_size,ms_predict_whole_seq.
  - Verified in: bench/runners/run_suite.py, observed run artifacts under runs/.../scenario_....

  ### 7.3 Report artifacts

  - Baseline make_report outputs:
  - reports/summary_<suite>.csv
  - reports/aggregate_<suite>.csv
  - Shift suite baseline plots:
  - reports/shift_recovery_<task_id>.png
  - reports/severity_sweep_<task_id>_R_scale.png
  - Optional add-on outputs:
  - --plan-views: plan compare / failure tables + track plots
  - --include-ops: ops table + tradeoff plot
  - --budget-curves: budget curve plot
  - --fig5a-plot: fig5a CSV/plots
  - Organized mirror (default on): reports/<suite>/<YYYY-MM-DD>/<HHMMSS>/{tables,plots,misc} and reports/<suite>/latest/....
  - Regenerate:

  .venv/bin/python -m bench.reports.make_report --suite-yaml bench/configs/suite_shift.yaml

  .venv/bin/python -m bench.reports.make_report --suite-yaml bench/configs/suite_plan_matrix_smoke.yaml --plan-views --include-ops --budget-curves

  - Verified in: bench/reports/make_report.py, bench/reports/aggregate.py, bench/reports/plots.py, bench/reports/README.md.

  ## 8. Run Directory Artifacts (What files you get per run)

  - config_snapshot.yaml: resolved suite/task/model/track/scenario/seed snapshot and NPZ paths.
  - run_plan.json: resolved plan (init_id, track_id, budgets, cache flags, shift t0, adaptation settings).
  - budget_ledger.json: train/adapt update counts and cache hit/skip accounting.
  - metrics.json: final scalar metrics, timing summary, shift recovery block, adapter metadata, run_dir pointer.
  - metrics_step.csv: per-time-step MSE/RMSE/MSE(dB).
  - timing.csv: per-batch inference timing.
  - checkpoints/model.pt: checkpoint for trained plans or saved model states.
  - checkpoints/train_state.json: train-state summary for trained plans.
  - artifacts/preds_test.npz: optional prediction dump, adapter-dependent.
  - env.txt: lightweight runtime info.
  - env.json, pip_freeze.txt, requirements.lock (if present), git_versions.txt: reproducibility snapshots.
  - stdout.log, stderr.log: run-level logs.
  - failure.json: written on missing-data or runtime failure paths.
  - Notes:
  - Early missing-data failures may contain only failure-focused files (for example failure.json, stderr.log).
  - stderr.log may be absent on clean successful runs.
  - Verified in: bench/runners/run_suite.py, bench/utils/io.py, observed run directories under runs/.

  ## 9. Testing & Validation

  ### 9.1 Smoke tests

  BENCH_DATA_CACHE=/tmp/bench_data_cache_userguide .venv/bin/python -m bench.tasks.smoke_data --suite-yaml bench/configs/suite_plan_matrix_smoke.yaml --task C_shift_plan_matrix_smoke_v0 --seed 0

  BENCH_DATA_CACHE=/tmp/bench_data_cache_userguide .venv/bin/python -m bench.runners.smoke_run --suite-yaml bench/configs/suite_plan_matrix_smoke.yaml --task-id C_shift_plan_matrix_smoke_v0 --model-id
  adaptive_knet --seed 0 --track frozen --init-id trained --device cpu

  .venv/bin/python -m bench.reports.make_report --suite-yaml bench/configs/suite_plan_matrix_smoke.yaml

  - Verified in: bench/tasks/smoke_data.py, bench/runners/smoke_run.py, bench/reports/make_report.py.

  ### 9.2 Full test suite

  .venv/bin/python -m bench.tests.run_all --device cpu

  .venv/bin/python -m bench.tests.run_gpu_checks

  - Verified CLI in: bench/tests/run_all.py, bench/tests/run_gpu_checks.py.

  ### 9.3 Common failure modes

  - Symptom: failure.json with status=missing_data and missing test.npz path. Likely cause: cache not prepared for (suite,task,scenario,seed). Fix: run bench.tasks.smoke_data for that suite/task/seed before
  run_suite. Verified in: bench/runners/run_suite.py.
  - Symptom: failure_type=budget_overflow during adapt. Likely cause: adaptation budget exhausted, per-step cap exceeded, or updates before t0 under gate. Fix: adjust runner.tracks[].adaptation_budget and/
  or plan; inspect budget_ledger.json. Verified in: bench/runners/run_suite.py, bench/models/adaptive_knet.py.
  - Symptom: model load/adapter errors for my_model. Likely cause: suite references bench.models.my_model:MyModelAdapter, but no registry entry. Fix: filter with --models ... or implement/register adapter.
  Verified in: bench/configs/suite_basic.yaml, bench/configs/suite_shift.yaml, bench/models/registry.py.
  - Symptom: io_error / import failure mentioning missing third_party repo root. Likely cause: repo.path does not exist. Fix: place required third_party repos under configured paths. Verified in: bench/
  models/kalmannet_tsp.py, bench/models/adaptive_knet.py, bench/models/maml_knet.py, bench/models/split_knet.py.
  - Symptom: TG7 dataset loader failures mentioning NCLT_ROOT or UZH_FPV_ROOT. Likely cause: env vars unset or dataset layout incomplete. Fix: set env var and ensure expected NPZ layout. Verified in: bench/
  tasks/generator/datasets/nclt.py, bench/tasks/generator/datasets/uzh_fpv.py, bench/tasks/generator/datasets/common.py, bench/tests/test_datasets_tg7.py.

  ## 10. Configuration Guide

  ### 10.1 suite YAML anatomy

  - Verified keys used by current code:
  - suite.name, suite.version, seeds
  - tasks[]: task_id, enabled, dims, dataset sizes, noise, observation, sweep
  - models[]: model_id, enabled, adapter, repo.path, model hyperparameters
  - runner: device, precision, deterministic, enabled_policy, budget, tracks, optional model_cache_dir
  - metrics: list and shift-recovery definition block
  - reporting: output template, artifact list, summary CSV path
  - Minimal annotated snippet (fields that exist in current suite files):

  suite:
    name: plan_matrix_smoke
  seeds: [0]

  tasks:
    - task_id: C_shift_plan_matrix_smoke_v0
      enabled: true
      x_dim: 2
      y_dim: 2
      sequence_length_T: 16
      dataset_sizes: { N_train: 32, N_val: 8, N_test: 8 }
      noise:
        pre_shift:
          Q: { q2: 1.0e-3 }
          R: { r2: 1.0e-3 }
        shift:
          t0: 8
          post_shift: { R_scale: 10.0 }
      observation: { H: canonical_inverse }
      sweep: {}

  models:
    - model_id: adaptive_knet
      enabled: true
      adapter: "bench.models.adaptive_knet:AdaptiveKNetAdapter"
      repo: { path: "third_party/Adaptive-KNet-ICASSP24" }

  runner:
    device: cpu
    precision: fp32
    deterministic: true
    enabled_policy:
      task_default: true
      model_default: true
      skip_if_disabled: true
    budget: { train_max_updates: 6, train_batch_size: 4, eval_batch_size: 4 }
    tracks:
      - track_id: frozen
        adaptation_enabled: false
      - track_id: budgeted
        adaptation_enabled: true
        adaptation_budget:
          max_updates: 8
          max_updates_per_step: 1
          allowed_after_t0_only: true

  reporting:
    output_dir_template: "runs/{suite.name}/{task_id}/{model_id}/{track_id}/seed_{seed}/scenario_{scenario_id}"
    tables:
      summary_csv: "reports/summary_plan_matrix_smoke.csv"

  - Verified in: bench/configs/suite_plan_matrix_smoke.yaml, bench/configs/suite_basic.yaml, bench/configs/suite_shift.yaml, bench/runners/run_suite.py.

  ### 10.2 Model configs

  - Model configs are embedded in suite YAML models[] entries; no separate model-config directory is used in current code.
  - Model selection is done by model_id and optional CLI filter --models.
  - Adapter class resolution is through model registry (get_model_adapter_class).
  - Verified in: bench/models/registry.py, bench/runners/run_suite.py, suite YAMLs.

  ## 11. Extending the Benchmark (Developer Notes)

  - Add a new task generator module under bench/tasks/generator/ and wire it into family normalization + dispatch in bench/tasks/bench_generated.py.
  - Ensure generator output follows contract (GeneratorOutput with NTD float32 x/y + meta), then passes schema/validator checks.
  - Add a new model adapter under bench/models/ implementing ModelAdapter methods (setup, train, eval, load, predict, adapt, save), then register it in bench/models/registry.py.
  - Verified in: bench/tasks/generator/contract.py, bench/tasks/generator/schema.py, bench/tasks/generator/validate.py, bench/tasks/bench_generated.py, bench/models/base.py, bench/models/registry.py.

  ## 12. Appendix

  ### A) Command Catalog (copy/paste)

  | ID | Purpose | Command | Expected key outputs |
  |---|---|---|---|
  | CMD-001 | Generate smoke dataset cache (plan matrix) | BENCH_DATA_CACHE=/tmp/bench_data_cache_userguide .venv/bin/python -m bench.tasks.smoke_data --suite-yaml bench/configs/suite_plan_matrix_smoke.yaml
  --task C_shift_plan_matrix_smoke_v0 --seed 0 --batch-size 4 | prints scenario_id and cache_dir; writes train/val/test NPZ |
  | CMD-002 | Run one filtered plan | BENCH_DATA_CACHE=/tmp/bench_data_cache_userguide .venv/bin/python -m bench.runners.run_suite --suite-yaml bench/configs/suite_plan_matrix_smoke.yaml --tasks
  C_shift_plan_matrix_smoke_v0 --models adaptive_knet --seeds 0 --plans trained:frozen --device cpu | run_dir with metrics/artifacts; summary CSV updated |
  | CMD-003 | One-combination wrapper run | BENCH_DATA_CACHE=/tmp/bench_data_cache_userguide .venv/bin/python -m bench.runners.smoke_run --suite-yaml bench/configs/suite_plan_matrix_smoke.yaml --task-id
  C_shift_plan_matrix_smoke_v0 --model-id adaptive_knet --seed 0 --track frozen --init-id trained --device cpu | prints status + run_dir |
  | CMD-004 | Generate report (baseline) | .venv/bin/python -m bench.reports.make_report --suite-yaml bench/configs/suite_plan_matrix_smoke.yaml | writes reports/summary_plan_matrix_smoke.csv, reports/
  aggregate_plan_matrix_smoke.csv |
  | CMD-005 | Generate report (plan/ops/budget views) | .venv/bin/python -m bench.reports.make_report --suite-yaml bench/configs/suite_plan_matrix_smoke.yaml --plan-views --include-ops --budget-curves |
  writes plan/failure/ops CSVs + track/budget/ops plots |
  | CMD-006 | Basic suite single-task data prep | BENCH_DATA_CACHE=/tmp/bench_data_cache_userguide .venv/bin/python -m bench.tasks.smoke_data --suite-yaml bench/configs/suite_basic.yaml --task
  A_linear_canonical_v0 --seed 0 --batch-size 8 | one basic scenario NPZ cache |
  | CMD-007 | Basic suite single-task run | BENCH_DATA_CACHE=/tmp/bench_data_cache_userguide .venv/bin/python -m bench.runners.run_suite --suite-yaml bench/configs/suite_basic.yaml --tasks
  A_linear_canonical_v0 --models kalmannet_tsp --seeds 0 --plans untrained:frozen --device cpu | one basic run entry in reports/summary_basic.csv |
  | CMD-008 | Shift suite task data prep | BENCH_DATA_CACHE=/tmp/bench_data_cache_userguide .venv/bin/python -m bench.tasks.smoke_data --suite-yaml bench/configs/suite_shift.yaml --task C_shift_Rscale_v0
  --seed 0 --batch-size 8 | three sweep scenario caches (R_scale grid) |
  | CMD-009 | Shift suite filtered run | BENCH_DATA_CACHE=/tmp/bench_data_cache_userguide .venv/bin/python -m bench.runners.run_suite --suite-yaml bench/configs/suite_shift.yaml --tasks C_shift_Rscale_v0
  --models kalmannet_tsp --seeds 0 --plans untrained:frozen --device cpu | three shift runs; reports/summary_shift.csv updated |
  | CMD-010 | Shift suite report regeneration | .venv/bin/python -m bench.reports.make_report --suite-yaml bench/configs/suite_shift.yaml | writes summary/aggregate + shift baseline plots |
  | CMD-011 | Full CPU test harness | .venv/bin/python -m bench.tests.run_all --device cpu | prints PASS/FAIL/SKIP per test; exit non-zero on failure |
  | CMD-012 | GPU checks harness | .venv/bin/python -m bench.tests.run_gpu_checks | runs GPU determinism/all-model smoke checks; skip-aware |
  | CMD-013 | Re-organize existing reports | .venv/bin/python -m bench.reports.organize_reports --reports-dir reports --configs-dir bench/configs --dry-run | prints planned moves without modifying files |

  ### B) Glossary

  - Task: one benchmark problem definition entry in suite YAML.
  - Suite: a YAML collection of tasks, models, runner policies, and reporting config.
  - Scenario: one expanded sweep configuration for a task.
  - Seed: deterministic run key for generation/splitting/training/eval reproducibility.
  - bench_generated: benchmark-owned NPZ cache mode used by runner.
  - Track: evaluation mode (frozen or budgeted).
  - Plan: <init_id>:<track_id> combination (for example trained:budgeted).
  - run_dir: per-combination artifact directory under runs/....
  - budget_ledger.json: train/adapt update accounting file.
  - failure.json: canonical per-run failure record.

  ### C) “How to verify” checklist

  - UNVERIFIED FAIRNESS.md content.
  - Verify: ensure FAIRNESS.md exists at repo root, then search Frozen, Budgeted, BN, t0, max_updates.
  - UNVERIFIED METRICS.md content.
  - Verify: ensure METRICS.md exists at repo root, then search mse_db, nll, shift_recovery, window_W, tolerance_eps.
  - UNVERIFIED STEP0_CHECKLIST.md presence.
  - Verify: ensure STEP0_CHECKLIST.md exists at repo root and matches locked decisions.
  - UNVERIFIED install command for this machine.
  - Verify: run python3 -m pip install -e ., then run .venv/bin/python -m bench.runners.run_suite --help.
  - UNVERIFIED runtime AMP behavior behind --precision amp.
  - Verify: inspect bench/runners/run_suite.py for torch.autocast/GradScaler; run same config with --precision fp32 and --precision amp, compare run_plan.json plus logs.


