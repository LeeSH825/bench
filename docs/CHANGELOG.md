# Changelog

## 2026-05-20

- Added plan-aware Fig5a reporting guidance: default plots split by `model_id | init_id:track_id`, while `--fig5a-official-plans` produces paper-style official comparisons.
- Added model-specific official-plan policy documentation and excluded currently diagnostic-only Adaptive-KNet/MAML-KalmanNet integrations from official Fig5a filtering.
- Documented high `mse_db` interpretation and the invariant `mse_db == 10*log10(mse)`.
- Documented residual diagnostics under `run_dir/diagnostics/` for separating true-state scale from estimator divergence.
- Added a dry-run-first stale failure cleanup workflow via `bench.tools.cleanup_stale_failures`.
