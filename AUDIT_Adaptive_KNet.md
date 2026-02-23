# AUDIT: Adaptive-KNet Route-B S3

## Confirmed Import-Mode Entry Points

- Config / args:
  - `third_party/Adaptive-KNet-ICASSP24/simulations/config.py`
  - symbol: `general_settings()`
- System model:
  - `third_party/Adaptive-KNet-ICASSP24/simulations/Linear_sysmdl.py`
  - symbol: `SystemModel`
- Model class (pinned by adapter config):
  - `third_party/Adaptive-KNet-ICASSP24/mnets/KNet_mnet.py`
  - symbol: `KalmanNetNN`
- Reference training/eval pipeline in third_party:
  - `third_party/Adaptive-KNet-ICASSP24/pipelines/Pipeline_EKF.py`
  - symbols: `Pipeline_EKF.NNTrain`, `Pipeline_EKF.NNTest`
- Reference adaptation/noise-search components in third_party:
  - `third_party/Adaptive-KNet-ICASSP24/noise_estimator/search.py`
  - symbol: `Pipeline_NE` (`grid_search`, `innovation_based_estimation`)
  - `third_party/Adaptive-KNet-ICASSP24/noise_estimator/KF_search.py`

## Bench Call Flow (S3)

```text
bench.runners.run_suite.run_one
  -> adapter.setup(cfg, system_info, run_ctx)
  -> adapter.train(train_dl, val_dl, budget, ckpt_dir)                # init_id=trained
  -> adapter.save(...) optional fallback / or train ckpt path
  -> if track_id == budgeted:
       adapter.adapt(test_dl, budget, t0, allowed_after_t0_only, ...)
  -> adapter.eval(test_dl, ckpt_path|None, track_cfg)
  -> bench.metrics.core (official metrics only in bench)
```

## Data Layout Transform (Fixed, No Runtime Discovery)

- Bench canonical dataset: `x,y = [N,T,D]` (NTD)
- Adapter batch API in runner: `x,y = [B,T,D]` (BTD)
- Adapter internal layout:
  - sequence staging `y_repo = [B,Dy,T]`
  - per-step model input `y_t = [B,Dy,1]`
- Adapter output back to bench:
  - `x_hat = [B,T,Dx]`

## ASSUMPTION + HOW TO VERIFY

1. ASSUMPTION: Adaptive-KNet forward is step-wise (`[B,Dy,1]`), not full-sequence direct.
   - HOW TO VERIFY:
   - Open `third_party/Adaptive-KNet-ICASSP24/mnets/KNet_mnet.py`
   - Check `InitSequence()`, `KNet_step()`, `forward()` and `torch.squeeze(y,2)` usage.

2. ASSUMPTION: `config.general_settings()` must be isolated from bench CLI args.
   - HOW TO VERIFY:
   - Open `third_party/Adaptive-KNet-ICASSP24/simulations/config.py`
   - Check `argparse.ArgumentParser(...).parse_args()`.

3. ASSUMPTION: Third-party reference adaptation flow is SoW/noise search oriented and not the exact bench `adapt()` contract.
   - HOW TO VERIFY:
   - Open `third_party/Adaptive-KNet-ICASSP24/noise_estimator/search.py`
   - Check `Pipeline_NE.grid_search` / `innovation_based_estimation`.

4. ASSUMPTION: Bench adapter `adapt()` must remain unsupervised at test time (no GT x usage).
   - HOW TO VERIFY:
   - Open `bench/models/adaptive_knet.py`
   - Search `_project_state_to_obs` and adaptation loss (`MSE(y_hat_step, y_step)`).

5. ASSUMPTION: Frozen and budgeted fairness checks are enforced in runner from `budget_ledger.json`.
   - HOW TO VERIFY:
   - Open `bench/runners/run_suite.py`
   - Search `_try_call_adapt`, `_normalize_adapt_updates_per_step`, `budget_overflow`.

