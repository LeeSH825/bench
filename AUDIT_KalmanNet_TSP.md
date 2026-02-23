# AUDIT: KalmanNet_TSP Route-B S2

## Confirmed Import-Mode Entry Points

- Config / args:
  - `third_party/KalmanNet_TSP/Simulations/config.py`
  - symbol: `general_settings()`
- System model:
  - `third_party/KalmanNet_TSP/Simulations/Linear_sysmdl.py`
  - symbol: `SystemModel`
- Model:
  - `third_party/KalmanNet_TSP/KNet/KalmanNet_nn.py`
  - symbol: `KalmanNetNN`
- Original training pipeline (reference only):
  - `third_party/KalmanNet_TSP/Pipelines/Pipeline_EKF.py`
  - symbol: `Pipeline_EKF`

## Bench Call Flow (S2 trained,frozen)

```text
bench.runners.run_suite.run_one
  -> adapter.setup(cfg, system_info, run_ctx)
  -> adapter.train(train_dl, val_dl, budget, ckpt_dir)         # init_id=trained
  -> adapter.save(...) optional fallback / or train ckpt path
  -> adapter.eval(test_dl, ckpt_path, track_cfg)
      -> adapter.predict(...) (internal sequence step loop)
  -> bench.metrics.core (official metrics only in bench)
```

## Data Layout Transform

- Bench canonical dataset: `x,y = [N,T,D]` (NTD)
- Adapter batch API: `x,y = [B,T,D]` (BTD)
- KalmanNet_TSP stepwise forward input:
  - sequence staging `y_repo = [B,Dy,T]`
  - per-step `y_t = [B,Dy,1]`
- Adapter output back to bench:
  - `x_hat = [B,T,Dx]`

## ASSUMPTION + HOW TO VERIFY

1. ASSUMPTION: KalmanNet_TSP forward is stepwise and does not accept full `[B,D,T]` sequence directly.
   - HOW TO VERIFY:
   - Open `third_party/KalmanNet_TSP/KNet/KalmanNet_nn.py`
   - Check `forward()` and `KNet_step()` signatures and use of `torch.squeeze(y,2)`.

2. ASSUMPTION: `config.general_settings()` parsing CLI args can break bench CLI unless isolated.
   - HOW TO VERIFY:
   - Open `third_party/KalmanNet_TSP/Simulations/config.py`
   - Check `argparse.ArgumentParser(...).parse_args()`.

3. ASSUMPTION: Bench-generated splits (`train/val/test`) must be consumed as-is; no re-splitting in adapter.
   - HOW TO VERIFY:
   - Open `bench/tasks/bench_generated.py`
   - Search `save_split(train|val|test)` and deterministic split permutation logic.

4. ASSUMPTION: Frozen track must have zero adaptation updates.
   - HOW TO VERIFY:
   - Open `bench/runners/run_suite.py`
   - Search for `adapt_updates_used` validation and frozen-track check.

5. ASSUMPTION: Official benchmark metrics are computed only in bench runner/metrics layer.
   - HOW TO VERIFY:
   - Open `bench/runners/run_suite.py` and `bench/metrics/core.py`
   - Confirm adapter outputs predictions only, and metrics are derived in runner.
