# Generator Contract v1 (TG0)

## Interface

All task-family generators should follow this bench-internal signature:

```python
generate(task_cfg, split_cfg, seed, rng, device=None) -> GeneratorOutput
```

`GeneratorOutput` must include:

- `x`: `float32` `[N,T,x_dim]` (canonical `NTD`)
- `y`: `float32` `[N,T,y_dim]` (canonical `NTD`)
- `meta`: JSON-serializable dict
- `extras`: optional arrays/values (`q2_t`, `r2_t`, `SoW_t`, `task_key`, ...)

## Required Meta Keys (v1)

`enforce_meta_v1(...)` upgrades/fills metadata in append-only mode and guarantees:

- `schema_version`
- `task_family`
- `dims` (`x_dim`, `y_dim`, `T`)
- `splits` (`train`, `val`, `test` metadata)
- `ssm.true`, `ssm.assumed`
- `mismatch` (`enabled`, `kind`, `params`)
- `noise_schedule` (`enabled`, `kind`, `q2_t`, `r2_t`, `SoW_t`, optional `SoW_hat_t`)
- `switching` (`enabled`, `models`, `t_change`, `retrain_window`)

Legacy keys remain untouched (append-only compatibility).

## Determinism Expectations

For fixed `(task_cfg, split_cfg, seed)` on CPU:

- first-k value hashes of `x` and `y` must match across runs
- required meta projection hash must match across runs

This is checked by:

- `validate_artifacts(...)`
- `determinism_fingerprint(...)`
- `bench/tests/test_generator_contract_tg0.py`
