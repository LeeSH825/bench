# Fairness Policy (Route-B)

## Scope
- Applies to benchmark-official runs produced by `bench.runners.run_suite`.
- Run plans are explicit `init_id:track_id` combinations.

## Track Rules
- `frozen`:
- `adapt_updates_used` must remain `0`.
- Any nonzero adaptation update is treated as `budget_overflow`.

- `budgeted`:
- Adaptation is allowed only when track enables it.
- Enforced caps:
- `adapt_max_updates`
- `adapt_max_updates_per_step`
- Optional t0 gate: when `allowed_after_t0_only=true`, pre-`t0` updates are forbidden.

## Training Budget Rule
- `train_max_updates` uses model-specific train semantics.
- For MAML-KNet:
- budget is enforced on `train_outer_updates_used` only.
- `train_inner_updates_used` is tracked and reported separately.
- `train_updates_used` is a backward-compatible alias of `train_outer_updates_used`.

## Reporting Fairness
- Default report ingestion scope is latest run manifest (`latest_manifest`) to avoid historical run contamination.
- Legacy broad scan remains available via `--input-scope all_runs`.

## Failure Policy
- Budget/fairness violations are recorded as `failure_type=budget_overflow`.
- If both `metrics.json` (status `ok`) and stale `failure.json` exist, reporting treats the run as successful.
