# P1A-CP4 Integration Validation Report

Date: 2026-08-02 (Asia/Seoul)

Execution: Phase 1A CP4 Integration Step 2

Decision: **GO**

## 1. Scope and fixed identities

The existing Gate A MEKF, Gate B1 typed event/replay schema, Gate B2 Basilisk
generator, Gate C metrics, and D1 bridge were treated as frozen. The integrated
pair is exactly:

```text
task_family = mekf_unit_st_v1
model_id    = mekf_event_replay_v1
```

No Phase 1B benchmark, Package C work, neural model, or visualization work was
started. `bench/tasks/data_format.py` and
`bench/tasks/generator/contract.py` were not modified.

## 2. Current-tree protection and preflight

The current working tree at execution start was the approved baseline. No
branch, HEAD, commit history, or prior commit delta was inspected. No reset,
restore, clean, stash, stage, commit, push, merge, rebase, switch, or checkout
command was run.

Preflight evidence is under:

```text
experiments/phase1a/preflight_snapshots/05D2_20260801T163044Z/
experiments/phase1a/agent_logs/05D2_20260801T163044Z_*
```

The snapshot contains exact recovery copies and hashes for all three editable
existing files, the existing tracked/staged patches, status, allowlist
existence, and frozen path hashes. All five new required integration paths were
absent before execution.

## 3. Files changed

Existing files, append-only relative to the approved snapshot:

| File | Agent-only line delta |
|---|---:|
| `bench/tasks/bench_generated.py` | +359 / -0 |
| `bench/models/registry.py` | +23 / -0 |
| `bench/runners/run_suite.py` | +547 / -0 |

New files:

| File | Lines |
|---|---:|
| `bench/configs/suite_phase1a_unit_st_smoke.yaml` | 140 |
| `tests/test_mekf_runner_integration.py` | 519 |
| `docs/research/phase1a/P1A_CP4_INTEGRATION_CONTRACT.md` | 170 |
| `docs/research/phase1a/P1A_CP4_TEST_MATRIX.md` | 45 |
| `experiments/phase1a/reports/P1A_CP4_VALIDATION_REPORT.md` | this report |

Only allowed provenance snapshot/log paths were additionally created.

## 4. Implementation result

### Typed task dispatch and cache

`prepare_mekf_unit_st_v1` is independent of the legacy `DatasetArtifactsV0`
dense sequence path. It resolves the synthetic or Basilisk producer, binds the
suite seed to the generator, uses a complete configuration hash namespace, and
produces the existing strict three-file Gate B dataset.

Both fresh and cache-hit paths invoke
`load_event_dataset(..., expected_generator_id=producer_id)`. Cache validation
also recomputes current runtime/source identities, deterministic trajectory IDs,
all derived seeds, and whole-trajectory split membership. It validates all five
semantic hashes through the frozen loader. Stale or incomplete caches fail
loudly and are not overwritten.

### Registry and runner branch

The D1 bridge is registered only in a new typed-event bridge table. The legacy
`ModelAdapter` registry and `list_model_ids()` behavior are unchanged.

`run_suite.run_one` branches on either reserved task/model identity, validates
the exact pair, and returns through the CP4 path before `_load_split_npz`,
`_SeqDataset`, `_predict_batches`, legacy adapter loading, training, or
adaptation. Only `untrained:frozen` is accepted.

The filter starts from the YAML's explicit identity quaternion, zero bias,
positive `P` diagonal, time zero, and nonnegative `Q_c` diagonal. These are
recorded with units and are not truth-derived.

### Estimation, truth boundary, and metrics

The runner passes only the typed event table, trajectory ID, explicit state,
time, process noise, and verified identity to D1. Every selected whole
trajectory is fully replayed before truth is first accessed. Truth is then
joined by exact trajectory ID and exact float64 timestamp with no interpolation
or tolerance.

Only frozen Gate C functions compute attitude geodesic error, bias error, NIS,
NEES, consistency intervals, and P/S SPD diagnostics.

### Artifacts

Each trajectory NPZ contains lossless float64/int estimator state and compact
star-tracker evidence only. It contains no truth/oracle/label/future arrays and
loads with `allow_pickle=False`. The canonical compact JSON manifest records the
task/model and adapter identities, all eight dataset identity fields,
configuration hashes, split/trajectory counters, cache state, filenames, and
metric contract. Temporary sibling publication prevents valid-looking partial
artifacts.

## 5. Fresh and verified cache-hit CLI evidence

The actual command was executed twice against one newly created cache root:

```bash
PYTHONDONTWRITEBYTECODE=1 BENCH_DATA_CACHE=<fresh-temp-root> \
  /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m bench.runners.run_suite \
  --suite-yaml bench/configs/suite_phase1a_unit_st_smoke.yaml \
  --device cpu --plans untrained:frozen
```

The first command completed six `fresh_generation` runs. The second completed
six `verified_cache_hit` runs. Dataset hashes were identical across the two
invocations:

| Producer | Seed | Dataset hash |
|---|---:|---|
| synthetic | 6101 | `b039abe71d0863965f3be5b1d390c99ebb097bf8d769d4e857941dcfb2384eeb` |
| synthetic | 6102 | `d79391e4a803a64298e4451654515fb8fd58a84ac13045b0baed8c6438cc4ec9` |
| synthetic | 6103 | `9ab654fa04480cd479b45a7929192e24e2a766499a4b8711750fe8e1f084bc01` |
| Basilisk | 6101 | `25c591f24b83a47de3e0b07010fdddbd644860836dabb3ea2b97c34d5a080a66` |
| Basilisk | 6102 | `4a700930dd4604b9496b5a9b252ab12b6721af6ba614581b02ef337fe6608dcf` |
| Basilisk | 6103 | `321876d8584b39db6a337d7e65286442f8ab18266df9eba03c06a1fe15e51134` |

CLI evidence:

```text
experiments/phase1a/agent_logs/05D2_20260801T163044Z_cli_verified_fresh.txt
experiments/phase1a/agent_logs/05D2_20260801T163044Z_cli_verified_cache_hit.txt
```

## 6. Direct / D1 / runner equivalence

An additional audit loaded the actual synthetic seed-6101 sidecar and runner
artifact. Every comparison used `np.array_equal`:

```text
q_hat_NB:    direct == D1 == runner  True
b_hat_rad_s: direct == D1 == runner  True
P:           direct == D1 == runner  True
st_residual: direct == D1 == runner  True
st_S:        direct == D1 == runner  True
dataset identity:                    True
q / -q all five groups:              True
exact truth join:                    True
```

Evidence:
`experiments/phase1a/agent_logs/05D2_20260801T163044Z_exact_equivalence_audit.txt`.

Changing model display/training notes and nonsemantic task metadata also left
the dataset configuration hash, all eight identity fields, and every runner NPZ
array unchanged.

## 7. Representative Tier-0 metric evidence

The smoke suite is representative Tier-0 and is not flight-grade. All six
verified-hit outputs were finite and strictly SPD. Observed ranges were:

| Quantity | Minimum | Maximum |
|---|---:|---:|
| attitude RMSE (rad) | 0.07357146345440722 | 0.12790550074931425 |
| bias vector RMSE (rad/s) | 0.002432589154133336 | 0.004295358253420381 |
| NIS mean | 6.702601508708808 | 20.307688662264415 |
| NEES mean | 5.870541634762954 | 17.169561333143996 |
| minimum P eigenvalue | 9.973064195446497e-07 | 9.984414175508635e-07 |
| minimum S eigenvalue | 0.0010049998952765251 | 0.0010049999704965012 |

These values are integration evidence, not acceptance thresholds or flight
performance claims. NIS/NEES counts exactly matched star-tracker updates and
posterior events. Evidence is in
`05D2_20260801T163044Z_cli_metric_audit.txt`.

## 8. Stale cache, truth boundary, and failure safety

- A self-consistent cache whose recorded MEKF source hash was stale passed
  semantic self-consistency but was rejected by the CP4 current-source check.
- D1 public arguments expose no truth/oracle/label/future input.
- A one-ULP timestamp perturbation was rejected; no interpolation occurred.
- Dense loaders, dense batching, legacy adapter loading, and prediction were
  replaced with fatal sentinels while the exact runner pair still succeeded.
- `trained:frozen` was rejected before replay.
- An injected replay failure produced `failure.json` and no final or temporary
  valid-looking `mekf_replay` artifact.

## 9. Tests and commands

All test commands used the required interpreter and environment.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider tests/test_mekf_runner_integration.py

# Gate A
... -m pytest -q -p no:cacheprovider tests/test_mekf_conventions.py tests/test_mekf_core.py
# Gate B1
... -m pytest -q -p no:cacheprovider tests/test_mekf_events.py tests/test_unit_st_synthetic.py tests/test_mekf_replay.py
# Gate B2
... -m pytest -q -p no:cacheprovider tests/test_basilisk_unit_st_generator.py
# Gate C
... -m pytest -q -p no:cacheprovider tests/test_mekf_metrics.py
# D1
... -m pytest -q -p no:cacheprovider tests/test_mekf_adapter.py
# Legacy
... -m pytest -q -p no:cacheprovider tests/test_basilisk_imu_generator.py tests/test_basilisk_mrp_ekf.py bench/tests/test_generator_contract_tg0.py bench/tests/test_adcs_event_metrics.py
```

Final results:

```text
P1A-CP4 integration: 22 passed in 12.86s
Gate A:              55 passed in 1.47s
Gate B1:             55 passed in 14.05s
Gate B2:             67 passed in 14.00s
Gate C:              43 passed in 4.72s
D1 bridge:           24 passed in 13.47s
Legacy:              18 passed, 5 subtests passed in 8.10s
```

No tolerance was relaxed. No skip, xfail, jitter, pseudo-inverse, dense proxy,
or truth-derived initialization was introduced.

## 10. Dirty-tree integrity

Final pre-report integrity audit:

```text
frozen path SHA-256:             all OK
unrelated tracked paths:         943 compared
unrelated tracked mismatches:    0
staged patch unchanged:          True
allowed existing files changed:  exactly 3
data_format.py changed:          no
generator/contract.py changed:   no
```

External unrelated artifact paths remained ledger-only. Existing dirty changes
were not reset, restored, cleaned, stashed, staged, committed, or pushed.

## 11. Decision and next stage

All CP4 integration requirements passed. Phase 1A now has a complete classical
foundation from MEKF math through typed event generation, Basilisk truth,
canonical metrics, the D1 artifact bridge, and suite-runner integration.

```text
Status: PASS_P1A_CP4_INTEGRATION

Task dispatch: PASS
Registry append-only integration: PASS
Typed sidecar delivery: PASS
Verified cache identity: PASS
Fresh synthetic runner: PASS
Fresh Basilisk runner: PASS
Verified cache-hit replay: PASS
Stale cache rejection: PASS
Direct/bridge/runner exact equivalence: PASS
Same-realization preservation: PASS
Truth-free estimator boundary: PASS
Exact truth join: PASS
Lossless q/b/P artifact: PASS
Compact ST r/S artifact: PASS
Canonical Gate C metrics: PASS
q/-q invariance: PASS
No dense coercion: PASS
Training disabled: PASS
Failure/partial-artifact safety: PASS
Legacy isolation/regression: PASS
Gate A/B1/B2/C/D1 regressions: PASS
Dirty-tree integrity: PASS

P1A-CP4 Integration: GO
Phase 1A foundation: COMPLETE
Next authorized stage: Phase 1B classical MEKF benchmark completion
```

Phase 1B was not started.
