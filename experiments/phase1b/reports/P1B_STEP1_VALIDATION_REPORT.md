# Phase 1B Step 1 Validation Report

## 1. Decision

```text
Status: PASS_P1B_STEP1_UNIT_ST_CLASSICAL

Implementation/tests: PASS
C1 stationary baseline: PASS
Fixed Q/R tuning: PASS
Mismatch sensitivity: PASS
Long-horizon subset: PASS
C2 process-uncertainty pilot: COMPLETE
C3 ST-reliability pilot: COMPLETE
C5 RMS-matched A/B pilot: COMPLETE
Paired Monte Carlo N>=50 per required condition: PASS
Oracle/fixed information boundary: PASS
Same raw sensor realization: PASS
Gate C metrics: PASS
Phase 1A regressions: PASS
Dirty-tree integrity: PASS

Phase 1B Step 1: GO
Next authorized step: Phase 1B Step 2 sensor expansion and combined-event validation
```

Step 2 was not started.

## 2. Contract resolution and scope

The two requested `docs/research/phase1b/...` input paths were absent. Unique
same-name files existed under `docs/research/phase1a/`; those complete files
were read and used as the intended handoff and highest-priority execution
contract. All supporting Phase 0/1A contracts, reports, source and tests named
by that prompt were also read.

No branch, HEAD, history or prior commit delta was used as an approval
condition. No magnetometer, sun sensor, C4, gating, neural model or training
path was started.

## 3. Created files

Source/config/tests:

- `bench/tasks/generator/unit_st_regimes.py`
- `bench/experiments/phase1b_unit_st_classical.py`
- `bench/configs/suite_phase1b_unit_st_classical.yaml`
- `tests/test_phase1b_unit_st_regimes.py`
- `tests/test_phase1b_unit_st_classical.py`

Contracts:

- `docs/research/phase1b/P1B_UNIT_ST_CLASSICAL_BENCHMARK_CONTRACT.md`
- `docs/research/phase1b/P1B_EXPERIMENT_MATRIX.md`
- `docs/research/phase1b/P1B_STEP1_TEST_MATRIX.md`

Reports:

- `experiments/phase1b/reports/P1B_UNIT_ST_BASELINE_REPORT.md`
- `experiments/phase1b/reports/P1B_PROBLEM_EXISTENCE_REPORT.md`
- `experiments/phase1b/reports/P1B_IDENTIFIABILITY_PILOT_REPORT.md`
- this validation report

Generated provenance/results are below
`experiments/phase1b/preflight_snapshots/01_20260801T173416Z`,
`experiments/phase1b/agent_logs`,
`experiments/phase1b/results/unit_st_classical_v1`, and
`experiments/phase1b/manifests/unit_st_classical_v1`.

No existing Phase 1A source, config, test or contract was modified.

## 4. Implementation evidence

- Raw typed sensor artifacts and oracle sidecars are separate and have separate
  semantic hashes.
- C2 changes only event-window gyro measurements; C3 changes only event-window
  ST quaternions. Truth, timing, event order and unaffected streams are exact.
- Fixed/tuned replay accepts no sidecar. Oracle and wrong-side replay use a
  forward-only current-event cursor.
- All-one fixed replay is bit-exact to Phase 1A direct replay for q/b/P/r/S.
- Every filter operation calls frozen Gate A APIs; every canonical metric calls
  frozen Gate C APIs.
- Fixed/tuned deployable artifacts contain only policy ID and fixed Q/R scales.

## 5. Tests and Phase 1A regressions

| Suite | Before | After | Status |
|---|---:|---:|---|
| Gate A | 55 passed | 55 passed in 1.59 s | PASS |
| Gate B1 | 55 passed | 55 passed in 9.45 s | PASS |
| Gate B2 | 67 passed | 67 passed in 12.61 s | PASS |
| Gate C | 43 passed | 43 passed in 4.48 s | PASS |
| D1 bridge | 24 passed | 24 passed in 12.68 s | PASS |
| CP4 integration | 22 passed | 22 passed in 12.08 s | PASS |
| Legacy | 18 + 5 subtests | 18 + 5 subtests in 8.44 s | PASS |
| New Step 1 | n/a | 52 passed in 0.97 s | PASS |

The actual Phase 1A smoke CLI ran twice against one fresh `/tmp` cache. First
run: six `fresh_generation`; second: six `verified_cache_hit`. All six dataset
hashes matched across runs. Visualization emission remained off. The required
smoke command produced its normal ignored `reports/` and `runs/` runtime
artifacts; it did not alter the tracked patch.

No tolerance relaxation, skip, xfail, inverse, pseudo-inverse, jitter or
clipping was used.

## 6. Tuning, debug and pilot

Single-seed Basilisk C1/C2/C3 debug: PASS. The stationary fixed tuning evaluated
exactly 42 train/validation candidates and froze:

```text
F-TUNED = (s_Qg=0.125, s_Qb=0.125, s_R=8.0)
```

C5 selected `alpha_R=1.08` on the pilot validation 17; validation innovation
RMS difference was 0.396%. Independent test difference was 1.485%, below the
locked 5% requirement.

The full pilot workload was 84 generated trajectories, nine conditions, 50
paired test trajectories/condition, 1,950 policy/trajectory records and about
234,000 filter event steps. Initial complete pilot runtime was 148.06 s. Results
occupy 8.1 MiB and sensor/oracle manifests 43 MiB. Checkpoint granularity is one
canonical JSON per scenario/policy/trajectory.

Long horizon completed 10 stationary trajectories at T=600 s. F-BASE and
F-TUNED had zero divergence and no SPD failure. F-TUNED's long-horizon
attitude/consistency penalty is reported without post-test retuning.

## 7. Scientific summary

- H1: preliminary support. C2 shows a modest monotonic process-consistency
  effect; C3 shows a strong severity-dependent fixed-filter degradation.
- H2: preliminary support for C3 attitude/consistency and C2 consistency. C2
  attitude RMSE oracle benefit was not resolved.
- H3: preliminary support from C2 wrong-side harm and C3 correct-side benefit.
- H4: preliminary support only for scalar innovation RMS in this paired UNIT-ST
  construction. Raw gyro increments distinguish A/B, so no general
  indistinguishability claim is made.

## 8. Dirty-tree integrity

Preflight snapshot: `01_20260801T173416Z`.

```text
tracked unstaged binary patch vs start: byte-for-byte equal
staged binary patch vs start:           byte-for-byte equal
frozen source/test/config hashes:       84 checked, 0 mismatches
baseline untracked source archive:      exact compare, 0 mismatches
new untracked paths outside allowlist:  0
```

Two tracked bytecode files changed during an early import before bytecode
suppression was enabled. They were reconstructed exactly to the execution-start
snapshot using the saved binary patch/index; the final whole tracked patch
comparison is exact. No user source or data was restored to repository state.

No reset, restore, clean, stash, stage, commit, push, merge, rebase, switch or
checkout was run.

## 9. Blocking and deferred items

Blocking issue: none.

Deferred findings for an explicitly authorized next step:

- F-TUNED is over-conservative and has a long-horizon penalty; retain F-BASE as
  the stable reference rather than silently promoting F-TUNED.
- Recovery equals one gyro sample in this profile and is not discriminative.
- C2 oracle improves consistency but not measured attitude RMSE.
- Sensor expansion/C4/outlier/latency and all learned models remain unstarted.

Phase 1B Step 2 is recommended only as the next separately authorized action;
there is no automatic continuation from this report.
