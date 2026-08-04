# Phase 1A Gate A Math Validation Report

## 1. Final status

```text
PASS_GATE_A_AMENDMENT_A1
CURRENT_TREE_ACCEPTED_WITHOUT_HEAD_REVIEW: YES
Gate A final: GO
```

Gate A pure NumPy/SciPy MEKF math/core와 지정 수학 test를 구현했고, Amendment A1에서 exact-pi antipodal determinism과 `MEKFState` 내부 배열의 실제 read-only 동작을 보강했다. Gate B 또는 Prompt 3은 실행하지 않았다.

## 2. Current-tree provenance

- Repository root: `/home/dss-pc-05/bench`
- Current branch, provenance only: `benchmark-viz/stabilize-release-baseline`
- Current HEAD, provenance only: `d92cd0ce590f1ebfdf3edb756064d94cba551174`
- Baseline policy: 실행 시작 시점의 working tree 전체를 사용자 승인 기준선으로 사용
- Commit history/delta approval: 수행하지 않음
- Approval marker/reconciliation report: 요구하거나 생성하지 않음

## 3. Dirty snapshot

- Path: `experiments/phase1a/preflight_snapshots/02_math_core_20260731T032458Z/`
- Initial status entries: 1,182
- Modified: 264
- Deleted: 681
- Untracked: 237
- Staged: 없음 (`INDEX_STAGED.patch` 0 bytes)
- Tracked working-tree patch: complete binary patch, 7,041,876 bytes
- Dirty hash ledger: 1,182 rows, SHA-256/state/size/mtime
- Untracked archive: 생성하지 않음; 현재 prompt에서는 optional

Snapshot artifacts:

```text
REPO_ROOT.txt
BRANCH.txt
HEAD.txt
STATUS_BEFORE.txt
STATUS_BEFORE.z
WORKTREE_TRACKED.patch
INDEX_STAGED.patch
UNTRACKED_BEFORE.z
PREEXISTING_DIRTY_HASHES.tsv
SNAPSHOT_MANIFEST.md
```

Invalid-byte path는 filesystem surrogate-safe access와 ASCII-escaped JSON path column으로 보존했다.

## 4. Environment

```text
Python: 3.10.13
NumPy: 2.2.6
SciPy: 1.15.3
pytest: 9.1.1
```

Dependency를 설치·업그레이드하거나 config/lock file을 수정하지 않았다.

## 5. Pre-implementation baseline regression

Exact command:

```text
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -m pytest -q -p no:cacheprovider tests/test_basilisk_imu_generator.py tests/test_basilisk_mrp_ekf.py bench/tests/test_generator_contract_tg0.py bench/tests/test_adcs_event_metrics.py
```

Result:

```text
18 passed, 5 subtests passed in 4.40s
exit_code=0
duration_seconds=6.595777
```

Evidence: `experiments/phase1a/agent_logs/02_math_core_baseline_before.txt`.

## 6. Files created

### Implementation

```text
bench/estimators/__init__.py
bench/estimators/mekf.py
```

### Tests

```text
tests/test_mekf_conventions.py
tests/test_mekf_core.py
```

### Contract/report documents

```text
docs/research/phase1a/P1A_IMPLEMENTATION_CONTRACT.md
docs/research/phase1a/P1A_TEST_MATRIX.md
experiments/phase1a/reports/P1A_MATH_VALIDATION_REPORT.md
```

### Logs/snapshot

```text
experiments/phase1a/preflight_snapshots/02_math_core_20260731T032458Z/**
experiments/phase1a/agent_logs/02_math_core_baseline_before.txt
experiments/phase1a/agent_logs/02_math_core_new_tests.txt
experiments/phase1a/agent_logs/02_math_core_baseline_after.txt
experiments/phase1a/agent_logs/02_math_core_numeric_diagnostics.txt
experiments/phase1a/agent_logs/02_math_core_agent_only.patch
experiments/phase1a/agent_logs/02_math_core_agent_only_stat.txt
experiments/phase1a/agent_logs/02_math_core_changed_paths.txt
experiments/phase1a/agent_logs/02_math_core_status_after.txt
experiments/phase1a/agent_logs/02_math_core_dirty_integrity_check.tsv
```

## 7. Gate A new tests

Exact command:

```text
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python -m pytest -q -p no:cacheprovider tests/test_mekf_conventions.py tests/test_mekf_core.py
```

Result:

```text
42 passed in 0.91s
exit_code=0
duration_seconds=1.760958
```

Evidence: `experiments/phase1a/agent_logs/02_math_core_new_tests.txt` and `docs/research/phase1a/P1A_TEST_MATRIX.md`.

## 8. Post-implementation legacy regression

Exact command is identical to §5.

```text
18 passed, 5 subtests passed in 4.50s
exit_code=0
duration_seconds=6.239512
```

Evidence: `experiments/phase1a/agent_logs/02_math_core_baseline_after.txt`.

## 9. Package B evidence

| Gate | Result | Evidence |
|---|---|---|
| B1 propagation/discretization/bias sign | PASS | zero/constant-rate/bias tests, exact `F/G`, nonlinear local-transition FD, Van Loan substep and small-dt relations |
| B3 body-vector Jacobian | PASS | 100 deterministic quaternion/reference cases; relative `<=1e-6`, max absolute `<=1e-9`; locked identity sign |
| B4 sun tangent Jacobian | PASS | 100 deterministic cases; U orthogonality/rank and analytic-vs-FD criteria |
| B5 injection/reset | PASS | known correction removal, bias addition, exact `J_r` finite difference, reset SPD |
| B6 sign invariance including exact pi | PASS | ordinary/near-pi and exact-pi x/y/z/arbitrary-axis measurement and nominal antipodal pairs produce identical residual/correction/physical posterior/P |
| Convention vectors | PASS | Q01-Q06, M01, S01, ST01, INJ01, RST01, SIGN01, KF01, COV01, LONG01 |

Deterministic random seed: `20260731`.

## 10. Numerical safety evidence

- Strict SPD matrices use Cholesky plus triangular solves.
- Deliberate non-SPD P and S raise `NumericalSafetyError`.
- Joseph update and exact right-reset covariance remain symmetric and SPD.
- Qc/Qd use PSD validation without eigenvalue modification.
- NaN, Inf, invalid shapes fail before immutable input state mutation.
- No explicit inverse, pseudo-inverse, eigenvalue clipping, silent diagonal perturbation, or non-SPD repair exists.
- `MEKFState.q_NB`, `b_g`, and `P` are defensive copies with writeable flags disabled; direct element assignment raises `ValueError`.

Representative diagnostic ledger:

```text
qd_symmetrization_relative_correction=4.6507421506266225e-19
propagated_p_symmetrization_relative_correction=2.1940040356527061e-19
s_symmetrization_relative_correction=0
joseph_pc_symmetrization_relative_correction=2.5389109687178334e-19
reset_p_symmetrization_relative_correction=0
minimum_qd_eigenvalue=2.9999988578686047e-10
minimum_prior_p_eigenvalue=0.0059997431738312628
minimum_s_eigenvalue=0.030000620000172132
minimum_posterior_p_eigenvalue=0.0059992810376132228
quaternion_norm=1
jitter_used=NO
eigenvalue_clipping_used=NO
pseudoinverse_used=NO
```

Evidence: `experiments/phase1a/agent_logs/02_math_core_numeric_diagnostics.txt`.

## 11. Information and import boundaries

- `bench.estimators.mekf` imports only standard library, NumPy, and `scipy.linalg`.
- Fresh-process import test confirms Basilisk, torch, runner, model, task, metric, viz, visualization modules are not loaded.
- Estimator functions have no truth/oracle/event input parameter.
- Visualization code/test/server/GUI/dashboard was not read, modified, executed, or imported.

## 12. Agent-only diff/stat

- Changed-path ledger: `experiments/phase1a/agent_logs/02_math_core_changed_paths.txt`
- Patch: `experiments/phase1a/agent_logs/02_math_core_agent_only.patch`
- Stat: `experiments/phase1a/agent_logs/02_math_core_agent_only_stat.txt`
- Final deliverable stat: `7 files, 1,935 insertions`; provenance artifacts are listed separately to avoid self-referential patching.

Global dirty-tree stat is recorded separately and is not agent-only evidence.

## 13. Dirty-tree and allowlist integrity

```text
current working tree accepted without HEAD review: PASS
recoverable dirty snapshot: PASS
pre-existing allowlist-outside files unchanged: PASS
new changes confined to allowlist: PASS
target files newly created only: PASS
baseline regression before implementation: PASS
Gate A new tests: PASS
legacy regression after implementation: PASS
new-file whitespace check only: PASS
```

Integrity evidence path: `experiments/phase1a/agent_logs/02_math_core_dirty_integrity_check.tsv`.

## 14. Unresolved issues

No Gate A math/core blocker remains. The following are intentionally outside this Gate:

- built-wheel inclusion of the new subpackage
- sensor event schema and UNIT-ST generator
- canonical metric aggregation and NIS/NEES reporting
- runner/model registry integration
- delayed measurement handling and actual sensor Q/R profiles

## 15. Next allowed scope

Gate B/Prompt 3 was not executed. After Chat review and a separate prompt, the next scope may add typed events and deterministic UNIT-ST generation while preserving this core as the only math source of truth.

## 16. Gate A Amendment A1 scope and snapshot

Amendment input:

```text
docs/research/phase1a/P1A_GATE_A_CHAT_REVIEW.md
docs/research/phase1a/prompts/02A_CODE_AGENT_GATE_A_HARDENING_EXACT_PI_PROMPT.md
```

The Amendment execution accepted the entire current working tree at start without querying or reviewing the current branch, HEAD, commit history, or past commit delta.

- Snapshot: `experiments/phase1a/preflight_snapshots/02A_20260731T043356Z/`
- Initial Git status records: 1,221
- Exact allowlist hashes captured: 6
- Tracked binary working-tree patch: 7,046,043 bytes
- Staged patch: 0 bytes
- Existing dirty changes were not reset, restored, cleaned, stashed, staged, committed, or pushed.

Only these six exact allowlist files were changed:

```text
bench/estimators/mekf.py
tests/test_mekf_conventions.py
tests/test_mekf_core.py
docs/research/phase1a/P1A_IMPLEMENTATION_CONTRACT.md
docs/research/phase1a/P1A_TEST_MATRIX.md
experiments/phase1a/reports/P1A_MATH_VALIDATION_REPORT.md
```

Provenance was written only below `experiments/phase1a/agent_logs/02A_*` and the snapshot path above. No visualization or Gate B code was read, modified, imported, or executed.

## 17. Amendment A1 exact-pi behavior

The original implementation selected the nonnegative scalar hemisphere only when the scalar or estimate-relative dot product was strictly negative. At an explicitly zero scalar, the following representation dependence was reproduced before modification:

```text
residual([0,+1,0,0]) = [+pi,0,0]
residual([0,-1,0,0]) = [-pi,0,0]
correction max absolute difference = 5.026548245743669
posterior physical attitude difference = 1.2566370614359168 rad
```

Amendment policy:

```text
EXACT_PI_TIE_TOL = 8 * eps(float64)
                   = 1.7763568394002505e-15
```

Only when `abs(relative_scalar) <= EXACT_PI_TIE_TOL`, the first vector component whose magnitude exceeds the same tolerance is made positive by choosing `q` or `-q`. Exact zero components are stored with positive-zero sign. Outside that tie, the original shortest-arc scalar-hemisphere behavior is unchanged.

After modification, x/y/z/arbitrary-axis residual pairs are value- and byte-identical. The x-axis full-update evidence is:

```text
correction max absolute difference = 0
posterior DCM/geodesic difference = 0
covariance max absolute difference = 0
```

The SO(3) logarithm axis at exactly pi remains mathematically non-unique. This is a deterministic representation convention, not a claim that the local MEKF always converges from an exact 180-degree initial error. The large-initial-error threshold remains a later experiment.

## 18. Amendment A1 state immutability

Before Amendment A1, caller arrays were defensively copied, but all three stored arrays had `writeable=True`, and direct assignments to `state.q_NB`, `state.b_g`, and `state.P` succeeded.

After Amendment A1:

```text
defensive q_NB copy: PASS
defensive b_g copy: PASS
defensive P copy: PASS
q_NB direct assignment rejected: PASS
b_g direct assignment rejected: PASS
P direct assignment rejected: PASS
failed predict/update leaves prior unchanged: PASS
successful predict/update returns new state and leaves prior unchanged: PASS
```

The change is limited to `MEKFState`; result dataclasses were not redesigned.

## 19. Amendment A1 tests and property checks

Pre-change Gate A command:

```text
42 passed in 0.72s
exit_code=0
measured duration=1.222206248s
```

Post-change Gate A command, retaining all original tests and adding 13 Amendment cases:

```text
55 passed in 0.59s
exit_code=0
measured duration=0.98792902s
```

Specified legacy regression:

```text
18 passed, 5 subtests passed in 2.08s
exit_code=0
measured duration=3.089991448s
```

The independent deterministic sweep covered four exact-pi axes, 1,000 ordinary `q/-q` update pairs, 256 near-pi outside-tie pairs, an exact-pi nominal-sign pair, defensive copies, direct writes, and prior preservation. All antipodal update differences were zero; the largest near-pi shortest-arc round-trip error was `1.7763568394002505e-15`.

No tolerance was widened, and no test was skipped or xfailed. No jitter, pseudo-inverse, eigenvalue clipping, or numerical repair was introduced.

## 20. Amendment A1 integrity and final gate

```text
allowlist-only agent change: PASS
pre-existing dirty status/content integrity: PASS
modified-file whitespace: PASS
```

| Gate | Amendment A1 decision |
|---|---|
| B1 propagation/discretization | PASS |
| B3 body-vector Jacobian | PASS |
| B4 sun tangent Jacobian | PASS |
| B5 injection/reset | PASS |
| B6 exact-pi included | PASS |
| Numerical safety | PASS |
| State immutability | PASS |
| Gate A final | GO |

The final comparison found zero changes among the 1,221 pre-existing status/content fingerprints and exactly six changed implementation allowlist files. Four new files appeared under the already-running external root `artifacts/benchmark_write_control/20260731T043329Z/`; that root's `preflight/*` files existed in the starting snapshot, and the four later files share a creation time after the snapshot. They were neither read nor modified and are recorded separately in `experiments/phase1a/agent_logs/02A_concurrent_change_ledger.tsv`. Unexpected or agent-attributed paths outside the Amendment allowlist/provenance prefixes: zero.

Gate B has not started and will not start automatically after this report.
