# Phase 1A Gate C Execution Contract
## Canonical MEKF Geodesic, Bias, NIS, NEES, and SPD Metrics

당신은 `/home/dss-pc-05/bench` repository에서 Phase 1A Gate C만 수행하는 구현 agent다.

이번 단계의 목적은 검증 완료된 Gate A MEKF, Gate B1 typed replay, Gate B2
Basilisk UNIT-ST를 수정하지 않고, 연구 결과에 사용할 canonical MEKF metric을
독립 NumPy/SciPy module로 구현하고 검증하는 것이다.

Gate D runner/registry integration은 이번 실행에서 시작하지 마라.

---

# 1. 반드시 먼저 읽을 문서

다음을 처음부터 끝까지 읽어라.

```text
docs/research/phase1a/P1A_GATE_B2_FINAL_APPROVAL.md
docs/research/phase1a/P1A_BASILISK_FRAME_CONVENTION_PROOF.md
docs/research/phase1a/P1A_BASILISK_UNIT_ST_CONTRACT.md
docs/research/phase1a/P1A_GATE_B2_TEST_MATRIX.md
experiments/phase1a/reports/P1A_GATE_B2_VALIDATION_REPORT.md

docs/research/phase1a/P1A_GATE_A_FINAL_APPROVAL.md
docs/research/phase1a/P1A_IMPLEMENTATION_CONTRACT.md
docs/research/phase1a/P1A_EVENT_SCHEMA_CONTRACT.md
docs/research/phase1a/P1A_GATE_B1_AMENDMENT_A1_CONTRACT.md

docs/research/phase1a/P1A_RISK_REGISTER.md
docs/research/phase1a/P1A_IMPLEMENTATION_MAP.md

bench/estimators/mekf.py
bench/tasks/generator/mekf_events.py
bench/tasks/generator/basilisk_unit_st.py
```

다음 기존 visualization consistency 구현은 문제점 확인을 위한 read-only
참고만 허용한다.

```text
viz/analysis/consistency.py
viz/figures/panels.py
```

그 구현의 additive state difference, inverse, pseudo-inverse, jitter 경로를
새 canonical metric에 복사하지 마라.

---

# 2. 현재 승인 baseline

다음 결과를 baseline으로 사용하라.

```text
Gate A: 55 passed
Gate B1 Amendment A1: 55 passed
Gate B2: 67 passed
Legacy: 18 passed, 5 subtests passed
```

시험 개수는 provenance로 기록하되 pass/fail은 exit code와 계약 유지로 판정하라.

---

# 3. Current-tree 및 dirty-tree 정책

실행 시작 시점의 현재 working tree 전체를 사용자가 승인한 기준선으로 사용하라.

다음을 승인 조건으로 검토하거나 비교하지 마라.

```text
branch
HEAD
commit history
과거 commit delta
repository 전체 whitespace
기존 visualization artifact
```

다음을 수행하지 마라.

```text
git reset
git restore
git clean
git stash
git add
git commit
git push
git merge
git rebase
git switch
git checkout
```

실행 전에 recoverable snapshot과 기존 dirty path fingerprint를 만들고,
실행 후 allowlist 밖 기존 path의 status/content 변화가 없는지 검사하라.

외부 unrelated non-source artifact가 추가되면 읽거나 수정하지 말고 path/status
ledger에만 기록하라.

---

# 4. 동결 파일

다음 파일은 읽고 import할 수 있으나 수정 금지다.

```text
bench/estimators/__init__.py
bench/estimators/mekf.py

bench/tasks/generator/mekf_events.py
bench/tasks/generator/unit_st_synthetic.py
bench/tasks/generator/basilisk_unit_st.py

tests/test_mekf_conventions.py
tests/test_mekf_core.py
tests/test_mekf_events.py
tests/test_unit_st_synthetic.py
tests/test_mekf_replay.py
tests/test_basilisk_unit_st_generator.py

docs/research/phase0a/**
docs/research/phase1a/P1A_IMPLEMENTATION_CONTRACT.md
docs/research/phase1a/P1A_EVENT_SCHEMA_CONTRACT.md
docs/research/phase1a/P1A_BASILISK_UNIT_ST_CONTRACT.md
docs/research/phase1a/P1A_BASILISK_FRAME_CONVENTION_PROOF.md
```

Gate A quaternion/SO(3) 함수를 복사하거나 재구현하지 말고
`bench.estimators.mekf`의 검증된 helper를 import하라.

---

# 5. Exact allowlist

이번 실행에서 생성할 수 있는 source/test/doc/report는 정확히 다음이다.

```text
bench/metrics/mekf.py
tests/test_mekf_metrics.py

docs/research/phase1a/P1A_CANONICAL_MEKF_METRICS_CONTRACT.md
docs/research/phase1a/P1A_GATE_C_TEST_MATRIX.md
experiments/phase1a/reports/P1A_GATE_C_VALIDATION_REPORT.md
```

다음 provenance 경로를 생성할 수 있다.

```text
experiments/phase1a/preflight_snapshots/04_*/
experiments/phase1a/agent_logs/04_*
```

allowlist 밖 source/test/config 수정이 필요하면 수정하지 말고:

```text
BLOCKED_GATE_C_SCOPE_EXTENSION_REQUIRED
```

로 중단하고 정확한 파일과 이유를 보고하라.

---

# 6. 수정 및 구현 금지 범위

다음을 수정하거나 구현하지 마라.

```text
bench/models/**
bench/runners/**
bench/tasks/bench_generated.py
bench/models/registry.py
bench/configs/**
bench/metrics/core.py
bench/metrics/adcs_event.py
viz/**
visualization/**
pyproject.toml
uv.lock
requirements*
third_party/**
기존 suite YAML
기존 test expected value
```

다음 기능도 시작하지 마라.

```text
runner artifact integration
model adapter
registry/dispatch
cache integration
dashboard/visualization
latency/OOSM
outage/false ST
magnetometer/sun sensor
Package C experiments
KalmanNet/ANN/SNN/FPGA
```

---

# 7. Metric convention

Gate A/B2의 고정 convention을 사용하라.

```text
q_NB:
  scalar-first Hamilton
  active body-to-navigation

right error:
  q_true = q_hat ⊗ Exp_q(delta_theta)

bias error:
  delta_b = b_true - b_hat
```

canonical state error는 다음이다.

```text
delta_q     = q_hat^-1 ⊗ q_true
delta_theta = Log_q(delta_q)
delta_b     = b_true - b_hat
e           = [delta_theta, delta_b] ∈ R^6
```

`q_true` 또는 `q_hat`을 `-q`로 바꾸어도 모든 physical metric 결과가 같아야 한다.

자세 error의 주 구현은 quaternion log map이어야 한다.

```text
attitude_error_rad = ||delta_theta||
```

machine-roundoff 영역의 주 판정에 `2*acos(abs(dot))`만 사용하지 마라.

---

# 8. 필수 public 기능

정확한 함수명은 repository style에 맞게 조정할 수 있지만, 다음 기능을
하나의 `bench/metrics/mekf.py` source of truth에 제공하라.

## 8.1 Right-local state error

입력:

```text
q_hat_NB  (...,4)
b_hat     (...,3)
q_true_NB (...,4)
b_true    (...,3)
```

출력:

```text
delta_theta (...,3) rad
delta_b     (...,3) rad/s
state_error (...,6)
```

요구:

- float64
- finite
- scalar-first active convention
- q/-q invariance
- input mutation 없음

## 8.2 Attitude geodesic error

출력은 per-sample angle in rad다. 선택적으로 deg helper를 제공할 수 있다.

필수 경계:

```text
identity -> 0
known axis-angle -> known magnitude
exact pi -> pi
q/-q -> identical
near-zero -> stable
near-pi -> stable
```

## 8.3 Bias error and summary

제공할 값:

```text
per-sample per-axis bias error
vector-norm bias error
per-axis RMSE
vector RMSE
```

단위는 rad/s다. 빈 배열, NaN/Inf, shape mismatch는 fail-loud한다.

## 8.4 Star-tracker NIS

정의:

```text
NIS = r.T @ S^-1 @ r
```

입력:

```text
r (...,3) rad
S (...,3,3) rad^2
```

요구:

- Cholesky solve
- explicit inverse/pseudo-inverse 금지
- S finite/symmetric/strict SPD
- 결과 finite/nonnegative
- residual이 없는 gyro row를 0으로 채우지 않음
- 실제 ST update evidence에 대해서만 계산

## 8.5 Right-local 6D NEES

정의:

```text
NEES = e.T @ P^-1 @ e
```

입력:

```text
q_hat, b_hat, P, q_true, b_true
```

요구:

- e는 위 right-local state error
- P는 같은 timestamp/posterior tangent의 6x6 covariance
- Cholesky solve
- explicit inverse/pseudo-inverse 금지
- P finite/symmetric/strict SPD
- q/-q invariant

## 8.6 SPD diagnostics

P와 S에 공통으로 사용할 진단을 제공하라.

최소 출력:

```text
relative asymmetry
minimum eigenvalue
Cholesky success
dimension
```

진단은 matrix를 수정하지 않는다.
non-SPD를 clipping, jitter, symmetrization으로 수리하지 마라.

## 8.7 Consistency summary

NIS/NEES 배열에 대해 최소 다음을 제공하라.

```text
count
dof_per_sample
sum
mean
normalized_mean = mean / dof_per_sample
```

선택적으로 `scipy.stats.chi2` batch-sum confidence interval을 제공하라.

```text
sum(values) ~ chi2(count * dof_per_sample)
```

이는 independence와 matched Gaussian assumptions 아래의 진단임을 문서화하라.
빈 sample 정책을 명시하고 tests로 잠가라.

---

# 9. 자료 경계

Metric 함수는 evaluation을 위해 estimate와 truth를 함께 받을 수 있다.

그러나:

```text
metric output이 estimator/generator/replay에 피드백되지 않음
metric module이 filter state나 truth를 수정하지 않음
metric module이 future sample을 estimator에 전달하지 않음
metric module이 oracle Q/R 또는 event label을 생성하지 않음
```

NIS는 truth를 받지 않는다.
NEES와 attitude/bias error만 evaluation truth를 받는다.

---

# 10. 수치 안전 정책

금지:

```text
numpy.linalg.inv
scipy.linalg.inv
pinv
lstsq fallback
silent diagonal jitter
eigenvalue clipping
non-SPD repair
NaN/Inf 무시
component-wise quaternion MSE
additive quaternion subtraction NEES
```

허용:

```text
Gate A quaternion normalization/log helper
Cholesky factorization 및 triangular solve
read-only defensive copy
명시적 shape/dtype/finite validation
```

source-level test로 금지 fallback 부재를 확인하라.

---

# 11. 필수 시험

최소 다음 logical tests를 구현하라.

```text
C-01 attitude identity 및 q/-q
C-02 known x/y/z/arbitrary axis-angle
C-03 near-zero/exact-pi/near-pi
C-04 known right-local state error recovery
C-05 closed-form bias RMSE
C-06 closed-form diagonal/full-SPD NIS
C-07 closed-form diagonal/full-SPD NEES
C-08 q/-q NEES invariance
C-09 batch consistency summary와 선택적 chi-square bounds
C-10 asymmetric/nonfinite/non-SPD P/S fail-loud
C-11 shape/dtype/batch mismatch validation
C-12 input nonmutation 및 result immutability policy
C-13 import boundary
C-14 no inverse/pinv/jitter/clipping source check
C-15 small deterministic B2 replay metric smoke
C-16 timestamp/posterior pairing mismatch fail-loud
```

C-15에서는 작은 Basilisk UNIT-ST dataset과 direct replay를 read-only로 사용하여:

```text
attitude error finite
bias error finite
ST NIS finite/nonnegative
state NEES finite/nonnegative
P/S Cholesky PASS
q/-q stream metric equivalent
```

를 확인하라.

metric module 자체는 Basilisk를 import하면 안 된다.

Gate C는 interpolation이나 event alignment를 추측하지 않는다.
timestamp/count mismatch는 fail-loud한다.

---

# 12. Baseline 시험

모든 Python 명령은 다음 interpreter를 사용하라.

```text
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python
```

환경:

```text
PYTHONDONTWRITEBYTECODE=1
pytest -p no:cacheprovider
```

구현 전에 Gate A, Gate B1, Gate B2, legacy를 각각 실행하라.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_mekf_conventions.py tests/test_mekf_core.py
```

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_mekf_events.py tests/test_unit_st_synthetic.py tests/test_mekf_replay.py
```

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_basilisk_unit_st_generator.py
```

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_basilisk_imu_generator.py tests/test_basilisk_mrp_ekf.py \
  bench/tests/test_generator_contract_tg0.py bench/tests/test_adcs_event_metrics.py
```

baseline failure 시 기존 코드를 수정하지 말고:

```text
BLOCKED_BASELINE_REGRESSION
```

으로 중단하라.

---

# 13. 구현 후 시험

다음을 모두 실행하라.

1. `tests/test_mekf_metrics.py`
2. Gate A regression
3. Gate B1 regression
4. Gate B2 regression
5. 지정 legacy regression
6. metric property sweep
7. source fallback/import-boundary 검사
8. modified-file whitespace 검사
9. dirty-tree integrity 검사

skip, xfail, tolerance 완화, legacy expected value 변경으로 실패를 숨기지 마라.

---

# 14. Property sweep

최소 10개 deterministic case에서:

```text
random unit q_hat
known local delta_theta
random b_hat/delta_b
random SPD P
random SPD S
random residual r
```

를 만들고 다음을 확인하라.

```text
right-local error recovery
q/-q NEES invariance
NIS/NEES finite and nonnegative
Cholesky reference equality
input immutability
```

추가로 B2 dataset 최소 3 seeds에 대해 replay metric smoke를 수행하라.

---

# 15. 문서 산출물

## P1A_CANONICAL_MEKF_METRICS_CONTRACT.md

포함:

```text
목적과 근거
locked convention
metric 수식
shape/dtype/unit
timestamp/posterior pairing
truth boundary
NIS/NEES 사용 조건
chi-square 해석 조건
numerical safety
empty sample policy
public API
Gate D artifact 요구사항
```

## P1A_GATE_C_TEST_MATRIX.md

각 test에 ID, contract, input, expected, tolerance, actual, evidence를 기록하라.

## P1A_GATE_C_VALIDATION_REPORT.md

포함:

```text
최종 판정
생성 파일
baseline 전후 결과
closed-form evidence
property sweep
B2 replay smoke
negative tests
import/fallback boundary
dirty-tree integrity
remaining deferred scope
```

---

# 16. Gate D용 artifact 요구사항

Gate C에서는 runner를 수정하지 않는다.

contract에 Gate D가 보존해야 할 최소 artifact를 기록하라.

```text
timestamp
q_hat_NB
b_hat
P
ST update mask/timestamp
ST residual r
ST innovation covariance S
q_true_NB
b_true
trajectory_id
```

pairing 조건:

```text
q_hat, b_hat, P, q_true, b_true:
  same trajectory
  same physical timestamp
  same prior/posterior convention

r, S:
  same ST update
  same update timestamp
```

metric module이 interpolation이나 event alignment를 추측하지 않게 하라.

---

# 17. 완료 판정

정상 완료 시 다음 형식으로 보고하라.

```text
Status: PASS_GATE_C

Attitude geodesic/right-local error: PASS
Bias error/RMSE: PASS
Star-tracker NIS: PASS
Right-local 6D NEES: PASS
q/-q metric invariance: PASS
SPD diagnostics: PASS
Chi-square consistency summary: PASS
Timestamp/posterior pairing: PASS
Numerical fail-loud policy: PASS
No inverse/pinv/jitter/clipping: PASS
Import boundary: PASS
B2 replay metric smoke: PASS
Gate A regression: PASS
Gate B1 regression: PASS
Gate B2 regression: PASS
Legacy regression: PASS
Dirty-tree integrity: PASS

Gate C: GO
Gate D authorized: YES
```

Gate D로 자동 진행하지 마라.

---

# 18. 종료 조건

이번 실행은 Gate C에서 종료한다.

다음을 시작하지 마라.

```text
bench model adapter
registry
runner
suite YAML
cache/sidecar integration
visualization
Package C
KalmanNet
ANN
SNN
FPGA
```
