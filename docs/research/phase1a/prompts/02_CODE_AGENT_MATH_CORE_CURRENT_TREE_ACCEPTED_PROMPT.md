# Code CLI Agent Prompt 2 — Current-Tree-Accepted, Dirty-Tree-Safe Phase 1A Gate A MEKF Math/Core

당신은 AI-ADCS-SNN benchmark 저장소의 **Phase 1A Gate A 구현 담당자**다.

이번 실행의 목적은 Phase 0A에서 잠근 quaternion/MEKF 수학 계약을 따르는
**runner 비의존 NumPy/SciPy reference MEKF core**와 해당 수학 검증 시험을 구현하는 것이다.

이번 실행은 전체 Phase 1A 구현이 아니다.
**Basilisk sensor generator, typed event schema, canonical metric module, model adapter,
registry, runner, YAML, Package C, neural network, SNN, FPGA는 구현하지 않는다.**

---

## 0. 사용자 승인 오버라이드 — 현재 working tree를 그대로 기준선으로 사용

이 절은 이번 실행에서 가장 높은 우선순위를 가진다.

사용자는 현재 repository의 branch, HEAD, commit history 및 기존 visualization 관련 commit을
추가 검토하지 않고, **실행 시작 시점의 현재 working tree 전체를 의도된 기준선으로 사용하도록 명시적으로 승인했다.**

따라서 다음을 수행하지 않는다.

```text
- 감사 당시 HEAD와 현재 HEAD 비교
- commit ancestry / merge-base 검사
- AUDIT_HEAD..HEAD commit delta 검사
- git log를 이용한 baseline 승인 판정
- repository 전체 git diff --check
- 기존 commit에 포함된 trailing whitespace 검사
- docs/research/phase1a/approvals/01_AUDIT_APPROVED.md 요구
- baseline reconciliation report 생성
- HEAD mismatch, branch mismatch, commit delta 때문에 중단
- visualization 동시작업 fingerprint 또는 공존 판정
```

현재 branch와 HEAD는 **provenance 기록용으로만** 조회할 수 있으며,
그 값이나 감사 문서의 이전 값과의 차이를 실행 차단 조건으로 사용하지 않는다.

다음 상태 코드를 사용하지 않는다.

```text
BLOCKED_BASELINE_MISMATCH
BLOCKED_BASELINE_BRANCH_MISMATCH
BLOCKED_BASELINE_DIVERGED_REAUDIT_REQUIRED
BLOCKED_BASELINE_DELTA_INVALID
BLOCKED_BASELINE_RECONCILIATION
BLOCKED_HEAD_CHANGED_DURING_RUN
```

현재 repository에 존재하는 기존 whitespace, 기존 dirty diff, 기존 문서 formatting 문제는
이번 Gate A의 실패 사유가 아니다. whitespace 검사는 **이번 agent가 새로 생성한 파일에만** 수행한다.

visualization tool 작업은 현재 중단된 것으로 간주한다. visualization 전용 공존 추적은 하지 않으며,
visualization 코드를 수정·실행·import하지 않는다.

---

## 1. 반드시 먼저 읽을 문서

다음 문서를 처음부터 끝까지 읽는다.

### Phase 0A locked source of truth

1. `docs/research/phase0a/decision_lock/P0A_PHASE_0A_SYNTHESIS.md`
2. `docs/research/phase0a/decision_lock/P0_01_DECISION_LEDGER.md`
3. `docs/research/phase0a/decision_lock/P0_02_TRUTH_SENSOR_ESTIMATOR_BOUNDARY.md`
4. `docs/research/phase0a/decision_lock/P0_05_MEKF_MATH_CONTRACT.md`
5. `docs/research/phase0a/decision_lock/P0_05_MEKF_CONVENTION_TEST_VECTORS.md`
6. `docs/research/phase0a/decision_lock/P0A_IMMEDIATE_TEST_SPEC.md`

### Phase 1A audit source

7. `docs/research/phase1a/P1A_REPOSITORY_AUDIT.md`
8. `docs/research/phase1a/P1A_IMPLEMENTATION_MAP.md`
9. `docs/research/phase1a/P1A_RISK_REGISTER.md`

### 현재 실행 프롬프트

10. 이 프롬프트 파일 자체

필수 문서가 없거나 읽을 수 없으면 코드 구현을 시작하지 말고
`BLOCKED_MISSING_INPUT`으로 종료한다.

감사 문서에 기록된 과거 branch/HEAD는 역사적 provenance일 뿐이며,
이번 실행의 승인 조건으로 다시 사용하지 않는다.

---

## 2. 변경 불가능한 수학 계약

다음 계약은 이번 실행에서 변경할 수 없다.

```text
filter:                 6D kinematic MEKF
nominal state:          [q_NB, b_g]
local error state:      [delta_theta, delta_b_g] in R^6
quaternion ordering:    scalar-first [w, x, y, z]
quaternion algebra:     Hamilton
attitude meaning:       active body-to-navigation q_NB
multiplicative error:   right-multiplicative
truth relation:         q_true = q_hat ⊗ Exp_q(delta_theta)
gyro model:             omega_m = omega_true + b_g + n_g
numeric reference:      float64
first update primitive: star-tracker tangent residual
covariance update:      Joseph form
covariance solve:       Cholesky/triangular solve 우선
```

세부 부호, frame, Jacobian, injection, reset 식은
`P0_05_MEKF_MATH_CONTRACT.md`와 convention test vectors를 그대로 따른다.

문서 내부에 실제 수학적 모순이 발견되면 임의 절충하지 않는다.
충돌 식, 영향 범위, 최소 재현 예를 기록하고
`BLOCKED_MATH_CONTRACT_CONFLICT`로 종료한다.

---

## 3. Dirty working tree 보호 정책

현재 working tree에는 이미 다수의 modified/deleted/untracked 파일이 존재할 수 있다.
이 기존 변경은 이번 agent의 작업 대상이 아니다.

### 3.1 절대 실행 금지 Git/환경 변경

다음을 실행하지 않는다.

```text
git reset
git clean
git restore
git checkout -- <path>
git switch
git stash
git add
git rm
git commit
git merge
git rebase
git cherry-pick
git push
```

다음도 금지한다.

```text
pip install / uninstall
uv sync
uv lock 변경
poetry/pipenv dependency 변경
apt 또는 system package 변경
pyproject.toml 수정
uv.lock 수정
requirements*.txt 또는 requirements*.lock 수정
repository-wide formatter/fixer
repository-wide black / ruff --fix / isort
```

### 3.2 실행 시작 상태 snapshot

source/test 구현 전에 현재 상태를 다음 위치에 보존한다.

```text
experiments/phase1a/preflight_snapshots/02_math_core_<UTC_TIMESTAMP>/
```

최소 artifact:

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

작성 방법의 핵심:

- `WORKTREE_TRACKED.patch`: `git diff --binary`
- `INDEX_STAGED.patch`: `git diff --cached --binary`
- `UNTRACKED_BEFORE.z`: `git ls-files --others --exclude-standard -z`
- 기존 dirty/untracked 실제 파일에는 SHA-256을 기록
- 삭제된 tracked path는 `DELETED`로 기록
- branch/HEAD는 기록만 하며 승인 또는 차단 판정에 사용하지 않음

대용량 untracked 전체 archive는 필수가 아니다.
기존 source/docs/tests/config 파일을 복구할 수 있도록 path/hash와 tracked patch를 우선 보존한다.

snapshot 디렉터리를 전혀 만들 수 없거나 tracked patch/status를 기록할 수 없을 때만
`BLOCKED_DIRTY_SNAPSHOT`으로 종료한다.
기존 untracked 대용량 파일을 tar로 묶지 못했다는 이유만으로 중단하지 않는다.

### 3.3 기존 파일 보호

실행 전에 dirty였던 파일의 상태/hash를 기록하고, 종료 전에 비교한다.

이번 agent는 §6의 allowlist 밖 기존 파일을 수정하면 안 된다.
종료 시 allowlist 밖 기존 dirty 파일이 실행 전후 달라졌다면 자동 복원하지 말고
`BLOCKED_UNINTENDED_CHANGE`로 보고한다.

단, 현재 repository의 기존 dirty 상태 자체는 실패가 아니다.

---

## 4. Target collision 검사

다음 고정 target이 실행 전에 이미 존재하는지 확인한다.

```text
bench/estimators/__init__.py
bench/estimators/mekf.py
tests/test_mekf_conventions.py
tests/test_mekf_core.py
docs/research/phase1a/P1A_IMPLEMENTATION_CONTRACT.md
docs/research/phase1a/P1A_TEST_MATRIX.md
experiments/phase1a/reports/P1A_MATH_VALIDATION_REPORT.md
```

- 하나라도 이미 존재하면 내용을 덮어쓰거나 합치지 않는다.
- path, Git 상태, size, SHA-256을 보고한다.
- `BLOCKED_TARGET_EXISTS`로 종료한다.

다음 timestamp/log 경로는 collision 검사 대상이 아니다.

```text
experiments/phase1a/agent_logs/02_math_core_*
experiments/phase1a/preflight_snapshots/02_math_core_*/**
```

---

## 5. 구현 전 baseline 회귀 시험

snapshot과 target collision 검사를 통과한 뒤 source 작성 전에 다음 interpreter를 사용한다.

```text
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python
```

다음을 기록한다.

```text
Python version
NumPy version
SciPy version
pytest version
repository root
branch
HEAD
```

branch/HEAD 값은 provenance일 뿐이며 검토·승인하지 않는다.

다음 baseline regression subset을 실행한다.

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_basilisk_imu_generator.py \
  tests/test_basilisk_mrp_ekf.py \
  bench/tests/test_generator_contract_tg0.py \
  bench/tests/test_adcs_event_metrics.py
```

출력과 exit code를 다음에 저장한다.

```text
experiments/phase1a/agent_logs/02_math_core_baseline_before.txt
```

조건:

- exit code 0이면 구현을 계속한다.
- 실패, collection error, import error가 있으면 기존 test/source를 고치지 않는다.
- 이 경우 `BLOCKED_BASELINE_REGRESSION`으로 종료한다.

이 baseline test는 현재 working tree의 실행 가능성만 검사한다.
감사 당시 commit 또는 과거 test 결과와 비교하지 않는다.

---

## 6. 이번 실행의 exact allowlist

### 6.1 새 구현 파일

```text
bench/estimators/__init__.py
bench/estimators/mekf.py
```

### 6.2 새 시험 파일

```text
tests/test_mekf_conventions.py
tests/test_mekf_core.py
```

### 6.3 새 계약/검증 문서

```text
docs/research/phase1a/P1A_IMPLEMENTATION_CONTRACT.md
docs/research/phase1a/P1A_TEST_MATRIX.md
experiments/phase1a/reports/P1A_MATH_VALIDATION_REPORT.md
```

### 6.4 실행 provenance/log/snapshot

```text
experiments/phase1a/agent_logs/02_math_core_*
experiments/phase1a/preflight_snapshots/02_math_core_*/**
```

위 목록 밖 파일이 필요하다고 판단되면 실제 수정하지 않는다.
필요 파일, 이유, public contract 영향, 가능한 대안을 보고하고
`BLOCKED_SCOPE_EXPANSION`으로 종료한다.

승인 marker 또는 baseline reconciliation 문서는 만들지 않는다.

---

## 7. 명시적 수정 금지 경로

다음 경로는 읽을 수 있지만 수정하지 않는다.

```text
bench/tasks/**
bench/models/**
bench/metrics/**
bench/runners/**
bench/configs/**
bench/tasks/generator/**
bench/models/basilisk_mrp_ekf.py
bench/tasks/generator/basilisk_adcs.py
bench/tasks/generator/basilisk_imu_adcs.py
bench/metrics/adcs_event.py
bench/tasks/data_format.py
bench/tasks/generator/contract.py
viz/**
visualization/**
bench/viz/**
bench/visualization/**
pyproject.toml
uv.lock
requirements*.txt
requirements*.lock
docs/research/phase0a/**
docs/research/phase1a/prompts/**
third_party/**
기존 test 파일 및 기존 기대값
기존 suite YAML
```

visualization tool 작업은 현재 수행하지 않는다.
visualization module을 import하거나 visualization test/server/GUI/dashboard를 실행하지 않는다.

---

## 8. 구현 범위 — Gate A pure math/core only

`bench/estimators/mekf.py`는 Python standard library, NumPy, SciPy만 사용하는
float64 reference implementation이어야 한다.

### 8.1 허용 import

```text
Python standard library
numpy
scipy.linalg
```

### 8.2 금지 import

```text
Basilisk
torch
bench.runners.*
bench.models.*
bench.tasks.*
bench.metrics.*
viz.*
visualization.*
YAML/config loader
training/neural/SNN/FPGA module
```

### 8.3 구현할 기능

함수명/class 구조는 저장소 스타일에 맞게 정할 수 있지만,
`P1A_IMPLEMENTATION_CONTRACT.md`에서 수식과 실제 함수 경로를 1:1로 대응시킨다.

#### Quaternion / SO(3)

- scalar-first Hamilton quaternion normalize
- conjugate, inverse, product
- quaternion ↔ DCM
- SO(3) quaternion Exp/Log
- hemisphere/sign alignment
- skew-symmetric matrix
- right Jacobian `J_r`와 안정적인 small-angle branch
- `q`와 `-q`의 물리적 동치 처리

#### Kinematic MEKF core

- nominal state `[q_NB, b_g]`
- local error state `[delta_theta, delta_b_g]`
- gyro propagation using `omega_m - b_g`
- continuous error matrices `F`, `G`, `Q_c`
- discrete transition `Phi`
- discrete process covariance `Q_d`
- Phase 0A 계약에 맞는 exact/Van-Loan 또는 계약상 승인된 reference discretization
- body-vector measurement prediction
- body-vector analytic Jacobian
- sun-vector tangent Jacobian primitive
- star-tracker quaternion residual의 3D tangent mapping primitive
- innovation covariance와 Kalman gain solve primitive
- Joseph covariance update
- multiplicative attitude injection
- gyro-bias correction
- local error reset
- right-reset covariance transport
- symmetry/SPD diagnostics

### 8.4 입력/출력 안전성

- public numeric input의 shape와 finite value를 검증한다.
- reference path는 `float64`를 사용한다.
- quaternion은 계약에서 정한 위치에서만 normalize한다.
- estimator core API는 truth attitude, true bias, event label, true Q/R scale을 입력으로 받지 않는다.
- event schema, sensor generator, latency/OOSM은 이번 실행에서 만들지 않는다.

---

## 9. 수치 안전 정책

다음 방식으로 오류를 숨기지 않는다.

```text
pseudo-inverse fallback 금지
eigenvalue clipping 금지
silent diagonal jitter 금지
non-SPD covariance 자동 보정 금지
NaN/Inf 무시 금지
```

허용되는 수치 처리:

```text
quaternion normalization
floating-point roundoff 수준 covariance symmetrization 0.5*(P+P.T)
small-angle analytic series
```

covariance symmetrization correction 크기를 진단 가능하게 만든다.
계약/test tolerance를 넘는 비대칭은 실패시킨다.

`P`, `Q_d`, `R`, `S`의 SPD/PSD 요구를 구분한다.
Cholesky가 필요한 행렬의 실패는 명시적 예외로 처리한다.
의도적인 non-SPD test에서는 fail-loud 동작을 검증한다.

---

## 10. 필수 시험

모든 시험은 deterministic해야 한다.
난수를 사용할 경우 seed를 고정하고 report에 기록한다.
실패를 `skip`, `xfail`, tolerance 완화로 숨기지 않는다.

### 10.1 `tests/test_mekf_conventions.py`

최소 포함:

1. Phase 0A convention vector 전부
2. identity quaternion/DCM
3. body x/y/z 축 +90° 회전과 basis-vector mapping
4. Hamilton composition order
5. inverse/conjugate consistency
6. Exp/Log small-angle round trip
7. quaternion normalization
8. `q`와 `-q`의 DCM/geodesic/residual 동치
9. right-multiplicative injection order
10. near-zero 및 계약상 필요한 near-pi 경계 동작

### 10.2 `tests/test_mekf_core.py`

#### B1 — propagation/discretization/bias sign

- zero motion
- constant angular rate
- known constant gyro bias cancellation
- gyro-bias sign test
- continuous local dynamics shape/unit check
- analytic local transition과 finite-difference/reference 비교
- `Phi`와 `Q_d` shape/symmetry
- 계약상 exact/first-order relation 확인

#### B3 — body-vector Jacobian

- analytic Jacobian vs central finite difference
- locked frame/sign convention 검증

#### B4 — sun-vector tangent Jacobian primitive

- tangent basis 포함 analytic Jacobian vs central finite difference
- rank/shape 검증

#### B5 — injection/reset

- known small attitude correction
- bias correction
- injection 후 nominal state와 residual relation
- reset Jacobian과 finite-difference/reference 비교
- covariance reset symmetry/SPD

#### B6 — sign invariance

- ST measurement `q`와 `-q`가 같은 innovation/update 생성
- nominal quaternion sign 변화에도 물리적 posterior/covariance 동일

#### Numerical safety

- Joseph update symmetry
- valid SPD covariance Cholesky success
- deliberate non-SPD `P`/`S` 명시적 실패
- pseudo-inverse/jitter/clipping fallback 부재
- nonfinite input fail-loud

#### Import boundary

- `bench.estimators.mekf` import 시 Basilisk, torch, runner, model, task, metric, visualization이 import되지 않음

### 10.3 tolerance 정책

- Phase 0A provisional tolerance가 있으면 그대로 사용한다.
- 없으면 machine precision, finite-difference step, conditioning 근거로 보수적 tolerance를 정한다.
- 실패 해결을 위해 tolerance를 반복적으로 넓히지 않는다.
- evidence 없이 tolerance 확대가 필요하면 `BLOCKED_TOLERANCE`로 종료한다.

---

## 11. 구현 후 시험 실행 순서

### 11.1 신규 Gate A test

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_mekf_conventions.py \
  tests/test_mekf_core.py
```

### 11.2 기존 regression subset 재실행

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_basilisk_imu_generator.py \
  tests/test_basilisk_mrp_ekf.py \
  bench/tests/test_generator_contract_tg0.py \
  bench/tests/test_adcs_event_metrics.py
```

각 명령의 stdout/stderr, exit code, duration을 저장한다.

```text
experiments/phase1a/agent_logs/02_math_core_new_tests.txt
experiments/phase1a/agent_logs/02_math_core_baseline_after.txt
```

선택 regression이 실패하면 기존 기대값을 변경하지 않는다.
원인과 영향 범위를 보고하고 Gate A를 STOP으로 판정한다.

---

## 12. 이번 agent가 생성한 파일만 whitespace 검사

repository 전체 또는 과거 commit delta에 대해 `git diff --check`를 실행하지 않는다.

이번에 새로 생성한 각 파일에 대해서만 다음과 동등한 검사를 수행한다.

```bash
git diff --no-index --check /dev/null <new-file>
```

`git diff --no-index`의 exit code 1은 내용 차이가 있다는 정상 상태이므로,
whitespace error 메시지 존재 여부를 별도로 판정한다.

검사 대상:

```text
bench/estimators/__init__.py
bench/estimators/mekf.py
tests/test_mekf_conventions.py
tests/test_mekf_core.py
docs/research/phase1a/P1A_IMPLEMENTATION_CONTRACT.md
docs/research/phase1a/P1A_TEST_MATRIX.md
experiments/phase1a/reports/P1A_MATH_VALIDATION_REPORT.md
```

기존 visualization 문서나 기존 dirty 파일의 trailing whitespace는 검사·수정하지 않는다.

---

## 13. 필수 문서 산출물

### 13.1 `P1A_IMPLEMENTATION_CONTRACT.md`

최소 포함:

- 목적, 입력 근거, 결정 상태, 남은 TBD, 다음 Gate
- locked convention 요약
- 수식 기능과 실제 함수/class 경로 1:1 mapping
- array shape, dtype, unit, frame
- prior/posterior notation
- normalization 위치
- Cholesky/fail-loud policy
- truth/sensor/estimator 경계
- Gate A에서 구현하지 않은 기능

### 13.2 `P1A_TEST_MATRIX.md`

최소 표:

```text
| Test ID | 수학 계약 | 시험 파일/함수 | 입력 | expected behavior | tolerance | 결과 | evidence |
```

B1, B3, B4, B5, B6, numerical safety, import-boundary test를 모두 기록한다.

### 13.3 `P1A_MATH_VALIDATION_REPORT.md`

최소 포함:

1. 최종 상태: `PASS`, `FAIL`, 또는 `BLOCKED`
2. repository root, 현재 branch, 현재 HEAD — provenance 기록 전용
3. 현재 working tree가 사용자 승인 기준선으로 사용되었음을 명시
4. dirty snapshot 경로와 보존 범위
5. Python/NumPy/SciPy/pytest version
6. 구현 전 baseline regression 결과
7. 생성 파일 목록
8. agent-only diff/stat
9. 신규 test exact command/result
10. 구현 후 legacy regression exact command/result
11. B1/B3/B4/B5/B6별 evidence
12. SPD/non-SPD safety evidence
13. numerical correction ledger
14. allowlist 밖 기존 파일 보호 검증
15. unresolved issue
16. 다음 단계 허용 범위 제안
17. Prompt 3/Gate B를 실행하지 않았음을 명시

다음 내용은 보고서에 포함하지 않는다.

```text
감사 HEAD와 현재 HEAD 비교
commit delta 승인 판정
baseline reconciliation
approval marker
visualization concurrent activity 판정
```

---

## 14. Agent-only diff와 종료 provenance

현재 repository 전체 diff는 기존 dirty 변경을 포함하므로 agent 작업량의 단독 근거로 사용하지 않는다.

이번 실행이 만든 allowlist 파일만 대상으로 다음 artifact를 만든다.

```text
experiments/phase1a/agent_logs/02_math_core_agent_only.patch
experiments/phase1a/agent_logs/02_math_core_agent_only_stat.txt
experiments/phase1a/agent_logs/02_math_core_changed_paths.txt
experiments/phase1a/agent_logs/02_math_core_status_after.txt
experiments/phase1a/agent_logs/02_math_core_dirty_integrity_check.tsv
```

새 파일은 `/dev/null`과의 `diff --no-index` 등으로 patch를 만든다.
전체 `git status --short --branch`와 global diff stat은 참고용으로만 기록하고,
agent-only diff라고 부르지 않는다.

종료 전에 다음을 판정한다.

```text
current working tree accepted without HEAD review:       PASS
recoverable dirty snapshot:                              PASS/FAIL
pre-existing allowlist-outside files unchanged:          PASS/FAIL
new changes confined to allowlist:                       PASS/FAIL
target files newly created only:                         PASS/FAIL
baseline regression before implementation:               PASS/FAIL
Gate A new tests:                                        PASS/FAIL
legacy regression after implementation:                  PASS/FAIL
new-file whitespace check only:                          PASS/FAIL
```

---

## 15. 중단 조건

다음 경우에만 중단한다.

```text
BLOCKED_MISSING_INPUT
BLOCKED_DIRTY_SNAPSHOT
BLOCKED_TARGET_EXISTS
BLOCKED_BASELINE_REGRESSION
BLOCKED_MATH_CONTRACT_CONFLICT
BLOCKED_SCOPE_EXPANSION
BLOCKED_TOLERANCE
BLOCKED_UNINTENDED_CHANGE
BLOCKED_ENVIRONMENT
```

다음은 중단 이유가 아니다.

```text
감사 HEAD와 현재 HEAD 불일치
현재 branch가 감사 당시와 다름
commit ancestry 미검토
commit delta에 기존 trailing whitespace 존재
기존 visualization 문서 whitespace
기존 dirty 파일 다수 존재
approval marker 부재
baseline reconciliation report 부재
```

중단 시:

- 기존 dirty source를 수정·복원하지 않는다.
- 가능한 snapshot/log만 남긴다.
- 허용 target report가 아직 생성되지 않았다면, 가능한 경우
  `P1A_MATH_VALIDATION_REPORT.md`에 BLOCKED 원인만 기록한다.
- 해결에 필요한 최소 사용자 행동을 한 가지로 보고한다.

---

## 16. 완료 조건

다음이 모두 충족되면 Gate A를 PASS로 판정한다.

1. 필수 Phase 0A/Phase 1A 문서를 읽음
2. 현재 working tree를 사용자 승인 기준선으로 사용
3. recoverable dirty snapshot 생성
4. target collision 없음
5. 구현 전 baseline regression exit code 0
6. exact allowlist 안에서만 파일 생성
7. B1/B3/B4/B5/B6와 convention test 전부 통과
8. deliberate non-SPD fail-loud test 통과
9. 구현 후 legacy regression exit code 0
10. allowlist 밖 기존 파일을 agent가 수정하지 않음
11. 새 파일에 한정한 whitespace 검사 통과
12. 필수 구현/test/report 파일 완성
13. stage, commit, push를 수행하지 않음
14. Gate B/Prompt 3을 시작하지 않음

---

## 17. 최종 응답 형식

### Status

```text
PASS / FAIL / BLOCKED_<REASON>
```

### Current-tree and dirty-tree handling

- repository root
- current branch / HEAD — provenance only
- `CURRENT_TREE_ACCEPTED_WITHOUT_HEAD_REVIEW: YES`
- snapshot path
- initial dirty counts
- allowlist 밖 기존 파일 integrity 결과

### Files created

- implementation
- tests
- documents
- logs/snapshot

### Test results

- pre-implementation baseline
- Gate A new tests
- post-implementation legacy regression
- exact pass/fail count와 duration

### Gate evidence

```text
Current-tree override: PASS
B1: PASS/FAIL
B3: PASS/FAIL
B4: PASS/FAIL
B5: PASS/FAIL
B6: PASS/FAIL
Numerical safety: PASS/FAIL
Dirty-tree integrity: PASS/FAIL
New-file whitespace: PASS/FAIL
Gate A: GO/STOP
```

### Diff provenance

- agent-only changed paths
- agent-only stat
- global dirty-tree stat는 참고값으로 별도 표기

### Unresolved issues

- 이번 Gate A에 남은 문제만 기록

### Next action

- Gate A가 PASS여도 Prompt 3을 실행하지 말고 종료
- 다음 단계는 Chat 검토 후 별도 프롬프트로 수행

---

## 18. 핵심 실행 원칙 요약

```text
현재 HEAD를 감사 HEAD와 비교하지 않는다.
commit history와 commit delta를 검토하지 않는다.
approval marker를 요구하거나 만들지 않는다.
기존 trailing whitespace를 검사하거나 수정하지 않는다.
현재 working tree를 사용자가 승인한 기준선으로 그대로 사용한다.
기존 dirty 상태는 snapshot으로 보호한다.
Gate A exact allowlist만 구현한다.
수학 test와 legacy regression으로 결과를 판정한다.
visualization tool은 읽기/수정/실행하지 않는다.
Gate B는 시작하지 않는다.
```
