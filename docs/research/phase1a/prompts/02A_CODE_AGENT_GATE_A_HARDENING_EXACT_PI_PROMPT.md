# Code CLI Agent Prompt 02A — Gate A Amendment A1
## Exact-pi antipodal determinism and true MEKFState immutability

당신은 Phase 1A Gate A의 **제한된 보강 담당자**다.
기존 Gate A 구현을 다시 설계하거나 Gate B로 확장하지 말고, Chat 독립 검토에서 발견된 두 가지 경계 문제만 최소 수정한다.

이번 실행은 현재 working tree를 사용자 승인 기준선으로 그대로 사용한다.
branch, HEAD, commit history, 과거 commit delta를 검토·비교하지 않는다.
repository-wide `git diff --check`를 실행하지 않는다.
기존 dirty 변경을 reset, restore, clean, stash, stage, commit, push하지 않는다.
visualization 코드를 읽거나 수정·실행·import하지 않는다.

---

## 1. 먼저 읽을 파일

1. `docs/research/phase0a/decision_lock/P0_05_MEKF_MATH_CONTRACT.md`
2. `docs/research/phase0a/decision_lock/P0_05_MEKF_CONVENTION_TEST_VECTORS.md`
3. `docs/research/phase0a/decision_lock/P0A_IMMEDIATE_TEST_SPEC.md`
4. `docs/research/phase1a/P1A_IMPLEMENTATION_CONTRACT.md`
5. `docs/research/phase1a/P1A_TEST_MATRIX.md`
6. `experiments/phase1a/reports/P1A_MATH_VALIDATION_REPORT.md`
7. `bench/estimators/mekf.py`
8. `tests/test_mekf_conventions.py`
9. `tests/test_mekf_core.py`
10. 사용자가 repository에 배치했다면 `docs/research/phase1a/P1A_GATE_A_CHAT_REVIEW.md`

읽은 뒤 현재 함수와 시험을 기준으로 최소 patch를 설계한다.

---

## 2. 현재 승인된 사항

다음은 변경하지 않는다.

- 6D kinematic MEKF
- nominal state `[q_NB, b_g]`
- local error `[delta_theta, delta_b_g]`
- scalar-first Hamilton quaternion
- active body-to-navigation `q_NB`
- right-multiplicative error
- gyro model과 `F/G/Phi/Q_d`
- Joseph covariance update
- exact `J_r` reset
- Cholesky fail-loud policy
- Basilisk/runner/metric/visualization 비의존 core

현재 Gate A의 42개 시험과 지정 legacy regression은 이미 통과했다.
이 실행은 그 동작을 보존해야 한다.

---

## 3. 수정이 필요한 문제 A — exact-pi antipodal tie-break

현재 구현에서 다음 exact input을 직접 재현하라.

```python
q_hat = np.array([1.0, 0.0, 0.0, 0.0])
q_z = np.array([0.0, 1.0, 0.0, 0.0])
```

현재는 다음처럼 representation-dependent residual이 나올 수 있다.

```text
residual(q_z)  = [+pi, 0, 0]
residual(-q_z) = [-pi, 0, 0]
```

SO(3) logarithm은 정확히 pi에서 축 부호가 수학적으로 비유일하다.
그러나 software output은 `q_z`와 `-q_z`에 대해 결정론적으로 같아야 한다.

### 3.1 구현 정책

- 일반적인 `|relative_scalar|`가 machine-roundoff tie 범위 밖인 경우 현재 shortest-arc 동작을 보존한다.
- exact/roundoff-level pi tie에서만 deterministic axis-sign rule을 사용한다.
- 권장 rule: vector part에서 절댓값이 tie tolerance보다 큰 첫 component가 양수가 되도록 `q` 또는 `-q`를 선택한다.
- tie tolerance는 float64 machine epsilon에서 도출한 작은 상수로 정의하고 문서화한다. 임의로 넓은 near-pi 구간을 canonicalize하지 않는다.
- 구현 위치는 `align_quaternion`, `quat_log`, 또는 내부 helper 중 최소·일관된 위치를 선택한다.
- public API를 불필요하게 늘리지 않는다.
- ordinary, small-angle, near-pi-but-not-tie behavior를 바꾸지 않는다.

### 3.2 의미 제한

문서에 다음을 명시한다.

- exact-pi tie-break는 `q/-q` representation dependence를 제거하기 위한 deterministic software convention이다.
- 정확히 180°인 attitude error의 log axis는 본질적으로 비유일하다.
- 이 변경은 local MEKF가 exact 180° initial error에서 항상 수렴한다는 주장이 아니다.
- large-initial-error convergence threshold는 여전히 후속 실험 대상이다.

---

## 4. 수정이 필요한 문제 B — MEKFState true immutability

현재 `@dataclass(frozen=True)`만으로는 내부 NumPy array가 read-only가 아니다.
Gate B event replay 전에 state aliasing과 accidental mutation을 차단한다.

### 4.1 요구 동작

`MEKFState.__post_init__`에서:

1. `q_NB`, `b_g`, `P`를 각각 독립 copy한다.
2. normalization/shape/SPD 검증 후 저장한다.
3. 저장하는 각 array를 non-writeable로 만든다.
4. caller가 원래 input array를 수정해도 state가 변하지 않는다.
5. `state.q_NB[...]`, `state.b_g[...]`, `state.P[...]` 직접 수정은 실패한다.
6. propagation/update 함수는 기존 state를 변경하지 않고 새 state를 반환한다.

다른 result dataclass 전체를 이번 범위에서 재설계하지 않는다.

---

## 5. 정확한 allowlist

이번 agent가 수정할 수 있는 파일은 다음뿐이다.

```text
bench/estimators/mekf.py
tests/test_mekf_conventions.py
tests/test_mekf_core.py
docs/research/phase1a/P1A_IMPLEMENTATION_CONTRACT.md
docs/research/phase1a/P1A_TEST_MATRIX.md
experiments/phase1a/reports/P1A_MATH_VALIDATION_REPORT.md
```

provenance log는 다음 아래에 새로 생성할 수 있다.

```text
experiments/phase1a/agent_logs/02A_*
experiments/phase1a/preflight_snapshots/02A_*
```

그 외 기존 파일은 수정하지 않는다.
허용 목록 밖 수정이 필요하면 수정하지 말고 `BLOCKED_SCOPE_EXPANSION`으로 종료한다.

특히 다음은 금지한다.

```text
bench/tasks/**
bench/models/**
bench/metrics/**
bench/runners/**
bench/configs/**
pyproject.toml
uv.lock
requirements*
docs/research/phase0a/**
docs/research/phase1a/prompts/**
third_party/**
viz/**
visualization/**
```

---

## 6. Dirty-tree 보호

- current tree를 기준선으로 사용한다.
- 실행 전 현재 status와 allowlist 파일 hash를 기록한다.
- 기존 dirty 파일을 정리하거나 복원하지 않는다.
- 실행 후 allowlist 밖 기존 파일의 status/content fingerprint가 바뀌지 않았는지 확인한다.
- 기존 repository 전체 whitespace 오류는 검사하지 않는다.
- 수정한 allowlist 파일만 `git diff --check` 또는 동등한 방식으로 검사한다.
- HEAD 변경 여부는 승인 조건으로 사용하지 않는다.

---

## 7. 필수 신규 시험

### 7.1 exact-pi antipodal tests

최소 다음을 추가한다.

1. identity estimate, exact pi about x:
   - `q_z=[0,1,0,0]`
   - `-q_z`
   - residual identical
   - correction identical
   - physical posterior identical
   - covariance identical
2. exact pi about y and z
3. exact pi about an arbitrary normalized axis with explicitly zero scalar quaternion
4. nominal quaternion도 sign-flip한 paired update
5. `quat_log(q_pi) == quat_log(-q_pi)` under deterministic tie rule
6. `quat_geodesic_angle(q_pi,-q_pi)==0`
7. near-pi but outside tie tolerance cases on both sides of pi preserve shortest-arc physical behavior

### 7.2 immutability tests

최소 다음을 추가한다.

1. caller input `q`, `b`, `P`를 state 생성 후 수정해도 state가 변하지 않음
2. `state.q_NB[...]` direct write fails
3. `state.b_g[...]` direct write fails
4. `state.P[...]` direct write fails
5. failed propagation/update input does not alter state
6. successful propagation/update returns a different state object and leaves prior unchanged

### 7.3 기존 시험 보존

기존 42개 Gate A test를 삭제, skip, xfail, 완화하지 않는다.
기존 tolerance를 실패 해결 목적으로 넓히지 않는다.

---

## 8. 실행할 시험

명시적 interpreter를 사용한다.

```text
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python
```

### 8.1 수정 전 Gate A baseline

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_mekf_conventions.py \
  tests/test_mekf_core.py
```

### 8.2 수정 후 Gate A 전체

동일 명령으로 기존+신규 시험을 모두 실행한다.

### 8.3 legacy regression

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_basilisk_imu_generator.py \
  tests/test_basilisk_mrp_ekf.py \
  bench/tests/test_generator_contract_tg0.py \
  bench/tests/test_adcs_event_metrics.py
```

### 8.4 독립 property checks

최소 다음 deterministic sweep를 별도 시험 또는 evidence script로 수행한다.

- exact-pi x/y/z/arbitrary-axis antipodal pairs
- ordinary random `q/-q` pairs
- near-pi outside tie tolerance pairs
- state read-only/defensive-copy checks

---

## 9. 문서 갱신

### `P1A_IMPLEMENTATION_CONTRACT.md`

- exact-pi deterministic axis tie-break 규칙
- tie tolerance 정의와 근거
- exact pi log non-uniqueness와 convergence claim 제한
- `MEKFState` arrays가 defensive-copy + read-only임을 명시
- Gate A Amendment A1 항목 추가

### `P1A_TEST_MATRIX.md`

- 신규 exact-pi 및 immutability test ID/입력/expected/tolerance/result 추가
- 전체 pass count와 실행시간 갱신
- 기존 결과를 삭제하지 않음

### `P1A_MATH_VALIDATION_REPORT.md`

- Amendment A1 목적과 변경 diff
- exact-pi counterexample before/fixed behavior after
- immutability before/after evidence
- 전체 신규/legacy test 결과
- Gate A final status

---

## 10. 완료 조건

다음을 모두 만족해야 한다.

1. exact-pi `q_z/-q_z` residual, correction, posterior rotation, covariance가 동일
2. ordinary/near-pi behavior regression 없음
3. `MEKFState` 내부 arrays가 read-only
4. caller array aliasing 없음
5. 모든 기존+신규 Gate A test 통과
6. 지정 legacy regression 통과
7. allowlist 밖 기존 dirty 파일 무변경
8. 수정 파일 whitespace 검사 통과
9. pseudo-inverse, jitter, clipping 추가 없음
10. Gate B 파일 생성/수정 없음

하나라도 실패하면 Gate A를 GO로 유지하지 말고 `FAIL_GATE_A_AMENDMENT_A1`로 종료한다.

---

## 11. 금지 사항

- exact-pi 시험을 제거하거나 xfail하지 않는다.
- exact pi를 단순히 unsupported로 선언하여 representation-dependence를 남기지 않는다.
- tie tolerance를 임의의 큰 각도 구간으로 확장하지 않는다.
- component-wise quaternion MSE를 추가하지 않는다.
- state immutability 대신 caller discipline만 요구하지 않는다.
- Gate B event schema, UNIT-ST generator, Basilisk adapter, metrics, runner를 구현하지 않는다.
- commit/push하지 않는다.

---

## 12. 최종 응답 형식

### Status

```text
PASS_GATE_A_AMENDMENT_A1 / FAIL_GATE_A_AMENDMENT_A1 / BLOCKED_<REASON>
```

### Changed files

- exact allowlist paths only
- agent-only diff stat

### Exact-pi evidence

```text
x-axis q/-q: PASS/FAIL
y-axis q/-q: PASS/FAIL
z-axis q/-q: PASS/FAIL
arbitrary-axis q/-q: PASS/FAIL
near-pi regression: PASS/FAIL
```

### Immutability evidence

```text
defensive copies: PASS/FAIL
q_NB read-only: PASS/FAIL
b_g read-only: PASS/FAIL
P read-only: PASS/FAIL
prior unchanged after predict/update: PASS/FAIL
```

### Tests

- pre-change Gate A count
- post-change Gate A count
- legacy regression count
- exact commands and durations

### Integrity

- allowlist-only change: PASS/FAIL
- pre-existing dirty integrity: PASS/FAIL
- new-file/modified-file whitespace: PASS/FAIL

### Gate decision

```text
B1: PASS/FAIL
B3: PASS/FAIL
B4: PASS/FAIL
B5: PASS/FAIL
B6 exact-pi included: PASS/FAIL
Numerical safety: PASS/FAIL
State immutability: PASS/FAIL
Gate A final: GO/STOP
```

Gate A final이 GO여도 Gate B를 시작하지 말고 종료한다.
