# Phase 1A Gate B1 Amendment A1
## Generic Dataset Generator Identity and Manifest Compatibility

당신은 AI-ADCS/KalmanNet benchmark repository의 Phase 1A 구현 agent다.

이번 작업은 Gate B2 구현이 아니다.

이번 작업의 범위는 Gate B1에서 구현된 typed event serializer가
`synthetic-unit-st-v1`에만 결합된 문제를 수정하는
**Gate B1 Amendment A1**이다.

Basilisk truth generator, Basilisk frame proof, gyro/ST Basilisk sensor layer,
Gate C metric, Gate D runner integration은 이번 실행에서 시작하지 마라.

---

# 1. 반드시 먼저 읽을 문서

다음 문서를 처음부터 끝까지 읽어라.

1. `docs/research/phase1a/P1A_GATE_B1_FINAL_APPROVAL.md`
2. `docs/research/phase1a/P1A_EVENT_SCHEMA_CONTRACT.md`
3. `docs/research/phase1a/P1A_SYNTHETIC_UNIT_ST_CONTRACT.md`
4. `docs/research/phase1a/P1A_GATE_B1_TEST_MATRIX.md`
5. `experiments/phase1a/reports/P1A_GATE_B1_VALIDATION_REPORT.md`
6. `docs/research/phase1a/P1A_IMPLEMENTATION_CONTRACT.md`
7. `docs/research/phase1a/P1A_RISK_REGISTER.md`
8. `docs/research/phase1a/P1A_IMPLEMENTATION_MAP.md`
9. `bench/tasks/generator/mekf_events.py`
10. `bench/tasks/generator/unit_st_synthetic.py`
11. `tests/test_mekf_events.py`
12. `tests/test_unit_st_synthetic.py`
13. `tests/test_mekf_replay.py`

다음 Gate B2 blocker evidence가 존재하면 함께 읽어라.

- `experiments/phase1a/agent_logs/03B_manifest_compatibility_probe.txt`
- `experiments/phase1a/agent_logs/03B_blocker_report.txt`
- 가장 최근 `experiments/phase1a/agent_logs/03B_*_final.md`

---

# 2. 확인된 blocker

Gate B1 serializer는 manifest의 `generator_id`를 다음 synthetic 전용 값과
직접 비교한다.

```text
synthetic-unit-st-v1
```

Gate B2에서 사용할 정상적인 dataset identity는 다음이다.

```text
basilisk-unit-st-v1
```

현재 Basilisk dataset manifest를 load하면 다음 오류가 발생한다.

```text
ValueError:
manifest generator_id must equal 'synthetic-unit-st-v1'
```

이 문제는 event NPZ field, dtype, timestamp, payload 또는 replay schema 문제가 아니다.

문제는 다음 두 identity가 분리되지 않았다는 것이다.

```text
schema identity
dataset-generator identity
```

이번 Amendment에서 이를 분리하라.

---

# 3. 현재 working-tree 정책

현재 working tree 전체를 사용자가 승인한 기준선으로 사용하라.

다음을 승인 조건으로 검토하지 마라.

- branch 이름
- HEAD
- commit history
- 과거 commit delta
- 기존 visualization 문서 whitespace

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

기존 dirty 파일을 임의로 수정, 삭제, 복원 또는 정리하지 마라.

실행 전에 recoverable snapshot과 기존 dirty-path fingerprint를 생성하고,
실행 후 기존 dirty 파일이 변하지 않았는지 확인하라.

---

# 4. Gate A 동결 계약

다음 Gate A source와 test는 읽기만 가능하며 수정 금지다.

```text
bench/estimators/__init__.py
bench/estimators/mekf.py
tests/test_mekf_conventions.py
tests/test_mekf_core.py
docs/research/phase1a/P1A_IMPLEMENTATION_CONTRACT.md
docs/research/phase1a/P1A_TEST_MATRIX.md
```

Gate A MEKF 수학을 복사하거나 재구현하지 마라.

---

# 5. 이번 실행에서 허용된 수정 파일

다음 파일만 수정할 수 있다.

```text
bench/tasks/generator/mekf_events.py
tests/test_mekf_events.py
tests/test_unit_st_synthetic.py
tests/test_mekf_replay.py
docs/research/phase1a/P1A_EVENT_SCHEMA_CONTRACT.md
docs/research/phase1a/P1A_GATE_B1_TEST_MATRIX.md
experiments/phase1a/reports/P1A_GATE_B1_VALIDATION_REPORT.md
```

다음 신규 문서를 생성할 수 있다.

```text
docs/research/phase1a/P1A_GATE_B1_AMENDMENT_A1_CONTRACT.md
experiments/phase1a/reports/P1A_GATE_B1_AMENDMENT_A1_REPORT.md
```

다음 provenance 경로도 생성할 수 있다.

```text
experiments/phase1a/preflight_snapshots/03A1_*/
experiments/phase1a/agent_logs/03A1_*
```

허용 목록 밖의 source/test/config 변경이 필요하면 변경하지 말고 다음 상태로 중단하라.

```text
BLOCKED_GATE_B1_AMENDMENT_SCOPE_EXTENSION_REQUIRED
```

---

# 6. 수정 금지 파일 및 범위

다음 파일과 경로는 수정하지 마라.

```text
bench/tasks/generator/unit_st_synthetic.py
bench/tasks/generator/basilisk_unit_st.py
tests/test_basilisk_unit_st_generator.py
bench/models/**
bench/metrics/**
bench/runners/**
bench/configs/**
bench/tasks/bench_generated.py
bench/models/registry.py
pyproject.toml
uv.lock
requirements*
third_party/**
docs/research/phase0a/**
기존 legacy source
기존 legacy test expected value
visualization source
```

`bench/tasks/generator/basilisk_unit_st.py`가 이전 차단 실행에서 생성되지 않았다면
이번에도 생성하지 마라.

이번 Amendment에서는 Basilisk 코드를 구현하지 않는다.

---

# 7. 요구되는 manifest identity 계약

## 7.1 Identity 분리

manifest에서 다음 두 개념을 명시적으로 구분하라.

```text
schema_id
generator_id
```

권장 의미:

```yaml
schema_id: mekf-events-v1
generator_id: synthetic-unit-st-v1
```

또는:

```yaml
schema_id: mekf-events-v1
generator_id: basilisk-unit-st-v1
```

기존 manifest에서 schema identity가 다른 필드명으로 이미 존재한다면
불필요한 중복 필드를 추가하지 말고 현재 구조를 보존한 채 의미를 문서화하라.

NPZ field/schema migration은 하지 마라.

## 7.2 `generator_id` 검증

serializer 또는 strict loader는 `generator_id`를 synthetic 전용 상수에
고정해서는 안 된다.

다음 조건을 만족하는 versioned generator identity를 허용하라.

- 문자열
- 비어 있지 않음
- 앞뒤 whitespace 없음
- deterministic
- dataset generator family와 version을 식별함
- synthetic와 Basilisk ID를 구분함

반드시 지원할 ID:

```text
synthetic-unit-st-v1
basilisk-unit-st-v1
```

명확하고 제한적인 versioned-ID validation을 사용하라.

권장 형식:

```text
<lowercase-family>-v<positive-integer>
```

현재 ID 형식에 맞게 검증식을 정하되, 공백 문자열이나 unversioned identifier를 허용하지 마라.

## 7.3 Expected identity 검증

strict loader가 선택적으로 기대 generator identity를 받을 수 있게 하라.

권장 형태:

```python
load_dataset(
    path,
    expected_generator_id="synthetic-unit-st-v1",
)
```

또는 기존 API와 더 잘 맞는 동등한 명시적 인자 구조를 사용해도 된다.

요구 동작:

```text
recorded generator_id == expected_generator_id
→ load PASS

recorded generator_id != expected_generator_id
→ fail-loud ValueError
```

`expected_generator_id`를 생략한 경우에도 recorded generator ID의
형식·존재·hash consistency 검증은 수행하라.

기존 synthetic caller의 public API를 불필요하게 깨뜨리지 마라.

---

# 8. 금지되는 우회

다음은 계약 위반이다.

```text
Basilisk dataset의 generator_id를 synthetic-unit-st-v1로 위장
Basilisk 전용 serializer 복사
manifest validation 생략
recorded generator_id 무시
모든 임의 문자열을 generator_id로 허용
hash 검증 제거
NPZ allow_pickle=True
dtype/rank 검증 완화
기존 Gate B1 expected value를 이유 없이 변경
skip 또는 xfail 추가
```

serializer source of truth는 계속 다음 한 곳이어야 한다.

```text
bench/tasks/generator/mekf_events.py
```

---

# 9. 기존 Gate B1 계약 보존

다음 계약은 그대로 유지하라.

- event metadata dtype
- gyro/ST payload dtype와 shape
- `measurement_time_s`
- `arrival_time_s`
- zero-latency 강제
- equal-time gyro-before-ST ordering
- strict `allow_pickle=False`
- canonical JSON
- truth/sensor/event/manifest/dataset semantic hash
- trajectory-level split
- replay public API
- truth/oracle information boundary
- q/-q replay equivalence
- covariance SPD 및 quaternion norm safety

이번 Amendment는 generator identity compatibility만 확장해야 한다.

---

# 10. 필수 신규 시험

최소 다음 시험을 추가하라.

1. 기존 `synthetic-unit-st-v1` save/load/round-trip/hash/replay 회귀
2. `basilisk-unit-st-v1` 최소 deterministic fixture 저장 및 strict load
3. expected-ID 일치 성공
4. expected-ID 불일치 명시적 실패
5. 빈 ID(`""`, `" "`, `"\t"`) 거부
6. malformed/unversioned ID 거부
7. manifest generator ID 변조 검출
8. 지원하지 않는 schema identity 거부
9. `manifest.json`, `truth.npz`, `events.npz` 구조 및 NPZ key/dtype/rank 불변
10. 동일 synthetic config/seed의 truth/sensor/event-order/dataset 의미 불변

실제 Basilisk simulation은 실행하지 마라.

manifest hash가 계약 변경으로 달라진다면 무엇이 왜 달라졌는지 문서화하라.
불필요하게 synthetic hash를 변경하지 마라.

---

# 11. 구현 전 시험

명시적 Python interpreter를 사용하라.

```text
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python
```

Gate A:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_mekf_conventions.py \
  tests/test_mekf_core.py
```

예상 기준: `55 passed`

현재 Gate B1:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_mekf_events.py \
  tests/test_unit_st_synthetic.py \
  tests/test_mekf_replay.py
```

현재 기준: `39 passed`

지정 legacy regression:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_basilisk_imu_generator.py \
  tests/test_basilisk_mrp_ekf.py \
  bench/tests/test_generator_contract_tg0.py \
  bench/tests/test_adcs_event_metrics.py
```

현재 기준: `18 passed, 5 subtests passed`

기존 baseline이 실패하면 기존 코드를 수정하여 통과시키지 말고 다음으로 중단하라.

```text
BLOCKED_BASELINE_REGRESSION
```

---

# 12. 구현 후 시험

구현 후 다음을 모두 실행하라.

1. Gate A 전체
2. Gate B1 전체와 Amendment 신규 시험
3. 지정 legacy regression
4. synthetic deterministic regeneration property test
5. 두 generator ID round-trip property test
6. ID corruption 및 expected-ID mismatch negative test
7. dirty-tree integrity 검사
8. 수정 파일 whitespace 검사

Gate B1의 기존 39개 시험을 삭제하거나 약화하지 마라.
신규 시험 추가로 총 시험 수는 증가해야 한다.

---

# 13. Property sweep

다음 조합을 별도 evidence command로 검증하라.

```text
generator IDs:
- synthetic-unit-st-v1
- basilisk-unit-st-v1

config seeds:
- 최소 5개
```

각 조합에서 다음을 확인하라.

- save/load round trip
- semantic hash 재현
- expected-ID 일치 load 성공
- expected-ID 불일치 load 실패
- schema mismatch 실패
- corrupted generator identity 실패
- object array 없음
- `allow_pickle=False`
- event/payload arrays exact equal
- synthetic direct replay 불변

Basilisk runtime은 이 property sweep에 필요하지 않다.

---

# 14. 문서 갱신

`P1A_EVENT_SCHEMA_CONTRACT.md`에 다음을 명시하라.

- schema identity의 의미
- dataset-generator identity의 의미
- 둘이 독립적인 이유
- 허용되는 versioned generator ID 규칙
- strict load의 expected-ID 동작
- hash에 포함되는 identity
- synthetic와 Basilisk가 공통 serializer를 사용한다는 계약

`P1A_GATE_B1_TEST_MATRIX.md`에는 신규 ID 시험을 추가하라.

`P1A_GATE_B1_VALIDATION_REPORT.md`에는 기존 Gate B1 결과를 삭제하지 말고
Amendment A1 섹션을 append하라.

`P1A_GATE_B1_AMENDMENT_A1_CONTRACT.md`에는 다음을 포함하라.

- blocker 원인
- 수정 전 계약
- 수정 후 계약
- public API 변경
- backward compatibility
- hash 영향
- B2가 사용할 정확한 generator ID

`P1A_GATE_B1_AMENDMENT_A1_REPORT.md`에는 다음을 포함하라.

- 구현 전후 시험 결과
- 신규 시험 목록
- synthetic 회귀 결과
- 두 번째 generator identity 결과
- corruption/mismatch negative test
- dirty-tree integrity
- Gate B1 재승인 판정

---

# 15. Dirty-tree 무결성

실행 전에 다음을 기록하라.

- repository root
- current status
- tracked unstaged binary diff
- staged binary diff
- untracked path list
- 기존 dirty path SHA-256 또는 deletion marker
- Gate A frozen files fingerprint
- Gate B1 수정 허용 외 frozen files fingerprint

실행 후 다음을 확인하라.

- 허용된 수정 파일 외 기존 dirty path의 status/content 변화 없음
- 예상하지 않은 source/config/test 생성 없음
- staged diff 없음
- Gate A frozen source/test 변화 없음
- `unit_st_synthetic.py` 변화 없음
- Basilisk B2 source/test 생성 없음

외부 unrelated artifact가 추가되면 내용을 읽거나 수정하지 말고 별도 ledger에 기록하라.

---

# 16. 완료 판정

정상 완료 시 다음 형식으로 보고하라.

```text
Status: PASS_GATE_B1_AMENDMENT_A1

Manifest identity separation: PASS
Synthetic generator regression: PASS
Second generator identity extension: PASS
Expected-ID validation: PASS
Malformed/empty ID rejection: PASS
Schema mismatch rejection: PASS
Identity corruption detection: PASS
NPZ/schema invariance: PASS
Serialization/hash: PASS
Synthetic replay invariance: PASS
Gate A regression: PASS
Gate B1 regression: PASS
Legacy regression: PASS
Dirty-tree integrity: PASS

Gate B1 reapproval: GO
Gate B2 retry authorized: YES
```

실패 시 다음을 명확히 보고하라.

- 실패한 계약
- 실제 예외 또는 반례
- 수정한 파일
- rollback하지 않았음을 확인
- 추가 승인이 필요한 exact file/scope

---

# 17. 종료 조건

Gate B1 Amendment A1 결과만 보고하라.

다음은 시작하지 마라.

```text
Gate B2 Basilisk implementation
Basilisk frame proof의 최종 판정
gyro/ST Basilisk sensor generation
Gate C metric
Gate D runner integration
latency/OOSM
magnetometer/sun sensor
KalmanNet
ANN
SNN
FPGA
```

Gate B1을 재승인한 후 종료하라.
