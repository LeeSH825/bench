# Phase 1A Gate D1 Execution Contract
## MEKF Adapter and Lossless Artifact Bridge

당신은 `/home/dss-pc-05/bench` repository에서 Phase 1A Gate D1만 수행하는 구현 agent다.

이번 단계의 목적은 검증 완료된 Gate A MEKF, Gate B1 typed replay, Gate B2
Basilisk UNIT-ST, Gate C canonical metrics를 수정하지 않고, 후속 benchmark
runner 통합에 사용할 **unregistered MEKF adapter와 lossless artifact bridge**를
구현하고 검증하는 것이다.

이번 실행에서는 registry, task dispatch, `run_suite.py`, cache, suite YAML을
수정하지 마라. 해당 범위는 Gate D2다.

---

# 1. 반드시 먼저 읽을 문서와 source

```text
docs/research/phase1a/P1A_GATE_C_FINAL_APPROVAL.md
docs/research/phase1a/P1A_CANONICAL_MEKF_METRICS_CONTRACT.md
docs/research/phase1a/P1A_GATE_C_TEST_MATRIX.md
experiments/phase1a/reports/P1A_GATE_C_VALIDATION_REPORT.md

docs/research/phase1a/P1A_GATE_B2_FINAL_APPROVAL.md
docs/research/phase1a/P1A_BASILISK_UNIT_ST_CONTRACT.md
docs/research/phase1a/P1A_EVENT_SCHEMA_CONTRACT.md
docs/research/phase1a/P1A_IMPLEMENTATION_CONTRACT.md

docs/research/phase1a/P1A_RISK_REGISTER.md
docs/research/phase1a/P1A_IMPLEMENTATION_MAP.md

bench/estimators/mekf.py
bench/tasks/generator/mekf_events.py
bench/tasks/generator/basilisk_unit_st.py
bench/metrics/mekf.py

bench/models/base.py
bench/models/registry.py
bench/runners/run_suite.py
bench/tasks/bench_generated.py
```

마지막 세 shared integration 파일은 read-only 참고다. 이번 실행에서 수정하지 마라.
기존 ModelAdapter lifecycle은 참고하되 legacy MRP/KalmanNet 수학이나 dense
zero-filled sensor 표현을 복사하지 마라.

---

# 2. 승인 baseline

```text
Gate A: 55 passed
Gate B1 Amendment A1: 55 passed
Gate B2: 67 passed
Gate C: 43 passed
Legacy: 18 passed, 5 subtests passed
```

시험 개수는 provenance로 기록하고 pass/fail은 exit code와 계약 유지로 판정하라.

---

# 3. Current-tree / dirty-tree 정책

실행 시작 시점의 current working tree 전체를 사용자 승인 기준선으로 사용하라.
branch, HEAD, commit history, 과거 delta, repository 전체 whitespace는 승인 조건으로
검토하지 마라.

다음을 수행하지 마라.

```text
git reset / restore / clean / stash / add / commit / push
git merge / rebase / switch / checkout
```

실행 전에 recoverable snapshot과 기존 dirty-path fingerprint를 만들고,
실행 후 allowlist 밖 기존 path의 status/content 변화가 없는지 검사하라.
외부 unrelated non-source artifact는 읽거나 수정하지 말고 ledger에만 기록하라.

---

# 4. 동결 범위

다음 source/test/contract는 읽고 import할 수 있으나 수정 금지다.

```text
bench/estimators/**
bench/tasks/generator/mekf_events.py
bench/tasks/generator/unit_st_synthetic.py
bench/tasks/generator/basilisk_unit_st.py
bench/metrics/mekf.py

tests/test_mekf_conventions.py
tests/test_mekf_core.py
tests/test_mekf_events.py
tests/test_unit_st_synthetic.py
tests/test_mekf_replay.py
tests/test_basilisk_unit_st_generator.py
tests/test_mekf_metrics.py

docs/research/phase0a/**
docs/research/phase1a/P1A_IMPLEMENTATION_CONTRACT.md
docs/research/phase1a/P1A_EVENT_SCHEMA_CONTRACT.md
docs/research/phase1a/P1A_BASILISK_UNIT_ST_CONTRACT.md
docs/research/phase1a/P1A_CANONICAL_MEKF_METRICS_CONTRACT.md
```

Gate A math, Gate B1 replay, Gate C metric을 adapter에서 중복 구현하지 마라.

---

# 5. Exact allowlist

이번 실행에서 생성할 수 있는 source/test/doc/report는 다음뿐이다.

```text
bench/models/mekf.py
tests/test_mekf_adapter.py

docs/research/phase1a/P1A_MEKF_ADAPTER_ARTIFACT_CONTRACT.md
docs/research/phase1a/P1A_GATE_D1_TEST_MATRIX.md
experiments/phase1a/reports/P1A_GATE_D1_VALIDATION_REPORT.md
```

허용 provenance:

```text
experiments/phase1a/preflight_snapshots/05D1_*/
experiments/phase1a/agent_logs/05D1_*
```

allowlist 밖 source/test/config 변경이 필요하면 수정하지 말고
`BLOCKED_GATE_D1_SCOPE_EXTENSION_REQUIRED`로 중단하라.

---

# 6. 수정 금지 범위

```text
bench/models/registry.py
bench/models/base.py
bench/runners/**
bench/tasks/bench_generated.py
bench/tasks/data_format.py
bench/tasks/generator/contract.py
bench/configs/**
bench/metrics/core.py
bench/metrics/adcs_event.py
viz/**
visualization/**
pyproject.toml
uv.lock
requirements*
third_party/**
기존 suite YAML 및 기존 test expected value
```

registry 등록, task dispatch, runner integration, cache/sidecar persistence,
YAML, visualization, Package C, neural 작업을 시작하지 마라.

---

# 7. Base API 조사와 구현 선택

먼저 `bench/models/base.py`의 실제 API를 조사하라.

## 기존 API로 lossless typed event가 가능한 경우

base/runner 수정 없이 다음이 모두 가능하면 unregistered subclass를 구현하라.

- typed `MEKFEventTable` 또는 immutable sidecar 입력
- dense zero-filled y를 source of truth로 사용하지 않음
- truth-free estimator input
- q/b/P 및 ST r/S lossless artifact 반환
- data regeneration 없음

## 기존 API로 불가능한 경우

호환되는 것처럼 위장하지 마라.

- `bench/models/mekf.py`에 explicit unregistered event-replay bridge를 구현
- Gate D2 runner가 호출할 public method를 정의
- base class 상속은 필수가 아님
- `predict(y_seq, ...)`만으로 부족한 이유와 Gate D2 최소 extension을 문서화

어느 경우든 direct replay reuse, lossless artifact, truth-free input,
Gate D2 exact extension이 충족되면 Gate D1 GO가 가능하다.

절대로 typed events를 float32 dense y로 변환하거나 ST 부재 row를 zero-fill하지 마라.

---

# 8. Adapter/bridge 입력 계약

estimator-facing public call은 최소 다음 의미를 받는다.

```text
event_table
trajectory_id
initial_state
initial_time_s
Q_c
dataset_identity
```

`dataset_identity` 최소 필드:

```text
schema_version
generator_id
convention_id
truth_hash
sensor_payload_hash
event_order_hash
manifest_hash
dataset_hash
```

adapter가 identity를 재계산하거나 변경하지 마라.

estimator-facing API에는 다음 입력이 없어야 한다.

```text
q_true_NB
b_true
omega_true
oracle Q/R
event label
future sample
metric result
model-dependent data seed
```

initial state는 caller가 명시적으로 제공한다. truth에서 만들지 마라.

---

# 9. Lossless artifact bundle

immutable/read-only artifact bundle을 반환하라.

## Filter event artifact

```text
trajectory_id
event_index 또는 event_order
timestamp_s
sensor_code
q_hat_NB
b_hat_rad_s
P
```

## Compact ST update artifact

```text
st_event_index
st_timestamp_s
st_residual_rad
st_innovation_covariance_rad2
```

## Provenance

```text
schema_version
generator_id
convention_id
truth_hash
sensor_payload_hash
event_order_hash
manifest_hash
dataset_hash
adapter_id
adapter_version
```

권장 추가:

```text
final_state
number_of_events
number_of_gyro_events
number_of_st_updates
```

원칙:

- q/b/P는 같은 posterior sample과 timestamp
- residual/S는 같은 실제 ST update
- gyro row에 residual/S를 0 또는 NaN으로 채우지 않음
- ST evidence는 compact table/index로 보존
- exact dtype 유지, defensive copy, read-only
- truth를 artifact에 섞지 않음
- Gate D2에서 evaluation truth를 별도 exact join

---

# 10. Direct replay 재사용

adapter는 반드시 Gate B1 검증 replay API를 호출해야 한다.
event loop, ordering, propagation, ST update를 다시 작성하지 마라.

정적/monkeypatch test로 다음을 확인하라.

```text
Gate B1 replay 호출
별도 propagation/update loop 없음
Gate A math 중복 없음
Gate C metric을 estimator 내부에서 호출하지 않음
```

Gate C metric은 evaluation smoke에서 artifact와 별도 truth를 pairing할 때만 사용한다.

---

# 11. 필수 시험

최소 다음을 구현하라.

1. `D1-01` import boundary: Basilisk, torch, runner, registry, viz 미로딩
2. `D1-02` direct replay와 adapter q/b/P/r/S/final state exact equivalence
3. `D1-03` serialization round-trip adapter equivalence
4. `D1-04` synthetic 및 `basilisk-unit-st-v1` 양쪽 검증
5. `D1-05` dataset identity exact 보존 및 mismatch fail-loud
6. `D1-06` adapter/model identity가 data hash와 numeric output에 영향 없음
7. `D1-07` public API truth/oracle/label/future/metric 입력 부재
8. `D1-08` data regeneration 및 input mutation 없음
9. `D1-09` artifact dtype/shape/read-only/count relation
10. `D1-10` ST residual/S compact count가 실제 ST update count와 동일
11. `D1-11` artifact + 별도 truth exact join 후 Gate C metrics가 direct evidence와 동일
12. `D1-12` raw ST q/-q stream에서 numeric artifact와 physical metrics 동일
13. `D1-13` invalid identity, trajectory, initial time/order, internal count/index mismatch 거부
14. `D1-14` deterministic no-training/frozen behavior
15. `D1-15` math/replay duplication, inverse/pinv/jitter, dense zero-fill, truth access 부재

---

# 12. Baseline 시험

모든 Python 명령:

```text
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python
PYTHONDONTWRITEBYTECODE=1
pytest -p no:cacheprovider
```

구현 전 다음을 실행하라.

```bash
# Gate A
python -m pytest -q -p no:cacheprovider tests/test_mekf_conventions.py tests/test_mekf_core.py

# Gate B1
python -m pytest -q -p no:cacheprovider tests/test_mekf_events.py tests/test_unit_st_synthetic.py tests/test_mekf_replay.py

# Gate B2
python -m pytest -q -p no:cacheprovider tests/test_basilisk_unit_st_generator.py

# Gate C
python -m pytest -q -p no:cacheprovider tests/test_mekf_metrics.py

# Legacy
python -m pytest -q -p no:cacheprovider tests/test_basilisk_imu_generator.py tests/test_basilisk_mrp_ekf.py bench/tests/test_generator_contract_tg0.py bench/tests/test_adcs_event_metrics.py
```

위 `python`은 반드시 명시적 interpreter 경로로 치환하라.
baseline failure 시 기존 코드를 수정하지 말고 `BLOCKED_BASELINE_REGRESSION`으로 중단하라.

---

# 13. 구현 후 시험과 property sweep

다음을 모두 실행하라.

- `tests/test_mekf_adapter.py`
- Gate A/B1/B2/C regression
- legacy regression
- synthetic 최소 5 seeds
- Basilisk 최소 3 seeds
- direct/adapter exact equivalence
- serialization equivalence
- identity preservation
- q/-q invariance
- artifact read-only
- compact ST evidence count
- Gate C metric equality
- P/S Cholesky safety
- import/source boundary
- whitespace 및 dirty-tree integrity

skip, xfail, tolerance 완화, expected legacy value 변경으로 실패를 숨기지 마라.

---

# 14. 문서 산출물

## `P1A_MEKF_ADAPTER_ARTIFACT_CONTRACT.md`

포함:

```text
목적과 입력 근거
base API 조사 결과
subclass 또는 explicit bridge 선택 이유
public API
truth-free input
identity contract
artifact field/dtype/shape/unit
compact ST evidence
direct replay dependency
metric pairing boundary
immutability
Gate D2 최소 extension
deferred scope
```

## `P1A_GATE_D1_TEST_MATRIX.md`

각 test의 ID, contract, input, expected, tolerance, actual, evidence, status를 기록하라.

## `P1A_GATE_D1_VALIDATION_REPORT.md`

최종 판정, 생성 파일, base API 조사, equivalence, synthetic/Basilisk 결과,
artifact/identity/truth boundary, metric smoke, negative tests, regression,
dirty-tree integrity, Gate D2 exact required changes를 기록하라.

---

# 15. Gate D2 요구사항 기록

Gate D1 문서에 다음을 exact하게 기록하라.

- runner가 typed event sidecar를 adapter에 전달할 위치
- truth를 estimator 후 evaluation에서 join할 위치
- q/b/P/r/S artifact 저장 위치와 형식
- append할 model ID와 task-family ID
- cache/manifest에서 확인할 semantic hashes
- direct/adapter/runner equivalence test 방법
- 수정이 필요한 기존 파일 exact shortlist

이번 실행에서는 그 파일들을 수정하지 마라.

---

# 16. 완료 판정

정상 완료 시 다음 형식으로 보고하라.

```text
Status: PASS_GATE_D1

Base API compatibility decision: PASS
Direct replay reuse: PASS
Direct/adapter exact equivalence: PASS
Synthetic generator adapter: PASS
Basilisk generator adapter: PASS
Serialization round-trip equivalence: PASS
Dataset identity preservation: PASS
Same-realization independence: PASS
Truth-free estimator boundary: PASS
Lossless q/b/P artifact: PASS
Compact ST r/S artifact: PASS
q/-q invariance: PASS
Gate C metric pairing smoke: PASS
Artifact immutability: PASS
No math/replay duplication: PASS
Import/source boundary: PASS
Gate A regression: PASS
Gate B1 regression: PASS
Gate B2 regression: PASS
Gate C regression: PASS
Legacy regression: PASS
Dirty-tree integrity: PASS

Gate D1: GO
Gate D2 authorized: YES
```

Gate D2로 자동 진행하지 마라.

---

# 17. 종료 조건

이번 실행은 Gate D1에서 종료한다.

다음을 시작하지 마라.

```text
registry
task dispatch
run_suite.py
suite YAML
cache/sidecar integration
visualization
Package C
KalmanNet
ANN
SNN
FPGA
```
