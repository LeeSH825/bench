# Phase 1A Implementation Map

- 기준일: 2026-07-31
- repository root: `/home/dss-pc-05/bench`
- 목적: 6D kinematic MEKF + gyro/star-tracker `UNIT-ST`를 legacy MRP 경로와 분리하여 구현할 최소 경로를 고정한다.
- 상태: 설계 지도만 작성했으며 실제 구현은 시작하지 않았다.

## 1. 권장 구조

```text
bench/
├── estimators/
│   ├── __init__.py
│   └── mekf.py                    # runner/torch/Basilisk 비의존 NumPy core
├── tasks/generator/
│   ├── mekf_events.py             # typed truth + sensor packet/replay contract
│   └── basilisk_unit_st.py        # Basilisk truth + gyro + UNIT-ST generator
├── metrics/
│   └── mekf.py                    # geodesic/bias/NIS/NEES/SPD canonical metric
├── models/
│   └── mekf.py                    # 기존 ModelAdapter와의 얇은 bridge
└── configs/
    └── suite_phase1a_unit_st_smoke.yaml

tests/
├── test_mekf_conventions.py
├── test_mekf_core.py
├── test_basilisk_unit_st_generator.py
├── test_mekf_metrics.py
└── test_mekf_replay.py
```

`bench/estimators/mekf.py`가 source of truth다. `bench/models/mekf.py`는 event payload를 core 입력으로 변환하고 core 결과를 기존 `Prediction`/artifact 형태에 맞추는 adapter일 뿐이며, quaternion 수학이나 filter update를 중복 구현하지 않는다.

이 배치가 필요한 근거는 다음과 같다.

- 현재 독립 estimator package가 없고 model API는 runner용 sequence adapter다 (`bench/models/base.py:121-135`).
- generator dispatch는 한 함수에 모여 있다 (`bench/tasks/bench_generated.py:557-675`).
- 기존 MRP EKF는 state와 수학이 다르다 (`bench/models/basilisk_mrp_ekf.py:188-196`, `:639-789`).
- 공식 metric은 generic state MSE 중심이다 (`bench/metrics/core.py:14-162`, `bench/runners/run_suite.py:2361-2397`).

## 2. 책임별 구현 표

| 책임 | 권장 새 파일 | 재사용 파일/함수 | 수정 금지 legacy | 필요한 test | integration 시점 |
|---|---|---|---|---|---|
| quaternion/Hamilton 기초 연산, `Exp/Log`, normalize, canonical sign, `C_NB` | `bench/estimators/mekf.py` | `docs/research/phase0a/decision_lock/P0_05_MEKF_CONVENTION_TEST_VECTORS.md`; `viz/analysis/attitude.py:1-88`은 결과 비교만 | `bench/models/basilisk_mrp_ekf.py`의 MRP helpers; `bench/metrics/adcs_event.py:65-79` | `tests/test_mekf_conventions.py`: identity, axis rotations, composition, inverse, sign invariance, basis-vector mapping | Gate A-1, 가장 먼저 |
| nominal propagation `[q_NB,b_g]`, local `F/G`, Van Loan/first-order discretization | `bench/estimators/mekf.py` | `scipy.linalg.expm`; Phase 0A math contract | `basilisk_mrp_ekf.py:639-687` MRP rigid-body/FD Jacobian | `tests/test_mekf_core.py`: zero-rate, constant-rate, bias compensation, analytic-vs-finite-difference local Jacobian, exact/first-order Qd | Gate A-2 |
| ST innovation/update, right injection, reset Jacobian, covariance policy | `bench/estimators/mekf.py` | NumPy/SciPy linear algebra; Joseph form의 구조만 legacy와 비교 | `basilisk_mrp_ekf.py:729-743` additive update를 복사하지 않음; pinv/silent jitter 금지 | `tests/test_mekf_core.py`: sign flip, known residual, injection/reset, symmetry, Cholesky failure, no-ground-truth access | Gate A-3 |
| truth/sensor packet 및 replay schema | `bench/tasks/generator/mekf_events.py` | `bench/tasks/generator/contract.py:83-123`의 dataclass style; `bench/tasks/data_format.py:135-142`의 deterministic JSON idea | v1 `GeneratorOutput` public contract 변경 금지; all-float32 extras를 packet source of truth로 사용 금지 | `tests/test_mekf_replay.py`: measurement/arrival sort, tie-break, dtype, serialization round-trip, same packet hash | Gate B-1, generator 전 |
| Basilisk truth와 frame adapter | `bench/tasks/generator/basilisk_unit_st.py` | `basilisk_adcs.py:_require_avs_basilisk`, simulation process/task/recorder pattern | direct MRP state/measurement, `_small_angle_model`, `_shadow_mrp` | `tests/test_basilisk_unit_st_generator.py`: identity/basis-vector frame proof, scalar-first `q_NB`, truth continuity, fixed version manifest | Gate B-2 |
| gyro sensor | `bench/tasks/generator/basilisk_unit_st.py` | `basilisk_imu_adcs.py:_trajectory_imu_cfg` (`:115-140`), `_configure_imu_sensor` (`:143-189`) | `_select_y` (`:199-216`)의 same-grid y contract; legacy assumed H | `tests/test_basilisk_unit_st_generator.py`: unit, bias sign, clean/noisy channel, per-trajectory deterministic seed | Gate B-2 |
| `UNIT-ST` sensor | `bench/tasks/generator/basilisk_unit_st.py` | truth recorder와 stable seed policy | sparse MRP reference path (`basilisk_imu_adcs.py:1258-1705`) | `tests/test_basilisk_unit_st_generator.py`: rate/mask, normalized quaternion, sign handling, zero latency measurement/arrival timestamps | Gate B-3 |
| trajectory split, manifest, cache identity | `bench/tasks/generator/mekf_events.py`; generator metadata | `bench/tasks/bench_generated.py:921-1015`의 whole-row split idea; stable hash helper | `_scenario_basis()` (`:122-135`)과 existence-only hit (`:873-919`)을 검증 없이 사용 금지 | `tests/test_mekf_replay.py`: disjoint trajectory IDs, full config/code/schema/Basilisk version hash, seed replay | Gate B-4 |
| geodesic, bias, NIS, NEES, SPD diagnostics | `bench/metrics/mekf.py` | `viz/analysis/attitude.py:83-88`은 numerical cross-check | `bench/metrics/adcs_event.py` MRP metric; `viz/analysis/consistency.py:7-28` pinv path | `tests/test_mekf_metrics.py`: closed-form angles, sign invariance, chi-square inputs, Cholesky solve, local error NEES, deliberate non-SPD failure | Gate C-1 |
| existing runner adapter | `bench/models/mekf.py` | `bench/models/base.py:121-135`; legacy adapter lifecycle/diagnostic ledger 형식 | legacy state/propagation/update 구현 및 public behavior | `tests/test_mekf_replay.py`: direct-core vs adapter bitwise/strict-tolerance equivalence, frozen/no-training behavior | Gate D-1, core/generator/metric gate 후 |
| generator dispatch 및 registry | 새 파일 없음 | `bench/tasks/bench_generated.py:557-675`; `bench/models/registry.py:26-50` | 기존 family/model ID 변경 금지 | 기존 contract/registry tests + 새 Phase 1A smoke selection test | Gate D-2 |
| official artifact/metric integration | 새 파일 없음 | `run_suite.py:_load_split_npz`, `_SeqDataset`, metric/artifact flow (`:1279-1368`, `:2024-2067`, `:2361-2397`) | generic baseline metric key 의미 변경 금지 | end-to-end smoke: same realization, q/b/P/r/S artifact, canonical metrics, legacy regression | Gate D-3, 마지막 |
| smoke suite config | `bench/configs/suite_phase1a_unit_st_smoke.yaml` | 기존 suite schema와 runner CLI | 기존 suite YAML 수정 금지 | config load + CPU smoke; seed 2회 artifact hash 비교 | Gate D-3 |

## 3. 모듈 경계

### 3.1 `bench/estimators/mekf.py`

허용 import:

- Python standard library
- `numpy`
- `scipy.linalg` (`expm`, triangular/Cholesky solve에 필요한 함수)

금지 import:

- `bench.runners.*`
- `bench.models.*`
- `torch`
- `Basilisk`
- YAML/config loader
- visualization module

권장 public surface:

```text
MEKFConfig
MEKFState(q_NB, b_g, P, timestamp)
GyroPacket(measurement_time, arrival_time, omega_m)
StarTrackerPacket(measurement_time, arrival_time, q_NB_meas, R)
MEKF.propagate_gyro(packet)
MEKF.update_star_tracker(packet)
MEKF.snapshot()
```

Packet dataclass를 core 파일과 schema 파일 중 어디에 둘지는 dependency 방향으로 결정한다. 권장은 primitive protocol/type를 `mekf_events.py`에 두고 `mekf.py`가 그 타입만 import하는 것이다. 이때 `mekf_events.py`는 generator/Basilisk를 import하지 않는다.

### 3.2 event schema

기존 계약은 `x/y [N,T,D]`와 same N/T를 강제한다 (`bench/tasks/generator/contract.py:16-24`, `:171-177`). 따라서 Phase 1A schema는 다음 의미를 별도로 고정한다.

| 필드 | dtype / shape | 의미 |
|---|---|---|
| `trajectory_id` | `int64 [N]` | split 전 생성된 불변 trajectory identity |
| `truth_time_s` | `float64 [N,T_truth]` | truth sample time |
| `q_NB_true` | `float64 [N,T_truth,4]` | scalar-first active B-to-N truth |
| `b_g_true` | `float64 [N,T_truth,3]` | gyro bias truth, rad/s |
| `sensor_code` | `int16 [N,E]` | manifest mapping (`GYRO=1`, `ST=2`) |
| `measurement_time_s` | `float64 [N,E]` | physical measurement epoch |
| `arrival_time_s` | `float64 [N,E]` | estimator availability epoch |
| `payload` | typed channel arrays | gyro 3-vector 또는 ST quaternion; zero-fill 의미에 의존하지 않음 |
| `valid` | `bool [N,E]` | padded event slot의 유효성 |
| `event_order` | `int64 [N,E]` | equal-arrival deterministic tie-break |

최초 `UNIT-ST`는 `arrival_time_s == measurement_time_s`로 설정하되 두 필드를 모두 저장한다. event 처리 순서는 `(arrival_time_s, event_order)`로 고정한다. delayed/OOSM 정책은 Phase 0A에서 보류된 항목이므로 zero-latency 이외를 임의 구현하지 않는다.

### 3.3 truth와 estimator의 분리

generator만 truth에 접근한다. estimator core의 입력은 sensor packet, initial nominal/covariance, configuration뿐이다. metric 함수만 filter output과 truth를 동시에 받는다. 이 경계는 `docs/research/phase0a/decision_lock/P0_02_TRUTH_SENSOR_ESTIMATOR_BOUNDARY.md:15-63`, `:113-139`에 따른다.

## 4. 단계별 integration gate

### Gate A — pure MEKF math

1. convention vectors를 executable tests로 옮긴다.
2. propagation, discretization, ST update, injection/reset을 구현한다.
3. Cholesky 기반 solve, symmetry/SPD 진단, fail-loud policy를 검증한다.

완료 조건: Basilisk, runner, registry 없이 `test_mekf_conventions.py`와 `test_mekf_core.py` 통과.

### Gate B — deterministic UNIT-ST data/replay

1. typed event schema와 serialization을 만든다.
2. Basilisk MRP recorder 의미를 basis-vector test로 검증하여 `q_NB` adapter를 고정한다.
3. gyro truth/measurement 및 normalized ST quaternion packet을 같은 truth realization에서 만든다.
4. trajectory ID split, full manifest/cache identity, repeat hash를 검증한다.

완료 조건: 같은 seed 2회 생성의 truth/sensor/event hash가 같고, train/val/test trajectory ID 교집합이 공집합.

### Gate C — canonical metrics

geodesic attitude, bias RMSE, ST NIS, right-local 6D NEES, P/S SPD를 새 metric module에서 검증한다. filter 내부의 `P`, innovation `r`, `S`를 metric 입력용 artifact로 내보내되 metric이 estimator 동작을 바꾸지 않게 한다.

완료 조건: closed-form test 및 deliberate non-SPD negative test 통과.

### Gate D — bench integration

1. 얇은 adapter를 만든다.
2. 새 task family와 model ID를 append-only 등록한다.
3. runner가 event arrays와 q/b/P/r/S artifacts를 보존하고 canonical metrics를 호출하도록 국소 확장한다.
4. 새 suite YAML로 CPU smoke를 수행하고 기존 MRP/contract regression을 재실행한다.

완료 조건: direct-core replay와 runner replay 결과가 같고, 새/legacy test가 모두 통과.

## 5. Prompt 2 정확한 파일 shortlist

Prompt 2의 구현 범위는 아래 파일로 제한하는 것을 권장한다. 이 목록 밖 파일이 필요하면 구현 전에 이유와 public contract 영향을 다시 검토한다.

### 5.1 새로 생성

```text
bench/estimators/__init__.py
bench/estimators/mekf.py
bench/tasks/generator/mekf_events.py
bench/tasks/generator/basilisk_unit_st.py
bench/metrics/mekf.py
bench/models/mekf.py
bench/configs/suite_phase1a_unit_st_smoke.yaml
tests/test_mekf_conventions.py
tests/test_mekf_core.py
tests/test_basilisk_unit_st_generator.py
tests/test_mekf_metrics.py
tests/test_mekf_replay.py
```

### 5.2 기존 파일의 국소 수정

```text
bench/tasks/bench_generated.py   # 새 task-family dispatch, versioned cache/manifest 검증
bench/models/registry.py         # 새 model_id append-only 등록
bench/runners/run_suite.py       # typed event/artifact 전달 및 canonical MEKF metric hook
```

`bench/tasks/data_format.py`와 `bench/tasks/generator/contract.py`는 legacy public format 회귀 위험 때문에 Prompt 2 기본 shortlist에서 제외한다. Phase 1A 전용 typed serializer를 `mekf_events.py`에 두고, 기존 split NPZ에는 runner가 필요한 index/sidecar reference만 append-only로 연결하는 편이 안전하다. 만약 runner가 sidecar를 허용하지 않아 이 두 파일 변경이 불가피하다는 것이 executable spike로 증명되면, Prompt 2를 중단하고 schema-version migration을 별도 승인받아야 한다.

### 5.3 수정 금지

```text
bench/models/basilisk_mrp_ekf.py
bench/tasks/generator/basilisk_adcs.py
bench/tasks/generator/basilisk_imu_adcs.py
bench/metrics/adcs_event.py
docs/research/phase0a/**
third_party/**
기존 suite YAML 및 기존 test 기대값
```

기존 generator helper를 재사용해야 할 때는 import하거나 새 generic helper로 명시적으로 추출하는 방안을 검토하되, Prompt 2에서는 dirty working tree와 legacy 회귀 위험 때문에 기존 Basilisk generator 파일 자체를 수정하지 않는 것을 기본값으로 한다.

## 6. 최소 diff 원칙

1. 기존 ID/format/metric key를 바꾸지 않고 새 task family/model/metric key만 append한다.
2. MEKF 수학은 한 파일의 한 구현만 유지한다.
3. generator와 runner는 MEKF core를 import할 수 있지만 core는 그 반대를 하지 않는다.
4. truth와 sensor realization은 estimator ID보다 먼저 생성·hash되고 모든 estimator가 같은 immutable replay artifact를 읽는다.
5. 모든 convention은 prose가 아니라 executable test가 최종 판정한다.
6. legacy regression이 실패하면 기대값을 완화하지 않고 새 경로의 영향 범위를 제거한다.
7. numeric recovery를 조용히 수행하지 않는다. 정규화/대칭화/허용 jitter의 사용은 ledger에 기록하고 계약 한도를 넘으면 실패시킨다.
