# Phase 1A Risk Register

- 기준일: 2026-07-31
- repository root: `/home/dss-pc-05/bench`
- 대상: 6D kinematic MEKF, gyro + `UNIT-ST`, deterministic Basilisk replay, canonical consistency metrics
- 범위: 구현 전 위험 식별. 이 문서는 source/config/dependency를 변경하지 않는다.

## 1. 등급 기준

- `Critical`: 잘못 진행하면 estimator의 수학적 의미, 데이터 공정성 또는 결과 재현성이 무효가 됨
- `High`: Phase 1A acceptance gate를 통과할 수 없거나 legacy 회귀 가능성이 큼
- `Medium`: 최초 zero-latency smoke는 가능하지만 후속 tier/운영 신뢰성을 제한함
- 상태 `BLOCK`: 표시된 gate 전에 반드시 해결
- 상태 `WATCH`: 구현 중 test/ledger로 통제
- 상태 `DEFER`: Phase 1A 최초 zero-latency 범위 밖이며 임의 결정 금지

## 2. 위험 목록

| ID | 등급 | 상태 | 위험 | repository 근거 | 영향 | 완화 및 종료 조건 |
|---|---|---|---|---|---|---|
| R-01 | Critical | BLOCK: Prompt 2 시작 전 | working tree가 대규모 dirty이고 Phase 1A 접점 파일도 이미 변경됨 | 감사 시작 시 `git status --porcelain=v1 -uall`: modified 263, deleted 681, untracked 226. 관련 변경: `bench/models/registry.py`, `bench/tasks/bench_generated.py`, `bench/tasks/generator/basilisk_imu_adcs.py` | 구현 diff의 소유권/원인을 분리하기 어렵고 기존 작업을 덮을 수 있음 | 현재 변경을 commit 또는 recoverable snapshot으로 고정하고 Prompt 2 기준 tree를 명시. 기존 변경을 reset/restore하지 않음 |
| R-02 | Critical | BLOCK: Gate B | Basilisk `sigma_BN`에서 locked active B-to-N `q_NB`로의 frame/convention 변환이 검증되지 않음 | `bench/tasks/generator/basilisk_adcs.py:398-450`은 recorder의 `sigma_BN`을 직접 state로 사용. quaternion adapter 없음 | inverse/conjugate 오류가 있어도 trajectory가 매끄러워 보여 전체 평가가 잘못될 수 있음 | identity 및 ±90° basis-vector executable tests로 MRP→quaternion 후보를 판별; `P0_05_MEKF_CONVENTION_TEST_VECTORS.md:1-400` 전부 통과 |
| R-03 | Critical | BLOCK: Gate B/D | 현재 generated-task contract가 asynchronous packet 의미를 직접 표현하지 못함 | `bench/tasks/generator/contract.py:16-24`, `:171-177` same-N/T; `bench/tasks/data_format.py:179-185`, `:198-203` extras 강제 float32 | measurement/arrival time, sensor identity, event order를 잃거나 zero-fill을 실제 측정으로 오인 | `bench/tasks/generator/mekf_events.py`에 versioned typed schema/serializer 추가; dtype/round-trip/order tests 통과. legacy v0 format은 변경하지 않음 |
| R-04 | Critical | BLOCK: Gate A | legacy MRP additive EKF를 MEKF로 오인·재사용할 위험 | `bench/models/basilisk_mrp_ekf.py:188-196`, `:639-789`은 MRP+omega, FD Jacobian, additive update | 상태·관측·공분산 의미가 모두 틀린 가짜 MEKF 생성 | 새 `bench/estimators/mekf.py`에서 독립 구현; legacy 파일 수정 금지; local Jacobian/injection/reset tests |
| R-05 | High | BLOCK: Gate B | 기존 absolute-attitude 경로가 UNIT-ST가 아님 | `bench/tasks/generator/basilisk_imu_adcs.py:1407-1430`, `:1491-1503`, `:1527-1539`은 masked/zero-filled MRP reference | ST innovation 및 quaternion sign 처리 검증 불가 | 새 normalized quaternion sensor를 truth에서 독립 noise stream으로 생성; rate/mask/covariance/sign/zero-latency tests |
| R-06 | Critical | BLOCK: Gate B/D | cache identity가 전체 configuration/code/version을 포함하지 않고 stale cache를 검출하지 않음 | `bench/tasks/bench_generated.py:122-135` scenario basis; `:873-919` existence-only cache hit | 다른 sensor rate/noise/convention 또는 generator code가 같은 cache로 재사용됨 | full resolved config + generator/schema/seed-policy/Basilisk version hash를 manifest/cache key에 포함; mismatch면 fail 또는 새 namespace; repeat/mutation tests |
| R-07 | High | BLOCK: Gate B | trajectory-level split이 contract 불변조건이 아님 | `contract.py:16-24`에 `trajectory_id` 없음. window helper `datasets/common.py:137-253`은 한 trajectory windowing 가능 | train/val/test leakage로 성능 과대평가 | 생성 시 불변 int64 trajectory ID 부여, whole-trajectory split, 세 ID set 교집합 assertion. window split 금지 |
| R-08 | High | BLOCK: Gate C/D | canonical geodesic/bias/NIS/NEES/SPD metric 부재 | `bench/metrics/core.py:14-162`; `bench/metrics/adcs_event.py:65-108`; `viz/analysis/consistency.py:7-28` | filter consistency와 수치 불안정을 공식 결과에서 판정할 수 없음 | 새 `bench/metrics/mekf.py`; right-local error 및 Cholesky solve; P/S artifact; closed-form/negative tests |
| R-09 | High | WATCH | pseudo-inverse/silent jitter가 비-SPD 문제를 숨길 위험 | `bench/models/basilisk_mrp_ekf.py:734-737` pinv fallback; `bench/metrics/core.py:141-148` inverse+jitter; `viz/analysis/consistency.py:12-28` pinv | 잘못된 covariance가 유효 NIS/NEES처럼 보고됨 | 새 path에서 Cholesky 우선, 허용된 recovery만 명시적으로 ledger에 기록, 임계 초과 시 fail-loud. legacy behavior는 변경하지 않음 |
| R-10 | High | BLOCK: 재현 가능한 Gate B/D 실행 | 기본 `python`이 실행되지 않고 lock이 설치된 Basilisk를 재현하지 못함 | pyenv shim은 version 미선택. explicit Python 3.10.13에는 `bsk==2.10.2`. `pyproject.toml:54-56`은 범위만 선언, `uv.lock:38-57`은 basilisk extra 누락, `requirements.lock:1-16` placeholder | 다른 환경/에이전트에서 import 실패 또는 다른 simulator version 사용 | 당장은 explicit interpreter와 runtime version manifest 사용. 정식 acceptance 전 별도 승인으로 interpreter/env 및 exact lock 고정 |
| R-11 | High | WATCH | same-realization은 현재 cache path에 의존하며 artifact identity가 estimator와 독립임을 검증하지 않음 | `run_suite.py:1641-1768`은 공통 data path를 사용하나 cache manifest 검증 없음 | 일부 baseline이 재생성하거나 변형된 packet을 받을 수 있음 | immutable replay hash를 run artifact에 기록; 모든 estimator run의 truth/sensor hash equality assertion; model ID를 data seed/key에서 제외 |
| R-12 | High | WATCH | 기존 `ModelAdapter.predict(y_seq, ...)`가 event queue와 diagnostics를 자연스럽게 표현하지 못함 | `bench/models/base.py:121-135`; loader extras는 batch에 넣지만 setup `system_info`는 제한적 (`run_suite.py:2024-2067`) | timestamp/order 또는 P/r/S가 adapter 경계에서 유실 | 얇은 Phase 1A adapter가 extras/sidecar를 event iterator로 변환; core 직접 replay와 adapter replay 등가 test; public API 변경 없이 append-only context 사용 |
| R-13 | High | WATCH | MEKF math가 runner/torch에 종속될 위험 | 독립 `bench/estimators` package 부재; estimator들은 `bench/models` 아래 adapter 형태 | unit test가 무거워지고 수학 구현이 runner lifecycle과 결합 | NumPy/SciPy-only `bench/estimators/mekf.py`; import-boundary test 또는 정적 import 검토; adapter에는 변환만 둠 |
| R-14 | Medium | WATCH | gyro bias의 unit/sign/noise interpretation 불일치 | `basilisk_imu_adcs.py:_configure_imu_sensor` (`:143-189`)은 sensor model을 설정하지만 기존 state는 MRP+omega | bias 추정이 반대 부호 또는 다른 단위로 평가될 수 있음 | metadata에 rad/s, measurement equation `omega_m = omega_true + b_g + n_g`, random-walk PSD를 기록하고 constant-bias test 수행 |
| R-15 | Medium | DEFER | delayed measurement/OOSM 처리 정책이 미확정 | `P0_05_MEKF_MATH_CONTRACT.md:459-488`은 async ordering을 고정하지만 delayed-update 상세는 후속 결정 대상 | latency Tier 1을 임의로 구현하면 결과 비교가 불명확 | 최초 `UNIT-ST`는 `arrival_time == measurement_time`; 두 timestamp 필드는 유지. nonzero latency는 별도 decision lock 전 구현 금지 |
| R-16 | Medium | WATCH | quaternion normalization/canonicalization이 covariance 의미를 왜곡하거나 sign jump를 만들 수 있음 | 기존 repo에는 Phase 1A convention test가 없음; viz helper는 표시 목적 (`viz/analysis/attitude.py:1-88`) | innovation spike, NEES 오류, 불연속 artifact | normalize/canonical sign 위치를 math contract대로 제한; `q`와 `-q` update equivalence, near-π tests; normalization correction ledger |
| R-17 | Medium | WATCH | sensor numeric parameter가 대표값/TBD 경계와 섞일 위험 | `P0_04_SENSOR_ROLE_AND_MODEL_SPEC.md:281-292`의 UNIT-ST는 역할을 고정하지만 일부 실제 하드웨어 수치는 후속 calibration 대상 | 임의 수치가 flight-grade claim으로 오해됨 | 모든 최초 수치를 `representative_normalized_UNIT-ST`로 metadata에 표시; flight-grade 주장 금지; config snapshot/hash 기록 |
| R-18 | Medium | WATCH | 기존 visualization NEES가 additive full-state 차이를 사용 | `viz/figures/panels.py:463-464` | 공식 metric과 dashboard가 다른 NEES를 표시 | canonical MEKF metric artifact를 runner가 생성하고 visualization은 그 값을 소비. 패널의 기존 generic fallback은 MEKF에 사용 금지 |
| R-19 | High | WATCH | runner 통합 중 legacy public API/metric regression | dispatch/registry/runner가 이미 많은 task/model을 공유 (`bench/tasks/bench_generated.py:557-675`, `bench/models/registry.py:26-50`, `run_suite.py:2361-2397`) | 기존 benchmark 결과/테스트 변화 | 새 family/model/metric key append-only, 기존 YAML 수정 금지, 선택된 회귀 + 전체 feasible test 실행, 기대값 완화 금지 |
| R-20 | Medium | WATCH | pyproject package declaration이 새 subpackage 배포를 누락할 수 있음 | `pyproject.toml:65-66`은 `packages = ["bench"]`로 명시 | editable repo에서는 import되지만 built wheel에서 `bench.estimators` 누락 가능 | Prompt 2에서는 repo-root test로 확인하고, package build/import test에서 누락되면 dependency/config 변경 권한을 별도로 요청. 이 감사 단계에서는 수정 금지 |

## 3. 현재 blocking issues

### Prompt 2 시작 전 반드시 해결

1. **Dirty baseline 소유권 고정(R-01):** 현재 관련 source가 이미 수정되어 있다. 기존 작업을 보존한 commit/snapshot과 Prompt 2 기준점을 정해야 한다.
2. **실행 interpreter 고정(R-10):** 최소한 Prompt 2 command는 `/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python`을 명시하거나 동일한 pyenv version을 활성화해야 한다.
3. **구현 경계 승인(R-03/R-04):** legacy schema/core를 바꾸지 않고 `bench/estimators/mekf.py`와 `mekf_events.py`를 새 source of truth로 두는 구조를 채택해야 한다.

### 구현 gate를 차단하지만 Prompt 2 안에서 해결 가능

1. Gate A: quaternion convention, propagation/update/injection/reset, SPD test 부재(R-04/R-09/R-16).
2. Gate B: Basilisk frame 변환 미검증, UNIT-ST 부재, typed event schema/cache/split 불변조건 부재(R-02/R-03/R-05/R-06/R-07).
3. Gate C: canonical metric 부재(R-08/R-18).
4. Gate D: adapter event/artifact 전달과 same-realization 검증 부재(R-11/R-12/R-19).

### Phase 1A release/재현 실행 전 별도 해결

- `bsk==2.10.2`와 Python 환경의 exact lock/snapshot(R-10).
- built-package에 새 `bench.estimators` subpackage가 포함되는지 검증 및 필요 시 packaging config 수정 승인(R-20).

## 4. Stop/Go gate

| Gate | Go 조건 | Stop 조건 |
|---|---|---|
| A: math | convention vectors, propagation, update, reset, SPD negative tests 통과 | sign/frame 모호성, inverse/pinv fallback, ground-truth leakage |
| B: data | q_NB basis-vector proof, deterministic UNIT-ST packet hash, typed replay round-trip, disjoint trajectory IDs | MRP ref를 ST로 대체 명명, stale cache hit, window leakage, timestamp/order 유실 |
| C: metrics | closed-form geodesic/bias/NIS/NEES 및 P/S Cholesky diagnostics 통과 | additive quaternion subtraction NEES, non-SPD를 jitter/pinv로 은폐 |
| D: integration | direct-core와 runner replay 등가, all-estimator replay hash 동일, legacy regression 통과 | model별 재생성, public legacy key/behavior 변경, 기대값 완화 |
| Release | explicit environment snapshot에 Python/Basilisk/config/seed/code hash 기록 | 기본 interpreter 불능, simulator version/lock 불명, dirty diff provenance 불명 |

## 5. 다음 단계 권고

1. 현재 working tree를 보존 가능한 방식으로 고정하고, Prompt 2의 허용 파일 목록을 `P1A_IMPLEMENTATION_MAP.md` §5로 제한한다.
2. Prompt 2의 첫 commit/diff는 `bench/estimators/mekf.py`와 convention/core tests만 포함한다. Basilisk/runner import 없이 Gate A를 끝낸다.
3. 그 다음 typed event schema와 UNIT-ST generator를 추가하고, frame proof·seed hash·trajectory split·cache identity를 Gate B에서 함께 잠근다.
4. canonical metrics를 독립 검증한 뒤에만 adapter/dispatch/registry/runner를 연결한다.
5. runner smoke에서 새 MEKF 결과뿐 아니라 기존 MRP generator/EKF/metric 및 generator-contract regression을 재실행한다.
6. nonzero latency, magnetometer/sun sensor, neural/SNN/FPGA, flight-grade sensor tuning은 이 범위에 넣지 않는다.
