# Phase 1A Repository Read-Only Audit

- 감사일: 2026-07-31 (Asia/Seoul)
- 저장소 루트: `/home/dss-pc-05/bench`
- 기준 branch / HEAD: `benchmark-viz/stabilize-release-baseline` / `ee862a2acc368fb631c45ef0b33a8f4feb5c28c0`
- 실행 계약: `docs/research/phase1a/prompts/01_CODE_AGENT_READ_ONLY_AUDIT_PROMPT.md`
- 감사 성격: 현재 working tree의 정적 감사. source code, config, dependency는 수정하지 않았다.

## 1. 결론

Phase 1A의 독립적인 6D kinematic MEKF를 새 모듈로 구현할 기반은 있다. Basilisk 2.10.2가 설치되어 있고, 기존 generator의 결정론적 seed 및 동일-realization cache 흐름도 제한적으로 재사용할 수 있다. 그러나 현재 저장소는 Phase 1A를 그대로 수용하지 못한다.

핵심 차이는 다음과 같다.

1. 기존 ADCS 기준선은 MRP+각속도 6상태 additive EKF이고, Phase 1A의 quaternion+gyro-bias nominal state 및 6D local error-state MEKF와 수학적으로 호환되지 않는다.
2. generator 계약은 `x`와 `y`가 같은 `N×T` 격자여야 하며, sensor packet의 `measurement_time`, `arrival_time`, `sensor_id`, payload를 직접 표현하는 이벤트 계약이 없다.
3. 현재 Basilisk IMU generator는 gyro를 만들지만 absolute-attitude sensor는 MRP를 zero-fill/mask한 sparse reference뿐이다. 이것은 `UNIT-ST` quaternion star tracker가 아니다.
4. 공식 runner metric에는 quaternion geodesic error, bias RMSE, NIS, right-local NEES, covariance SPD gate가 없다.
5. data cache key가 전체 task/simulator/sensor configuration과 generator version을 포함하지 않고, 파일 존재만으로 cache를 재사용한다. 새 scenario namespace/version 없이 Phase 1A 데이터를 만들면 stale cache 위험이 있다.

따라서 Prompt 2는 runner에 먼저 결합하지 말고, 순수 NumPy MEKF core와 명시적 packet schema, UNIT-ST generator, convention/consistency tests를 먼저 통과시킨 뒤 마지막에 adapter/registry/runner를 연결해야 한다. 상세 파일 배치는 `P1A_IMPLEMENTATION_MAP.md`, 차단 조건은 `P1A_RISK_REGISTER.md`에 기록한다.

## 2. 고정 계약

이 감사와 후속 구현의 불변 조건은 다음과 같다.

- estimator: 6D kinematic MEKF
- nominal state: `[q_NB, b_g]`
- local error state: `[delta_theta, delta_b_g]`
- quaternion: scalar-first Hamilton `[w, x, y, z]`
- attitude meaning: active body-to-navigation `q_NB`
- error injection: right-multiplicative
- 최초 sensor scenario: gyro + star tracker `UNIT-ST`
- 최초 latency: zero latency, timestamp/arrival-time 필드는 유지
- 기존 `MRP + omega` additive EKF/KNet: 비교용 legacy이며 새 core와 혼합 금지

근거는 `docs/research/phase0a/decision_lock/P0_01_DECISION_LEDGER.md:18-43`의 D03–D12, D21, D23과 `docs/research/phase0a/decision_lock/P0A_PHASE_0A_SYNTHESIS.md:62-119`이다. 수학 및 안전 계약은 `docs/research/phase0a/decision_lock/P0_05_MEKF_MATH_CONTRACT.md:15-30`, `:130-312`, `:409-488`, `:499-612`를 따른다.

## 3. 감사 항목별 판정

판정은 `충족`, `부분 충족`, `미충족`, `비호환`으로 표기한다.

| # | 항목 | 판정 | 현재 상태 및 코드 근거 |
|---:|---|---|---|
| 1 | repository root / 실행 진입점 | 부분 충족 | root는 `/home/dss-pc-05/bench`. 실제 suite CLI는 `bench/runners/run_suite.py:2845-3007`의 `main()`이며 핵심 실행은 `:1609`의 `run_one()`. smoke CLI는 `bench/runners/smoke_run.py:51-141`. root `main.py:1-5`는 scaffold라 공식 진입점으로 볼 수 없다. |
| 2 | 관련 디렉터리 구조 | 부분 충족 | `bench/models`, `bench/tasks/generator`, `bench/runners`, `bench/metrics`, `bench/configs`, `tests`, `bench/tests`는 존재한다. estimator/filter 독립 디렉터리인 `bench/estimators`와 `bench/filters`는 없다. |
| 3 | Basilisk generator I/O/convention/seed/cache | 부분 충족 | `bench/tasks/generator/basilisk_adcs.py:398-450`은 Basilisk `sigma_BN`, `omega_BN_B`를 직접 6D state로 만든다. `:453-640`의 `generate_basilisk_adcs_v0()`은 `x_dim=y_dim=6` 및 direct state measurement를 전제한다. `bench/tasks/generator/basilisk_imu_adcs.py:527-599`는 IMU와 truth를 같은 clock에 기록하고, `:602-915`는 gyro 계열 y를 만든다. seed는 `basilisk_imu_adcs.py:115-140`, `:649-650`에서 안정적으로 파생한다. 그러나 ST quaternion과 `q_NB` 변환은 없다. |
| 4 | legacy MRP additive EKF | 비호환 | `bench/models/basilisk_mrp_ekf.py:188-196`의 state는 `[sigma_BN, omega_BN_B]`; `:639-687`은 rigid-body MRP dynamics와 finite-difference Jacobian; `:689-789`은 additive innovation/update이다. quaternion nominal state, gyro bias, multiplicative injection/reset이 없으므로 새 MEKF core로 재사용하면 안 된다. |
| 5 | 비동기 sensor packet schema | 미충족 | `bench/tasks/generator/contract.py:16-24`, `:117-123`, `:171-180`은 same-N/T rank-3 `float32 x/y`만 강제한다. `bench/tasks/data_format.py:144-188`은 extras를 전부 `float32`로 저장하며, `:190-204`도 float32로 읽는다. 별도 `measurement_time`, `arrival_time`, typed `sensor_id`, variable event queue 계약이 없다. |
| 6 | 모든 baseline의 same realization | 부분 충족 | `bench/runners/run_suite.py:1641-1673`에서 data path/scenario가 model-specific run directory보다 먼저 결정되고 `:1733-1768`에서 공용 generated cache를 준비하므로 한 suite/task/scenario/seed 내 모델들은 같은 split 파일을 읽는다. 하지만 `bench/tasks/bench_generated.py:122-135`의 scenario basis가 task noise와 scenario override 중심이고 전체 generator config를 포함하지 않으며, `:873-919`은 세 split 파일이 있으면 code/version hash 확인 없이 재사용한다. |
| 7 | trajectory-level split / leakage | 부분 충족 | synthetic Basilisk는 trajectory 하나를 N의 한 row로 만들고 `bench/tasks/bench_generated.py:921-1015`가 row permutation 후 train/val/test로 나누므로 현재 synthetic ADCS 경로는 trajectory-level split이다. 하지만 `contract.py:16-24`는 `trajectory_id`나 split disjointness를 요구하지 않는다. 외부 dataset 공통 함수 `bench/tasks/generator/datasets/common.py:137-253`은 한 연속 궤적에서 windows를 자를 수 있고, NCLT는 `datasets/nclt.py:179-212`, UZH-FPV는 `datasets/uzh_fpv.py:166-243`에서 cursor/start index로 분리할 뿐 물리 trajectory 독립성 assertion은 없다. |
| 8 | Basilisk 설치/version/reproducibility | 부분 충족 | 명시적 Python 3.10.13에서 `Basilisk.__version__ == 2.10.2`, distribution `bsk==2.10.2`, 위치는 site-packages이다. `pyproject.toml:54-56`은 `bsk>=2.10.2,<3`을 선언한다. 하지만 기본 `python` pyenv shim은 선택 버전이 없어 실패하고, `uv.lock:38-57`에는 basilisk extra와 `bsk`가 반영되지 않았으며 `requirements.lock:1-16`도 실제 pin snapshot이 아니다. |
| 9 | geodesic/bias/NIS/NEES/SPD metric | 미충족 | `bench/metrics/core.py:14-162`은 generic MSE/RMSE/shift/NLL만 제공한다. `bench/metrics/adcs_event.py:65-108`의 geodesic metric은 MRP-to-quaternion 변환을 전제로 하며 bias/NIS/NEES가 없다. `viz/analysis/consistency.py:7-45`의 NIS/NEES는 시각화 보조이고 pseudo-inverse fallback을 사용한다. `viz/figures/panels.py:463-464`는 additive `x_hat-x_true`를 NEES에 사용하므로 right-local MEKF의 canonical metric으로 쓸 수 없다. |
| 10 | runner 독립 MEKF core 위치 | 미충족 | 현재 `bench/estimators`/`bench/filters`가 없고 model API는 `bench/models/base.py:121-135`의 sequence-level `predict(y_seq, ...)`이다. 권장 위치는 새 `bench/estimators/mekf.py`; 이 파일은 torch, registry, YAML, runner를 import하지 않는 NumPy core여야 한다. |
| 11 | 안전/비안전 재사용 경계 | 부분 충족 | seed, cache plumbing, generic Basilisk setup pattern, NPZ metadata pattern은 제한적으로 재사용 가능하다. MRP dynamics/update, sparse MRP reference, additive ADCS metric, silent pinv/jitter safety 처리, existing cache identity는 비안전하다. 상세는 §8 재사용 표 참조. |
| 12 | 최소 변경 지점 | 부분 충족 | core-only gate는 전부 새 파일로 가능하다. runner 통합에는 generator dispatch(`bench/tasks/bench_generated.py:557-675`), adapter registry(`bench/models/registry.py:26-50`), event/covariance artifact 및 metrics 경로(`bench/runners/run_suite.py:2024-2067`, `:2361-2397`)의 국소 수정이 필요하다. 기존 MRP 파일은 수정 금지다. |

## 4. 실행 경로와 데이터 흐름

현재 주요 실행 흐름은 다음과 같다.

```text
suite YAML
  -> bench.runners.run_suite.main
  -> run_one
  -> prepare_bench_generated_v0
  -> task-family generator
  -> train.npz / val.npz / test.npz
  -> _load_split_npz / _SeqDataset
  -> ModelAdapter.predict
  -> generic metrics / artifacts
```

근거:

- CLI 및 `run_one`: `bench/runners/run_suite.py:1609`, `:2845-3007`
- generator family dispatch: `bench/tasks/bench_generated.py:557-675`
- cache 검사/생성: `bench/tasks/bench_generated.py:873-1015`
- split 로드 및 extras 전달: `bench/runners/run_suite.py:1279-1368`, `:2024-2050`
- adapter setup의 `system_info`: `bench/runners/run_suite.py:2055-2067`
- 기존 metric 계산: `bench/runners/run_suite.py:2361-2397`

Phase 1A에서는 위 흐름의 `task-family generator -> adapter` 사이에 명시적인 sensor event contract가 필요하다. 단일 dense `y[N,T,D]`에 gyro와 ST를 zero-fill하여 합치는 방식은 최초 zero-latency replay를 구현할 수는 있어도, Phase 0A가 잠근 비동기 packet 의미를 직접 보존하지 못한다. 권장 방식은 gyro clock의 dense truth와 별도로 typed event arrays를 extras에 직렬화하고, MEKF core에는 timestamp 순서의 packet iterator를 전달하는 것이다. float64 timestamp와 integer sensor code를 보존하려면 Phase 1A 전용 serializer가 필요하다.

## 5. 현재 Basilisk generator 상세

### 5.1 `basilisk_adcs_v0`

- `_require_avs_basilisk()` (`bench/tasks/generator/basilisk_adcs.py:367-395`)은 필요한 Basilisk module을 import한다.
- `_simulate_one_trajectory()` (`:398-450`)은 초기 MRP/각속도, 관성, 외란, `dt`, `T`를 받아 `sigma_BN`과 `omega_BN_B`를 기록한다.
- `generate_basilisk_adcs_v0()` (`:453-640`)은 6D MRP+omega truth에 direct measurement noise/corruption을 더한다.
- 이 출력은 Phase 1A의 `q_NB` 4성분 truth, 3D gyro bias truth, gyro/ST sensor event를 제공하지 않는다.

### 5.2 `basilisk_imu_adcs_v0` 계열

- `_trajectory_imu_cfg()` (`bench/tasks/generator/basilisk_imu_adcs.py:115-140`)은 trajectory별 sensor seed를 파생한다.
- `_configure_imu_sensor()` (`:143-189`)은 frame, bias, noise, random walk, saturation, quantization을 설정한다.
- `_simulate_one_imu_trajectory()` (`:527-599`)은 spacecraft truth와 clean/measured IMU를 같은 simulation tick에서 기록한다.
- `generate_basilisk_imu_adcs_v0()` (`:602-915`)은 MRP+omega truth와 gyro 계열 measurement를 만든다. metadata도 absolute attitude measurement가 없음을 밝힌다 (`:777-785`).
- sparse reference variant (`:1258-1705`)는 `ref_mask_seq`와 zero-filled MRP reference를 만들 뿐 (`:1407-1430`, `:1491-1503`, `:1527-1539`) quaternion star tracker가 아니다.

### 5.3 frame/convention adapter 필요

Basilisk recorder field 이름 `sigma_BN`만으로 locked active `q_NB`를 가정하면 안 된다. Prompt 2에서는 다음 basis-vector test를 먼저 통과시켜야 한다.

1. identity attitude에서 `q_NB=[1,0,0,0]`.
2. body +x/+y/+z basis가 `C_NB(q_NB)`로 navigation frame에 기대 방향으로 매핑됨.
3. Basilisk MRP -> quaternion 변환 결과와 inverse/conjugate 후보를 각각 비교하여 의미를 고정함.
4. scalar-first Hamilton product 및 right injection test vector와 일치함.

근거 계약은 `docs/research/phase0a/decision_lock/P0_05_CONVENTION_TEST_VECTORS.md:1-400`이다.

## 6. 결정론, cache, same-realization, split

### 6.1 유지할 부분

- generator contract는 같은 task/split/seed에 대해 동일 hash를 요구한다 (`bench/tasks/generator/contract.py:22-24`).
- Basilisk IMU sensor seed는 trajectory identity에서 안정적으로 파생한다 (`basilisk_imu_adcs.py:115-140`).
- `run_one()`은 모델별 실행 전에 공통 scenario/data path를 해석하므로 동일 run matrix 안의 baseline이 동일 split을 읽는다 (`run_suite.py:1641-1768`).
- synthetic Basilisk의 N축이 trajectory이므로 현재 row-level permutation은 Phase 1A split 원칙과 맞는다 (`bench/tasks/bench_generated.py:921-1015`).

### 6.2 수정 전 해결할 부분

- `_scenario_basis()` (`bench/tasks/bench_generated.py:122-135`)은 전체 `task.raw`, sensor definition, simulator version, generator code/schema version을 cache identity에 넣지 않는다.
- cache hit는 split 파일 존재만 확인한다 (`:873-919`). metadata/config hash 검증이 없다.
- contract에 `trajectory_id`가 필수가 아니고 train/val/test disjointness assertion도 없다.
- 외부 dataset window helper는 겹치는 window를 허용할 수 있다 (`datasets/common.py:238-253`). Phase 1A synthetic path에 그대로 가져오지 말아야 한다.

Prompt 2의 최소 정책은 `generator_id + schema_version + full resolved config hash + Basilisk version + seed policy version`을 manifest/cache identity에 넣고, 모든 split에 integer `trajectory_id`를 보존하며 세 집합의 교집합이 공집합인지 test하는 것이다.

## 7. 기존 estimator 및 metric의 경계

### 7.1 legacy MRP EKF

`BasiliskMRPEKFAdapter`는 비교 baseline으로 동결한다.

- state와 목적: `bench/models/basilisk_mrp_ekf.py:188-196`
- `x_dim=y_dim=6` adapter setup: `:270-342`
- legacy metadata 및 identity measurement: `:607-637`
- MRP rigid-body propagation / FD Jacobian: `:639-687`
- additive innovation, Joseph covariance, pinv fallback: `:689-789`

재사용 가능한 것은 runner adapter의 외형과 diagnostics ledger 작성 방식뿐이다. state transition, measurement model, covariance dimension/meaning, initialization, innovation, update, reset은 하나도 MEKF core로 복사하지 않는다.

### 7.2 metric

새 canonical metric은 quaternion sign ambiguity와 local tangent covariance를 명시해야 한다.

- attitude: `2*acos(abs(dot(q_hat,q_true)))`, rad/deg 둘 다 기록
- bias: per-axis 및 vector RMSE
- NIS: ST innovation `r`와 innovation covariance `S`의 Cholesky solve
- NEES: `delta_theta = Log(q_hat^{-1}⊗q_true)`와 `b_true-b_hat`로 구성한 6D local error 및 `P`의 Cholesky solve
- SPD: `P`, `S`에 대해 Cholesky success, symmetry error, minimum eigenvalue를 기록하고 실패를 숨기지 않음

`viz/analysis/attitude.py:1-88`의 quaternion helper는 참고 비교는 가능하지만, canonical implementation은 MEKF convention tests를 공유하는 새 metric module에 둔다. `bench/metrics/core.py:124-162`와 `viz/analysis/consistency.py:7-28`의 inverse/pinv fallback은 새 safety path에 복사하지 않는다.

## 8. 재사용 경계

| 자산 | 판정 | 허용 범위 | 금지/주의 근거 |
|---|---|---|---|
| `bench/utils/seeding.py`의 stable seed helper | 안전 | truth/sensor/split stream을 이름으로 분리 | 동일 realization manifest에 실제 derived seed 기록 필요 |
| `basilisk_adcs.py:_require_avs_basilisk` | 안전 | dependency probe/import error formatting | simulator state convention은 별도 adapter/test로 검증 |
| 기존 Basilisk simulation setup/recorder 패턴 | 조건부 | process/task 생성, spacecraft scheduling, recorder 연결 | `sigma_BN`을 곧바로 `q_NB`로 해석 금지 |
| `basilisk_imu_adcs.py:_configure_imu_sensor` | 조건부 | gyro noise/bias/random walk config | truth bias와 estimator bias state의 sign/unit/initialization을 새 metadata로 고정 |
| `bench/tasks/bench_generated.py` cache/split plumbing | 조건부 | 공용 model-independent data path 및 row split | current cache key/hit logic은 그대로 사용 금지 |
| NPZ/meta JSON pattern | 조건부 | replay artifact와 immutable manifest | `save_npz_split_v0()`의 all-float32 extras는 packet schema에 부적합 |
| `bench/models/basilisk_mrp_ekf.py` | 비안전 | adapter lifecycle/diagnostic 형식만 참고 | MRP+omega/additive/FD model은 MEKF core로 재사용 금지 |
| sparse MRP reference generator | 비안전 | masking test 아이디어만 참고 | UNIT-ST quaternion sensor로 이름 변경하여 재사용 금지 |
| `bench/metrics/adcs_event.py` | 비안전 | 보고서 key naming 참고 | MRP 전용이며 bias/NIS/NEES 없음 |
| inverse/pinv/jitter fallback | 비안전 | 없음 | Cholesky 실패를 숨기므로 Phase 0A numerical gate 위반 |
| `third_party/**` | 수정 금지 | 없음 | Prompt 1 및 Phase 1A 범위 밖 |

## 9. 환경 및 재현성 확인

확인 결과:

```text
default python: /home/dss-pc-05/.pyenv/shims/python -> version 미선택으로 실행 실패
explicit python: 3.10.13
Basilisk.__version__: 2.10.2
bsk distribution: 2.10.2
Basilisk path: /home/dss-pc-05/.pyenv/versions/3.10.13/lib/python3.10/site-packages/Basilisk
```

`pyproject.toml:54-56`의 범위 선언은 설치 가능성을 제공하지만 lock 재현성을 제공하지 않는다. 현재 `uv.lock:38-57`에는 `basilisk` extra/`bsk`가 없고 `requirements.lock:1-16`도 placeholder다. 이 단계에서는 dependency를 수정하지 않았으며, Prompt 2 실행 전 별도 승인된 환경 고정 작업이 필요하다.

## 10. 검증 결과

실행한 관련 회귀 subset:

```text
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q \
  tests/test_basilisk_imu_generator.py \
  tests/test_basilisk_mrp_ekf.py \
  bench/tests/test_generator_contract_tg0.py \
  bench/tests/test_adcs_event_metrics.py
```

결과: `18 passed, 5 subtests passed in 2.52s`.

이 결과는 기존 코드의 선택된 회귀 상태만 확인한다. Phase 1A quaternion convention, right-multiplicative injection/reset, `UNIT-ST`, replay equivalence, NIS/NEES/SPD test는 현재 존재하지 않아 아직 검증할 수 없다.

## 11. 조사 명령 목록

다음 종류의 read-only 명령을 repository root에서 실행했다. 긴 문서/소스 열람은 `nl -ba ... | sed -n ...`를 반복하여 EOF까지 확인했다.

```text
git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git status --short --branch
git status --porcelain=v1 -uall
rg --files -g AGENTS.md
rg --files docs/research bench tests
find bench -maxdepth 3 -type d -print
wc -l <audit 대상 문서와 핵심 source files>
nl -ba <Prompt 1 및 필수 선행 문서 10개> | sed -n <범위>
nl -ba <generator/runner/model/metric/schema/config source> | sed -n <범위>
rg -n <entrypoint/generator/cache/split/quaternion/NIS/NEES/SPD 관련 패턴> <대상 경로>
command -v python
python --version
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python --version
PYTHONDONTWRITEBYTECODE=1 ... -c <Basilisk version/import 확인>
rg -n 'bsk|basilisk' pyproject.toml uv.lock requirements.lock
PYTHONDONTWRITEBYTECODE=1 ... -m pytest -q <관련 test 4개>
```

## 12. working tree 주의사항

감사 시작 전부터 working tree는 대규모 dirty 상태였다. 최초 집계는 `modified=263`, `deleted=681`, `untracked=226`, 총 `1170` entries였으며, tracked source 변경에는 `bench/models/registry.py`, `bench/tasks/bench_generated.py`, `bench/tasks/generator/basilisk_imu_adcs.py` 등이 포함되어 있었다. 따라서 이 문서는 clean `HEAD`가 아니라 **현재 working tree**를 감사한 결과다. 기존 변경은 수정·삭제·복원하지 않았다.

Prompt 2 전에 관련 파일들의 현재 상태를 별도 commit 또는 recoverable snapshot으로 고정하고, 이 감사가 참조한 working-tree revision과 구현 기준을 일치시켜야 한다.
