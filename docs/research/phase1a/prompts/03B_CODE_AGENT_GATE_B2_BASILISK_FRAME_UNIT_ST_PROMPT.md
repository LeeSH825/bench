# Phase 1A Gate B2 실행 계약

## Basilisk Frame/Convention Proof + Deterministic Gyro/Star-Tracker UNIT-ST Generator

당신은 `/home/dss-pc-05/bench` repository에서 **Phase 1A Gate B2만** 수행하는 구현 agent다.

이번 실행의 목표는 검증 완료된 Gate A MEKF core와 Gate B1 typed event/replay contract를 수정하지 않고 다음을 구현·검증하는 것이다.

1. Basilisk recorder의 `sigma_BN`을 locked active scalar-first `q_NB`로 변환하는 adapter
2. Basilisk `omega_BN_B`의 frame, sign, unit을 Gate A propagation과 연결하는 executable proof
3. minimal Tier-0 Basilisk rigid-body truth generator
4. truth에서 생성되는 parameterized gyro와 quaternion star-tracker sensor-output layer
5. Gate B1 typed schema, serialization, hashes, split, direct replay를 그대로 사용하는 deterministic Basilisk UNIT-ST dataset

이번 Gate에서 canonical metric, benchmark runner, model adapter, registry, visualization, neural network, orbit/environment, nonzero latency를 구현하지 않는다.

---

# 0. 최우선 실행 정책

## 0.1 현재 working tree 수용

실행 시작 시점의 현재 working tree 전체를 사용자 승인 기준선으로 사용한다.

다음을 승인 조건으로 비교하거나 검사하지 마라.

- branch 이름
- 현재 HEAD
- 과거 HEAD
- commit history
- merge-base 또는 ancestry
- 과거 commit delta
- repository 전체 `git diff --check`
- approval marker의 commit SHA

branch와 HEAD는 provenance로 기록할 수 있지만, 그 값 자체를 이유로 중단하지 마라.

## 0.2 기존 dirty 변경 보호

현재 working tree는 대규모 dirty 상태다. 다음을 절대로 수행하지 마라.

```text
git reset
git restore
git clean
git checkout --
git switch
git stash
git add
git commit
git push
git merge
git rebase
```

구현 전에 tracked/staged/untracked 상태를 recoverable snapshot과 hash ledger로 보존하라.

기존 dirty 경로는 이번 exact allowlist를 제외하고 실행 전후 status/content fingerprint가 동일해야 한다.

다른 프로세스가 source file을 변경하면 즉시 다음으로 중단하라.

```text
BLOCKED_CONCURRENT_SOURCE_CHANGE
```

`artifacts/benchmark_write_control/**`처럼 source가 아닌 외부 artifact의 신규 파일은 읽거나 수정하지 말고 path/status ledger에만 기록할 수 있다. 그것만으로 Gate B2를 실패시키지 마라.

## 0.3 Gate A와 Gate B1 동결

다음 Gate A 파일은 읽고 import할 수 있지만 수정하지 마라.

```text
bench/estimators/__init__.py
bench/estimators/mekf.py
tests/test_mekf_conventions.py
tests/test_mekf_core.py
docs/research/phase1a/P1A_IMPLEMENTATION_CONTRACT.md
docs/research/phase1a/P1A_TEST_MATRIX.md
experiments/phase1a/reports/P1A_MATH_VALIDATION_REPORT.md
```

다음 Gate B1 파일도 읽고 import할 수 있지만 수정하지 마라.

```text
bench/tasks/generator/mekf_events.py
bench/tasks/generator/unit_st_synthetic.py

tests/test_mekf_events.py
tests/test_unit_st_synthetic.py
tests/test_mekf_replay.py

docs/research/phase1a/P1A_EVENT_SCHEMA_CONTRACT.md
docs/research/phase1a/P1A_SYNTHETIC_UNIT_ST_CONTRACT.md
docs/research/phase1a/P1A_GATE_B1_TEST_MATRIX.md
experiments/phase1a/reports/P1A_GATE_B1_VALIDATION_REPORT.md
```

Gate A API 변경이 필요하면 수정하지 말고 다음으로 중단하라.

```text
BLOCKED_GATE_A_INTERFACE_CHANGE_REQUIRED
```

Gate B1 schema/serializer/hash/replay API 변경이 필요하면 수정하지 말고 다음으로 중단하라.

```text
BLOCKED_GATE_B1_INTERFACE_CHANGE_REQUIRED
```

각 경우 필요한 최소 변경, 이유, 영향받는 public contract를 보고하라.

## 0.4 명시적 Python과 simulator runtime

모든 Python/test 명령은 다음 interpreter를 명시적으로 사용하라.

```text
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python
```

pytest에는 기본적으로 다음을 적용하라.

```text
PYTHONDONTWRITEBYTECODE=1
-p no:cacheprovider
```

실행 시작 시 다음을 기록하라.

```text
Python version
NumPy version
SciPy version
Basilisk.__version__
설치 distribution의 bsk version
Basilisk package path
```

감사 당시 Basilisk는 `2.10.2`였지만 숫자를 억지로 맞추지 마라. 현재 runtime을 기록하고, import가 실패하면 dependency를 수정하지 말고 다음으로 중단하라.

```text
BLOCKED_BASILISK_RUNTIME
```

`pyproject.toml`, `uv.lock`, `requirements*`, environment를 이번 Gate에서 수정하지 마라.

---

# 1. 반드시 읽을 문서와 코드

다음 문서를 처음부터 끝까지 읽어라.

```text
docs/research/phase0a/decision_lock/P0A_PHASE_0A_SYNTHESIS.md
docs/research/phase0a/decision_lock/P0_01_DECISION_LEDGER.md
docs/research/phase0a/decision_lock/P0_02_TRUTH_SENSOR_ESTIMATOR_BOUNDARY.md
docs/research/phase0a/decision_lock/P0_03_TRUTH_MODEL_SPEC.md
docs/research/phase0a/decision_lock/P0_04_SENSOR_ROLE_AND_MODEL_SPEC.md
docs/research/phase0a/decision_lock/P0_05_MEKF_MATH_CONTRACT.md
docs/research/phase0a/decision_lock/P0_05_MEKF_CONVENTION_TEST_VECTORS.md
docs/research/phase0a/decision_lock/P0A_IMMEDIATE_TEST_SPEC.md

docs/research/phase1a/P1A_REPOSITORY_AUDIT.md
docs/research/phase1a/P1A_IMPLEMENTATION_MAP.md
docs/research/phase1a/P1A_RISK_REGISTER.md
docs/research/phase1a/P1A_IMPLEMENTATION_CONTRACT.md
docs/research/phase1a/P1A_EVENT_SCHEMA_CONTRACT.md
docs/research/phase1a/P1A_SYNTHETIC_UNIT_ST_CONTRACT.md
docs/research/phase1a/P1A_GATE_B1_TEST_MATRIX.md
docs/research/phase1a/P1A_GATE_A_FINAL_APPROVAL.md
docs/research/phase1a/P1A_GATE_B1_FINAL_APPROVAL.md
```

다음 source는 실제 public API와 dependency 방향을 확인하기 위해 read-only로 전부 읽어라.

```text
bench/estimators/mekf.py
bench/tasks/generator/mekf_events.py
bench/tasks/generator/unit_st_synthetic.py
bench/utils/seeding.py
```

다음 기존 Basilisk 코드는 process/task/spacecraft/recorder 패턴과 설치 API를 확인하기 위한 read-only 참고만 허용한다.

```text
bench/tasks/generator/basilisk_adcs.py
bench/tasks/generator/basilisk_imu_adcs.py
```

기존 MRP state, sparse MRP reference, all-float32 output contract, legacy sensor selection을 새 generator에 복사하지 마라.

설치된 Basilisk package의 다음 계열 module/source/docstring을 read-only로 조사할 수 있다.

```text
Basilisk.utilities.SimulationBaseClass
Basilisk.utilities.macros
Basilisk.utilities.RigidBodyKinematics
Basilisk.simulation.spacecraft
실제 사용되는 state message/recorder type
```

인터넷 검색에 의존하지 말고, 실행 중인 설치 version의 Python API와 source/docstring을 우선 근거로 사용하라.

---

# 2. 이번 Gate의 정확한 범위

Gate B2에서 구현할 것은 다음뿐이다.

1. Basilisk `sigma_BN` → locked active scalar-first `q_NB` conversion
2. identity, 각 축 ±90°, arbitrary attitude, MRP shadow-set executable frame proof
3. Basilisk `omega_BN_B` → Gate A body-frame angular-rate semantics proof
4. zero-torque spherical-inertia constant-rate dynamics proof
5. minimal Basilisk rigid-body truth trajectory generator
6. constant true gyro bias와 white-noise gyro measurement wrapper
7. low-rate right-local tangent-noise quaternion star-tracker wrapper
8. Gate B1 typed event dataset 생성
9. Gate B1 serialization/hash/split/direct replay 재사용
10. simulator identity와 convention proof identity 기록
11. deterministic regeneration과 seed-isolation 검증

Gate B2에서 구현하지 말아야 할 것은 다음이다.

```text
nonzero latency 또는 OOSM buffer
sensor outage, false ST solution, outlier, saturation
bias random walk 또는 temperature drift
magnetometer 또는 sun sensor
orbit propagation, eclipse, WMM, gravity-gradient
reaction wheel, magnetorquer, controller, closed loop
canonical geodesic/bias/NIS/NEES metric module
benchmark ModelAdapter
model registry
generator dispatch
run_suite integration
legacy cache migration
suite YAML
visualization/dashboard
KalmanNet, Split-KalmanNet, ANN, SNN, FPGA
Package C experiment
```

Basilisk에 built-in star-tracker module을 사용하지 않는다면 명시적으로 **project-owned parameterized sensor-output wrapper**라고 기록하라. built-in Basilisk sensor를 사용한 것처럼 표현하지 마라.

---

# 3. Exact allowlist

## 3.1 새로 생성 가능한 파일

다음 구현·시험·계약·보고서 파일만 새로 생성할 수 있다.

```text
bench/tasks/generator/basilisk_unit_st.py

tests/test_basilisk_unit_st_generator.py

docs/research/phase1a/P1A_BASILISK_FRAME_CONVENTION_PROOF.md
docs/research/phase1a/P1A_BASILISK_UNIT_ST_CONTRACT.md
docs/research/phase1a/P1A_GATE_B2_TEST_MATRIX.md

experiments/phase1a/reports/P1A_GATE_B2_VALIDATION_REPORT.md
```

실행 evidence는 다음 prefix 아래에 생성할 수 있다.

```text
experiments/phase1a/agent_logs/03B_*
experiments/phase1a/preflight_snapshots/03B_*/**
```

## 3.2 target collision

위 target이 실행 전에 이미 존재하면 덮어쓰지 말고 다음으로 중단하라.

```text
BLOCKED_TARGET_EXISTS
```

입력 문서인 이 prompt와 `P1A_GATE_B1_FINAL_APPROVAL.md`는 collision 대상이 아니다.

## 3.3 수정 금지 경로

다음을 수정하지 마라.

```text
bench/estimators/**
bench/tasks/generator/mekf_events.py
bench/tasks/generator/unit_st_synthetic.py
bench/tasks/generator/basilisk_adcs.py
bench/tasks/generator/basilisk_imu_adcs.py
bench/tasks/generator/contract.py
bench/tasks/data_format.py
bench/tasks/bench_generated.py
bench/models/**
bench/metrics/**
bench/runners/**
bench/configs/**
pyproject.toml
uv.lock
requirements*
docs/research/phase0a/**
third_party/**
viz/**
visualization/**
기존 tests와 기존 기대값
```

repository-wide formatter, import sorter, whitespace fixer를 실행하지 마라.

---

# 4. Preflight와 baseline

## 4.1 snapshot

다음 경로를 만들어라.

```text
experiments/phase1a/preflight_snapshots/03B_<UTC_TIMESTAMP>/
```

최소 다음을 저장하라.

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
FROZEN_GATE_A_HASHES.tsv
FROZEN_GATE_B1_HASHES.tsv
ALLOWLIST_EXISTENCE_BEFORE.tsv
RUNTIME_VERSIONS.txt
SNAPSHOT_MANIFEST.md
```

branch/HEAD는 provenance로만 기록하고 승인 판단에 사용하지 마라.

대용량 untracked archive 실패는 차단 사유로 삼지 말고 path/hash/status를 우선 보존하라.

## 4.2 구현 전 Gate A test

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_mekf_conventions.py \
  tests/test_mekf_core.py
```

기대: exit code 0. 현재 기록은 Amendment A1 후 `55 passed`다.

## 4.3 구현 전 Gate B1 test

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_mekf_events.py \
  tests/test_unit_st_synthetic.py \
  tests/test_mekf_replay.py
```

기대: exit code 0. 현재 기록은 `39 passed`다.

## 4.4 구현 전 legacy regression

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_basilisk_imu_generator.py \
  tests/test_basilisk_mrp_ekf.py \
  bench/tests/test_generator_contract_tg0.py \
  bench/tests/test_adcs_event_metrics.py
```

기대: exit code 0. 현재 기록은 `18 passed, 5 subtests passed`다.

어느 baseline이든 실패하면 기존 코드나 기대값을 수정하지 말고 다음으로 중단하라.

```text
BLOCKED_BASELINE_REGRESSION
```

---

# 5. Basilisk frame/convention adapter 계약

`bench/tasks/generator/basilisk_unit_st.py` 안에 Basilisk truth와 Gate B1 dataset을 연결하는 source of truth를 구현하라.

허용 import:

```text
Python standard library
numpy
Basilisk runtime modules
bench.estimators.mekf
bench.tasks.generator.mekf_events
bench.utils.seeding의 순수 deterministic helper
```

금지 import:

```text
torch
bench.runners
bench.models
bench.metrics
YAML/config loader
visualization module
legacy MRP EKF
```

## 5.1 명시적인 identity/version 상수

최소 다음 의미의 상수를 둬라.

```text
GENERATOR_ID = "basilisk-unit-st-v1"
SIMULATOR_ADAPTER_VERSION = "basilisk-sigmaBN-to-qNB-v1"
SENSOR_MODEL_VERSION = "parameterized-gyro-st-v1"
CONVENTION_ID = Gate B1과 동일한 locked convention
SCHEMA_VERSION = Gate B1 schema에서 읽음
SEED_POLICY_VERSION = Gate B1 또는 명시적인 compatible version
```

Gate B1 상수와 의미가 중복되면 import하여 사용하고 복사본을 별도 source of truth로 만들지 마라.

## 5.2 `sigma_BN`을 이름만 보고 해석하지 말 것

`Basilisk` recorder field 이름만으로 다음을 가정하지 마라.

- `MRP2C(sigma_BN)`이 바로 `R_NB`인지 `C_BN`인지
- `MRP2EP` 결과가 locked `q_NB`인지 그 inverse인지
- `omega_BN_B`의 부호가 Gate A propagation과 같은지

먼저 candidate conversion을 분리하여 executable proof로 판정하라.

최종 public adapter는 다음과 동등한 의미를 갖는다.

```text
basilisk_sigma_BN_to_q_NB(sigma_BN) -> float64 (4,)
```

출력은 반드시 다음이다.

```text
scalar-first Hamilton
active body-to-navigation q_NB
unit quaternion
Gate A quat_to_dcm(q_NB) = R_NB
```

내부에서 Basilisk `MRP2C`, `MRP2EP`, `C2MRP` 등을 사용할 수 있지만, 최종 방향은 아래 proof로 결정한다.

## 5.3 정적 basis-vector proof

최소 다음 attitude를 시험하라.

```text
identity
+x 90 deg, -x 90 deg
+y 90 deg, -y 90 deg
+z 90 deg, -z 90 deg
최소 10개의 deterministic arbitrary axis-angle cases
```

±90°의 closed-form MRP 후보는 축과 `tan(pi/8)`을 사용하여 구성하고, 부호 후보를 모두 비교하라. 단순히 Basilisk conversion 함수의 round trip만 시험하여 방향을 결정하지 마라.

각 경우 다음을 비교하라.

1. Basilisk에 설정한 `sigma_BNInit`
2. zero-rate/zero-torque 초기 recorder의 `sigma_BN`
3. candidate adapter의 `q_NB`
4. Gate A `quat_to_dcm(q_NB)`가 body basis를 navigation frame의 기대 방향으로 매핑하는지
5. 그 transpose가 Basilisk `MRP2C`의 물리 변환과 일치하는지

예를 들어 locked active +90° z 회전은 body `+x` basis를 navigation `+y`로 매핑해야 한다. 모든 축에 대해 세 basis vector 전체를 검사하라.

proof는 최종 선택한 관계를 다음처럼 명시적으로 문서화해야 한다.

```text
MRP2C(sigma_BN)의 실제 의미
R_NB와의 transpose/inverse 관계
최종 q_NB 생성식
검증된 basis mapping
```

## 5.4 MRP shadow-set proof

`||sigma|| > 0`인 deterministic attitude들에 대해 다음 shadow representation을 구성하라.

```text
sigma_shadow = -sigma / (sigma^T sigma)
```

원래 MRP와 shadow MRP가 adapter에서 같은 physical DCM을 생성해야 한다.

비교는 raw quaternion component equality가 아니라 다음으로 한다.

```text
DCM equality
abs(q1 dot q2) = 1 within float64 tolerance
```

invalid/nonfinite MRP는 fail-loud한다.

## 5.5 time-series quaternion representation

각 recorder sample을 `q_NB`로 변환한 뒤:

1. unit norm을 보장한다.
2. 첫 sample은 deterministic representative를 사용한다.
3. 후속 sample은 이전 sample과 dot product가 음수일 때만 sign을 뒤집어 representation continuity를 유지한다.
4. global `q0 >= 0` rule을 time series 전체에 강제하지 않는다.
5. sign alignment가 physical DCM을 바꾸지 않음을 시험한다.

이 continuity alignment는 truth representation 안정화를 위한 것이며 estimator에 truth를 제공하는 기능이 아니다.

---

# 6. `omega_BN_B`와 동역학 proof

## 6.1 minimal truth spacecraft

frame/dynamics proof와 최초 dataset은 다음 Tier-0 truth로 제한한다.

```text
single rigid spacecraft hub
spherical inertia 또는 명시적인 isotropic inertia
zero external torque
no gravity/orbit/environment
fixed mass and CoM at body origin
known initial sigma_BN
known initial omega_BN_B
```

대표 mass와 inertia는 실제 비행체 사양이 아니라 `representative_normalized_UNIT-ST`로 metadata에 표시한다.

spherical inertia를 사용하는 이유는 zero torque에서 arbitrary body angular rate가 body frame에서 일정하여 Gate A analytic propagation과 직접 비교할 수 있기 때문이다.

## 6.2 angular-rate semantic proof

최소 다음 rate cases를 시험하라.

```text
zero rate
+x, -x, +y, -y, +z, -z single-axis rates
최소 10개의 deterministic arbitrary body-rate vectors
```

각 case에서 recorder의 `omega_BN_B`와 변환된 `q_NB(t)`에 대해 다음 관계를 검증하라.

```text
q_NB(t + dt) physically equals q_NB(t) ⊗ Exp_q(omega_BN_B * dt)
```

또는 동등한 right-local increment 관계:

```text
Log_q(q_NB(t)^-1 ⊗ q_NB(t+dt)) / dt ≈ omega_BN_B
```

비교에는 Gate A quaternion/Log/Exp helper를 사용하고 수학을 복사하지 마라.

다음을 문서화하라.

```text
omega_BN_B의 frame
omega_BN_B의 unit
Gate A gyro propagation과의 sign 관계
simulator integration error의 실제 측정값
```

## 6.3 수치 합격 정책

정적 identity/±90° basis proof는 float64 roundoff 수준으로 맞아야 한다.

동적 proof는 simulator integration error가 존재할 수 있으므로:

1. tolerance를 구현 후 실패를 피하기 위해 임의 완화하지 마라.
2. 최초 test parameter와 tolerance를 문서에 먼저 기록하라.
3. 짧은 duration, bounded rate, 충분히 작은 task step을 사용하라.
4. task step을 절반으로 줄인 convergence check를 포함하라.
5. finer-step error가 coarse-step error보다 증가하면 실패하라.
6. 측정된 max geodesic/rate increment error를 report에 기록하라.

권장 시작값은 다음이며 실제 API 제약으로 조정 시 근거를 남겨라.

```text
duration <= 1 s
rate norm <= 0.2 rad/s
coarse step <= 0.01 s
fine step = coarse step / 2
max fine-step attitude error target <= 1e-8 rad
```

`1e-8 rad`를 넘으면 tolerance를 자동 확대하지 말고 원인 분석 후 Gate B2 STOP 또는 explicit BLOCKED로 보고하라.

---

# 7. Basilisk UNIT-ST config와 truth generator

## 7.1 immutable config

frozen dataclass 또는 동등한 immutable config를 구현하라.

최소 항목:

```text
num_trajectories
duration_s
gyro_rate_hz
star_tracker_rate_hz
master_seed
initial attitude distribution/range
initial body-rate distribution/range
constant gyro-bias distribution/range
gyro white-noise standard deviation
star-tracker tangent-noise covariance 또는 standard deviation
star-tracker raw-sign stream enable
representative mass
representative spherical inertia
Basilisk task step 또는 truth rate
train/val/test fractions
split seed namespace
```

초기 Gate B2에서는 다음을 강제하라.

```text
gyro rate는 truth recorder/task grid와 정렬
star-tracker rate는 gyro rate의 정수 약수
ST timestamp는 gyro timestamp subset
arrival_time_s == measurement_time_s
all nominal events valid
```

잘못된 rate, negative time/noise, non-SPD covariance, non-spherical proof inertia, zero/negative mass는 fail-loud한다.

## 7.2 truth generation

각 trajectory는 Basilisk가 생성한 최소 다음 truth를 갖는다.

```text
truth_time_s
q_NB_true
omega_true_B_rad_s
b_g_true_rad_s
```

- `q_NB_true`: recorder `sigma_BN`을 검증된 adapter로 변환
- `omega_true_B_rad_s`: recorder `omega_BN_B`
- `b_g_true_rad_s`: project-owned sensor layer의 deterministic constant bias truth
- 모든 truth 배열: float64, finite, correct shape
- truth quaternion: unit norm and representation-continuous

bias는 Basilisk rigid-body state가 아니라 sensor truth parameter다. 이 사실을 manifest와 contract에 명확히 기록하라.

## 7.3 gyro sensor-output layer

다음을 사용하라.

```text
omega_m = omega_true_B + b_g_true + n_g
```

- unit: rad/s
- frame: body
- constant true bias per trajectory
- white Gaussian measurement noise
- independent deterministic noise stream
- event at each gyro epoch after initial posterior time
- no missing/saturation/random walk in Gate B2

기존 `basilisk_imu_adcs.py`의 sensor model을 import하거나 복사할 필요는 없다. 사용하는 경우에도 legacy output contract나 state semantics를 가져오지 말고, 이번 식과 truth boundary를 별도로 증명해야 한다.

## 7.4 star-tracker sensor-output layer

다음을 사용하라.

```text
q_ST = q_NB_true ⊗ Exp_q(n_ST)
```

- `n_ST`: zero-mean right-local tangent noise, rad
- covariance: strictly SPD float64 `[3,3]`
- low-rate timestamp: gyro grid subset
- optional deterministic raw sign flip stream
- physical attitude unchanged under `q/-q`
- no outage/false solution/latency in Gate B2

Gate A quaternion helper를 import하여 사용하고 Exp/Log/multiply를 중복 구현하지 마라.

이 sensor layer는 project-owned parameterized wrapper임을 문서에 명시하라.

## 7.5 seed isolation

master seed에서 최소 다음 stream을 분리하라.

```text
truth initial attitude
truth initial angular rate
sensor bias
gyro white noise
star-tracker tangent noise
star-tracker raw sign
trajectory split
```

model/estimator ID를 seed derivation에 포함하지 마라.

다음 동작을 시험하라.

- sensor-noise seed만 변경: Basilisk truth hash 유지, sensor hash 변화
- bias seed만 변경: attitude/rate truth 유지, bias truth와 gyro payload 변화
- truth initial-condition seed 변경: truth hash 변화
- ST sign seed 변경: raw ST payload hash가 바뀔 수 있으나 physical replay 동일
- split seed 변경: dataset physical content hash 유지, split membership만 변화

---

# 8. Gate B1 schema/serialization/replay 재사용

## 8.1 새 schema를 만들지 말 것

Gate B2는 `bench.tasks.generator.mekf_events`의 다음 source of truth를 그대로 사용한다.

```text
event metadata dtype/shape
sensor codes
truth/event/payload container
validation
serialization/load
semantic hashes
trajectory split/select
direct replay
```

동일 의미의 새로운 dataclass, serializer, hash function, replay engine을 `basilisk_unit_st.py`에 중복 구현하지 마라.

Gate B1 manifest가 simulator identity를 담을 수 없다면 파일을 수정하지 말고 다음으로 중단하라.

```text
BLOCKED_GATE_B1_MANIFEST_EXTENSION_REQUIRED
```

그리고 필요한 최소 field와 migration 영향을 보고하라.

## 8.2 simulator identity

manifest/resolved config의 허용된 확장 영역에 최소 다음을 기록하라.

```text
generator_id
generator version
simulator adapter version
sensor model version
schema version
convention ID
seed-policy version
full resolved Basilisk UNIT-ST config
Python version
NumPy version
SciPy version
Basilisk runtime version
bsk distribution version
Gate A source fingerprint
Gate B1 schema/replay source fingerprint
basilisk_unit_st source fingerprint
frame-proof ID 또는 proof document fingerprint
```

hash 계산 순환을 피하는 규칙을 문서화하라.

## 8.3 serialization and replay

Basilisk dataset도 Gate B1 artifact 형식을 그대로 사용한다.

```text
manifest.json
truth.npz
events.npz
```

요구사항:

- `allow_pickle=False`
- object dtype 없음
- round-trip semantic hashes exact-equal
- direct replay output exact-equal after round trip
- replay API에 truth/oracle/label/future input 없음
- generated truth는 replay input으로 전달되지 않음

## 8.4 direct replay validation

최소 두 config를 검증하라.

### Case 1 — zero-noise exactness

```text
known initial state equals true initial q/b
fixed correct Q_c compatible with zero-noise case
zero gyro noise
zero ST tangent noise but strictly SPD nominal R_ST as required by core
```

필터가 event schedule을 정확히 처리하며 truth와 float64/simulator integration tolerance 내에서 일치해야 한다.

### Case 2 — representative noisy smoke

```text
small initial attitude/bias error
constant bias
positive gyro white noise
positive ST tangent noise
low-rate ST
```

성능 우월성을 주장하지 말고 다음만 합격 기준으로 둬라.

```text
all outputs finite
unit quaternion
posterior P symmetric/SPD
all ST S symmetric/SPD
deterministic repeated replay
serialization round-trip replay equality
bounded short-horizon attitude/bias error 기록
```

bias convergence threshold는 Gate C/후속 UNIT-ST validation 전에는 연구 합격 기준으로 사용하지 마라.

---

# 9. 필수 시험

`tests/test_basilisk_unit_st_generator.py`에 최소 다음을 포함하라.

## 9.1 import/runtime boundary

1. explicit Python에서 Basilisk import 및 runtime version 기록
2. `basilisk_unit_st` import가 runner/model/metric/torch/viz를 import하지 않음
3. missing Basilisk를 silent fallback으로 synthetic truth로 바꾸지 않음

## 9.2 static frame proof

4. identity `sigma_BN` → identity `q_NB`
5. 각 축 +90° 및 -90° body-basis mapping
6. deterministic arbitrary attitude 최소 10개
7. `MRP2C`와 final `R_NB/C_BN` relation
8. MRP shadow-set physical equivalence
9. nonfinite/invalid MRP fail-loud
10. quaternion unit norm and deterministic conversion

## 9.3 dynamic rate proof

11. zero rate preserves attitude
12. 각 축 ±rate의 Gate A right-propagation 일치
13. arbitrary body-rate 최소 10개
14. `Log(q_k^-1 q_{k+1})/dt`와 recorder `omega_BN_B`의 sign/frame 일치
15. spherical-inertia zero-torque constant rate 유지
16. coarse/fine task-step convergence
17. fine-step max attitude error target 충족

## 9.4 generator/schema

18. exact truth/event/payload dtypes and shapes
19. zero-latency exact equality
20. same-time gyro-before-ST order
21. ST timestamps are gyro timestamp subset
22. all nominal events valid
23. same seed/config regeneration hashes equal
24. simulator identity/version present in manifest
25. sensor seed changes preserve Basilisk attitude/rate truth
26. truth seed changes truth hash
27. bias seed changes bias truth and gyro payload without changing attitude/rate truth
28. ST sign seed preserves physical measurement
29. trajectory IDs unique and split disjoint
30. serialization round trip exact semantic hashes
31. corrupted or mismatched simulator identity/hash rejected by existing loader contract when applicable

## 9.5 replay and safety

32. zero-noise direct replay matches Basilisk truth within declared tolerance
33. representative noisy replay finite/unit/SPD
34. repeated replay deterministic
35. serialization round-trip replay equality
36. all-ST-sign-negated stream produces same physical posterior/bias/P/residual/S
37. truth arrays are not passed to replay and are not mutated
38. Gate A state immutability retained
39. non-SPD ST covariance and malformed config fail-loud

테스트 실패를 해결하기 위해 tolerance를 사후 확대하거나 skip/xfail을 추가하지 마라.

---

# 10. Property sweep

별도 evidence log로 최소 다음을 실행하라.

```text
5 dataset seeds
각 seed당 최소 3 trajectories
same-seed regeneration
static frame proof summary
constant-rate dynamic proof summary
serialization round trip
ST sign-paired replay
split disjointness
finite/unit-quaternion/Cholesky safety
```

Basilisk runtime 때문에 전체 시간이 과도하면 trajectory duration을 줄일 수 있지만, seed/trajectory 개수와 proof 종류를 줄이지 마라.

최소 다음 값을 요약하라.

```text
max static basis-vector error
max MRP shadow DCM error
max coarse-step attitude error
max fine-step attitude error
max local rate-increment error
max quaternion norm deviation
minimum posterior P eigenvalue
minimum ST S eigenvalue
semantic hash reproducibility count
```

---

# 11. Required test commands

## 11.1 Gate B2 신규 test

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_basilisk_unit_st_generator.py
```

## 11.2 Gate A regression

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_mekf_conventions.py \
  tests/test_mekf_core.py
```

## 11.3 Gate B1 regression

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_mekf_events.py \
  tests/test_unit_st_synthetic.py \
  tests/test_mekf_replay.py
```

## 11.4 legacy regression

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_basilisk_imu_generator.py \
  tests/test_basilisk_mrp_ekf.py \
  bench/tests/test_generator_contract_tg0.py \
  bench/tests/test_adcs_event_metrics.py
```

모든 명령의 full stdout/stderr, exit code, duration을 `03B_*` evidence log에 저장하라.

---

# 12. 문서 산출물

## 12.1 `P1A_BASILISK_FRAME_CONVENTION_PROOF.md`

최소 포함 내용:

```text
목적
실행 runtime/version
검토한 candidate convention
closed-form MRP test vectors
identity/±90°/arbitrary basis proof
MRP2C의 실제 물리 의미
locked R_NB/C_BN/q_NB relation
MRP shadow proof
omega_BN_B frame/sign/unit proof
constant-rate dynamics proof
coarse/fine convergence 결과
최종 locked adapter formula
반대/inverse 후보가 왜 기각됐는지
```

## 12.2 `P1A_BASILISK_UNIT_ST_CONTRACT.md`

최소 포함 내용:

```text
truth spacecraft Tier-0 definition
body/inertial frame
mass/inertia의 representative status
Basilisk process/task/recorder structure
truth arrays and units
gyro measurement equation
ST measurement equation
sensor layer가 project-owned wrapper라는 명시
seed namespaces
event schedule and zero latency
Gate B1 schema/serialization reuse
manifest simulator identity
truth/sensor/estimator boundary
deferred features
```

## 12.3 `P1A_GATE_B2_TEST_MATRIX.md`

각 test에 다음 열을 사용하라.

```text
Test ID
requirement
input
expected behavior
tolerance
actual result
evidence log
status
```

## 12.4 `P1A_GATE_B2_VALIDATION_REPORT.md`

최종 report에는 최소 다음을 포함하라.

```text
결정
생성 파일
runtime versions
frame conversion final formula
static proof results
dynamic rate proof results
generator/sensor contract
hash/seed/split results
replay results
new test result
Gate A regression
Gate B1 regression
legacy regression
dirty-tree integrity
blocking/deferred items
Gate B2 GO/STOP
```

---

# 13. Dirty-tree 종료 검사

실행 종료 전에 다음을 수행하라.

1. pre-existing dirty path의 status/content fingerprint 재계산
2. Gate A frozen file fingerprint 비교
3. Gate B1 frozen file fingerprint 비교
4. exact allowlist 밖 새/변경 source path 확인
5. staged diff가 비어 있는지 확인
6. allowlist 파일에 대해서만 whitespace 검사
7. agent-only patch/stat/changed-path list 생성
8. concurrent external artifact path/status ledger 생성

필수 evidence 예시:

```text
experiments/phase1a/agent_logs/03B_status_after.txt
experiments/phase1a/agent_logs/03B_dirty_integrity_check.tsv
experiments/phase1a/agent_logs/03B_gate_a_frozen_check.tsv
experiments/phase1a/agent_logs/03B_gate_b1_frozen_check.tsv
experiments/phase1a/agent_logs/03B_agent_only.patch
experiments/phase1a/agent_logs/03B_agent_only_stat.txt
experiments/phase1a/agent_logs/03B_changed_paths.txt
experiments/phase1a/agent_logs/03B_allowlist_whitespace.txt
experiments/phase1a/agent_logs/03B_runtime_versions.txt
experiments/phase1a/agent_logs/03B_frame_proof.txt
experiments/phase1a/agent_logs/03B_dynamic_proof.txt
experiments/phase1a/agent_logs/03B_property_sweep.txt
experiments/phase1a/agent_logs/03B_new_tests.txt
experiments/phase1a/agent_logs/03B_gate_a_regression.txt
experiments/phase1a/agent_logs/03B_gate_b1_regression.txt
experiments/phase1a/agent_logs/03B_legacy_regression.txt
```

기존 dirty path, Gate A/B1 frozen file, allowlist 밖 source에 예상하지 않은 변화가 있으면 자동 복원하지 말고 다음으로 중단하라.

```text
BLOCKED_UNINTENDED_CHANGE
```

---

# 14. 금지된 수치·구현 우회

다음을 사용하지 마라.

```text
pseudo-inverse
explicit inverse fallback
eigenvalue clipping
silent covariance jitter
non-SPD 자동 보정
NaN/Inf 무시
frame mismatch를 quaternion sign flip만으로 은폐
Basilisk import 실패 시 synthetic generator fallback
failed test skip/xfail
tolerance 사후 확대
legacy code 또는 expected value 수정
```

허용되는 것은 다음뿐이다.

```text
Gate A의 명시적 quaternion normalization
Gate A의 deterministic q/-q alignment
truth time-series의 adjacent-sign continuity alignment
roundoff 수준의 documented comparison tolerance
```

---

# 15. 최종 판정 형식

최종 응답은 다음 형식을 사용하라.

```text
Status:
  PASS_GATE_B2
  또는 BLOCKED_...
  또는 FAIL_GATE_B2

Runtime identity: PASS/FAIL
Static sigma_BN -> q_NB frame proof: PASS/FAIL
MRP shadow invariance: PASS/FAIL
omega_BN_B sign/frame/unit proof: PASS/FAIL
Constant-rate dynamics/convergence: PASS/FAIL
Basilisk truth generation: PASS/FAIL
Gyro sensor layer: PASS/FAIL
Star-tracker sensor layer: PASS/FAIL
Gate B1 schema/serialization reuse: PASS/FAIL
Determinism/semantic hashes: PASS/FAIL
Seed isolation: PASS/FAIL
Trajectory split: PASS/FAIL
Direct replay: PASS/FAIL
Truth boundary: PASS/FAIL
Numerical/replay safety: PASS/FAIL
Gate A regression: PASS/FAIL
Gate B1 regression: PASS/FAIL
Legacy regression: PASS/FAIL
Dirty-tree integrity: PASS/FAIL
Gate B2: GO/STOP
```

반드시 다음을 보고하라.

```text
생성 파일
final adapter formula
MRP2C의 검증된 방향 의미
max static basis error
max shadow-set DCM error
max coarse/fine dynamic attitude error
max local rate-increment error
runtime versions
신규 test count/result
Gate A/B1/legacy results
agent-only diff/stat
blocking/deferred items
```

Gate B2가 PASS하더라도 Gate C를 자동으로 시작하지 말고 종료하라.

---

# 16. Gate B2 합격 조건

다음이 모두 만족될 때만 `PASS_GATE_B2`와 `Gate B2: GO`를 선언한다.

1. `sigma_BN` → locked `q_NB` 관계가 identity/±90°/arbitrary basis proof로 확정됨
2. MRP shadow-set physical invariance가 통과함
3. `omega_BN_B`가 Gate A body-rate propagation과 같은 frame/sign/unit임이 증명됨
4. spherical-inertia zero-torque constant-rate trajectory가 analytic propagation과 declared tolerance 내 일치함
5. coarse/fine convergence가 정상임
6. Basilisk truth q/rate와 sensor bias가 분리 저장됨
7. gyro와 ST measurement equations가 locked contract와 일치함
8. Gate B1 schema/serialization/hash/split/replay를 수정 없이 재사용함
9. 같은 config/seed의 semantic hashes가 재현됨
10. seed namespaces의 독립성이 증명됨
11. direct replay가 finite/unit-quaternion/SPD를 유지함
12. raw ST `q/-q` stream의 physical posterior가 동일함
13. truth/oracle/label leakage가 없음
14. Gate A, Gate B1, legacy regression이 모두 통과함
15. dirty-tree integrity가 통과함
16. allowlist 밖 source/config/dependency가 수정되지 않음

하나라도 실패하면 GO를 선언하지 마라.

다음 단계의 제목만 report 마지막에 기록하라.

```text
Phase 1A Gate C — Canonical MEKF Geodesic/Bias/NIS/NEES/SPD Metrics
```
