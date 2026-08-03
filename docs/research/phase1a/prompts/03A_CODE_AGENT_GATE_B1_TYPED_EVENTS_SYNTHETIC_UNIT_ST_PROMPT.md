# Phase 1A Gate B1 실행 계약

## Typed Event Schema + Synthetic UNIT-ST + Deterministic Direct-Core Replay

당신은 `/home/dss-pc-05/bench` repository에서 Phase 1A Gate B1만 수행하는 구현 agent다.

이번 실행의 목표는 **검증 완료된 Gate A MEKF core를 수정하지 않고**, gyro와 quaternion star tracker를 위한 versioned typed event schema, analytic synthetic UNIT-ST data generator, deterministic serialization/hash/split, direct-core replay를 구현하고 검증하는 것이다.

이번 실행에서 Basilisk, benchmark runner, model registry, official metric module, visualization, neural network를 구현하거나 수정하지 않는다.

---

# 0. 최우선 실행 정책

## 0.1 현재 working tree 수용

실행 시작 시점의 현재 working tree 전체를 사용자 승인 기준선으로 사용한다.

다음을 승인 조건으로 사용하거나 비교하지 마라.

- branch 이름
- 현재 HEAD
- 과거 HEAD
- commit history
- merge-base 또는 ancestry
- 과거 commit delta
- repository 전체 `git diff --check`
- approval marker의 commit SHA

branch와 HEAD를 provenance로 기록하는 것은 허용하지만, 그것을 이유로 중단하지 마라.

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

구현 전에 현재 tracked/staged/untracked 상태를 recoverable snapshot과 hash ledger로 보존하라.

기존 dirty 경로는 이번 exact allowlist를 제외하고 실행 전후 status/content fingerprint가 동일해야 한다.

## 0.3 Gate A source 동결

다음 파일은 읽고 import할 수 있지만 수정하지 마라.

```text
bench/estimators/__init__.py
bench/estimators/mekf.py
tests/test_mekf_conventions.py
tests/test_mekf_core.py
docs/research/phase1a/P1A_IMPLEMENTATION_CONTRACT.md
docs/research/phase1a/P1A_TEST_MATRIX.md
experiments/phase1a/reports/P1A_MATH_VALIDATION_REPORT.md
```

Gate A core의 API가 부족하다고 판단되더라도 임의로 변경하지 마라. 필요한 기능을 event/replay layer에서 조합할 수 없을 때는 다음으로 중단하라.

```text
BLOCKED_GATE_A_INTERFACE_CHANGE_REQUIRED
```

그리고 필요한 API, 이유, 최소 변경안을 보고하라.

## 0.4 외부 동시 artifact 처리

다음처럼 repository의 source가 아닌 별도 실행 artifact root에 외부 파일이 추가될 수 있다.

```text
artifacts/benchmark_write_control/**
```

그 경로의 새 파일은 읽거나 수정하지 말고 concurrent external ledger에만 기록하라. 그것만으로 Gate B1을 실패시키지 마라.

다만 이번 allowlist target, Gate A source, 아래 shared-critical 경로가 실행 도중 다른 프로세스에 의해 바뀌면 즉시 중단하라.

```text
BLOCKED_CONCURRENT_SOURCE_CHANGE
```

## 0.5 명시적 Python

모든 Python/test 명령은 다음 interpreter를 명시적으로 사용하라.

```text
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python
```

pytest에는 기본적으로 다음을 적용하라.

```text
PYTHONDONTWRITEBYTECODE=1
-p no:cacheprovider
```

---

# 1. 반드시 읽을 문서와 코드

다음 문서를 처음부터 끝까지 읽어라.

```text
docs/research/phase0a/decision_lock/P0A_PHASE_0A_SYNTHESIS.md
docs/research/phase0a/decision_lock/P0_01_DECISION_LEDGER.md
docs/research/phase0a/decision_lock/P0_02_TRUTH_SENSOR_ESTIMATOR_BOUNDARY.md
docs/research/phase0a/decision_lock/P0_05_MEKF_MATH_CONTRACT.md
docs/research/phase0a/decision_lock/P0A_IMMEDIATE_TEST_SPEC.md

docs/research/phase1a/P1A_REPOSITORY_AUDIT.md
docs/research/phase1a/P1A_IMPLEMENTATION_MAP.md
docs/research/phase1a/P1A_RISK_REGISTER.md
docs/research/phase1a/P1A_IMPLEMENTATION_CONTRACT.md
docs/research/phase1a/P1A_TEST_MATRIX.md
docs/research/phase1a/P1A_GATE_A_FINAL_APPROVAL.md
```

다음 코드는 Gate A API와 안전한 seed helper를 확인하기 위해 읽을 수 있다.

```text
bench/estimators/mekf.py
bench/utils/seeding.py
```

기존 generator/contract는 public behavior와 금지 경계를 확인하기 위한 read-only 참고만 허용한다.

```text
bench/tasks/generator/contract.py
bench/tasks/data_format.py
bench/tasks/bench_generated.py
```

기존 all-float32 `GeneratorOutput` 계약을 이번 typed event source of truth로 재사용하지 마라.

---

# 2. 이번 Gate의 정확한 범위

Gate B1에서 구현할 것은 다음뿐이다.

1. versioned typed gyro/ST event schema
2. deterministic semantic hash와 serialization round trip
3. trajectory identity와 whole-trajectory split
4. Basilisk 비의존 analytic synthetic UNIT-ST generator
5. zero-latency event ordering
6. 검증된 Gate A core를 호출하는 direct replay
7. same seed / separate truth-sensor seed 재현성 검증
8. truth leakage가 없는 replay API 검증

Gate B1에서 구현하지 말아야 할 것은 다음이다.

```text
Basilisk import 또는 simulator adapter
sigma_BN -> q_NB frame proof
실제 Basilisk gyro/ST sensor generator
nonzero latency 또는 OOSM buffer
magnetometer 또는 sun sensor
canonical NIS/NEES metric module
benchmark ModelAdapter
model registry
generator dispatch
run_suite integration
legacy NPZ/cache migration
suite YAML
visualization/dashboard
KalmanNet, Split-KalmanNet, ANN, SNN, FPGA
Package C experiment
```

---

# 3. Exact allowlist

## 3.1 생성 가능한 구현·시험·문서 파일

다음 파일만 새로 생성할 수 있다.

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

실행 evidence는 다음 prefix 아래에 생성할 수 있다.

```text
experiments/phase1a/agent_logs/03A_*
experiments/phase1a/preflight_snapshots/03A_*/**
```

## 3.2 target collision

위 구현·시험·계약·보고서 target이 실행 전에 이미 존재하면 덮어쓰지 마라.

다음으로 중단하라.

```text
BLOCKED_TARGET_EXISTS
```

단, 이번 prompt 파일과 `P1A_GATE_A_FINAL_APPROVAL.md`는 입력 문서이므로 target collision 대상이 아니다.

## 3.3 수정 금지 shared-critical 경로

다음을 수정하지 마라.

```text
bench/estimators/**
bench/models/**
bench/metrics/**
bench/runners/**
bench/configs/**
bench/tasks/bench_generated.py
bench/tasks/data_format.py
bench/tasks/generator/contract.py
bench/tasks/generator/__init__.py
pyproject.toml
uv.lock
requirements*
docs/research/phase0a/**
third_party/**
viz/**
visualization/**
기존 tests와 기존 기대값
```

repository-wide formatter, import sorter 또는 whitespace fixer를 실행하지 마라.

---

# 4. Preflight와 baseline

## 4.1 snapshot

다음 경로를 만들어라.

```text
experiments/phase1a/preflight_snapshots/03A_<UTC_TIMESTAMP>/
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
ALLOWLIST_EXISTENCE_BEFORE.tsv
SNAPSHOT_MANIFEST.md
```

branch/HEAD는 기록만 하고 검토·승인에 사용하지 마라.

대용량 untracked archive 생성 실패는 차단 사유로 삼지 말고 path/hash/status를 우선 보존하라.

## 4.2 구현 전 test

먼저 Gate A를 재확인하라.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_mekf_conventions.py \
  tests/test_mekf_core.py
```

기대: 현재 Amendment A1 test 전체 PASS. 현재 기록 기준은 `55 passed`지만 숫자를 hard-code하여 통과를 조작하지 말고 exit code 0과 실제 수집 결과를 기록하라.

기존 legacy subset도 실행하라.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_basilisk_imu_generator.py \
  tests/test_basilisk_mrp_ekf.py \
  bench/tests/test_generator_contract_tg0.py \
  bench/tests/test_adcs_event_metrics.py
```

기대: exit code 0. 현재 기록 기준은 `18 passed, 5 subtests passed`다.

기존 baseline이 실패하면 기존 코드나 기대값을 수정하지 말고 다음으로 중단하라.

```text
BLOCKED_BASELINE_REGRESSION
```

---

# 5. Typed event schema 계약

`bench/tasks/generator/mekf_events.py`를 standalone Phase 1A typed schema와 replay utility의 source of truth로 구현하라.

허용 import:

```text
Python standard library
numpy
bench.estimators.mekf
선택적으로 bench.utils.seeding의 순수 deterministic helper
```

금지 import:

```text
Basilisk
torch
bench.runners
bench.models
bench.metrics
YAML loader
visualization module
```

## 5.1 고정 schema version

명시적인 상수를 둬라.

```text
SCHEMA_VERSION = "p1a-mekf-events-v1"
GENERATOR_ID = "synthetic-unit-st-v1"
SEED_POLICY_VERSION = "p1a-separated-streams-v1"
CONVENTION_ID = "qNB-scalar-first-hamilton-right-v1"
```

이름은 코드 convention에 맞게 약간 조정할 수 있지만 의미와 versioning은 유지하라.

## 5.2 sensor code

정수 enum 또는 동등한 상수를 사용하라.

```text
GYRO = 1
STAR_TRACKER = 2
```

저장 dtype은 `int16`로 고정하라.

## 5.3 event table

각 event에 최소 다음 필드를 보존하라.

| 필드 | dtype/shape | 의미 |
|---|---|---|
| `trajectory_id` | `int64 [E]` | 불변 trajectory identity |
| `sensor_code` | `int16 [E]` | gyro 또는 ST |
| `measurement_time_s` | `float64 [E]` | 물리 측정 epoch |
| `arrival_time_s` | `float64 [E]` | estimator 이용 가능 epoch |
| `event_order` | `int64 [E]` | 동일 arrival time tie-break 포함 전체 순서 |
| `valid` | `bool [E]` | packet validity |
| `payload_index` | `int64 [E]` | sensor-specific payload table index |

payload는 zero-fill union array로 저장하지 마라. sensor-specific typed table을 별도로 둬라.

### Gyro payload

```text
gyro_omega_m_rad_s: float64 [G,3]
```

### Star-tracker payload

```text
st_q_NB_meas: float64 [S,4]
st_R_rad2: float64 [S,3,3]
```

모든 ST quaternion은 normalized scalar-first active `q_NB`다. raw representation의 `q/-q`는 모두 허용한다.

`payload_index`는 해당 sensor table을 정확히 가리켜야 하며, 중복·범위 오류·sensor/payload mismatch는 fail-loud validation으로 거부하라.

## 5.4 truth table

truth는 estimator event와 별도 구조로 저장하라. 최소 다음을 포함하라.

| 필드 | dtype/shape | 의미 |
|---|---|---|
| `trajectory_id` | `int64 [N]` | event와 조인되는 trajectory ID |
| `truth_offsets` | `int64 [N+1]` | flattened per-trajectory truth offsets |
| `truth_time_s` | `float64 [T_total]` | truth epoch |
| `q_NB_true` | `float64 [T_total,4]` | scalar-first active B-to-N truth |
| `b_g_true_rad_s` | `float64 [T_total,3]` | true gyro bias |
| `omega_true_B_rad_s` | `float64 [T_total,3]` | true body angular rate |

truth table은 generator와 evaluation test만 사용한다. direct replay API는 truth table을 인자로 받지 않는다.

## 5.5 zero-latency와 event ordering

Gate B1에서는 모든 valid event에 대해 다음을 정확히 강제하라.

```text
arrival_time_s == measurement_time_s
```

nonzero latency를 발견하면 조용히 정렬하거나 처리하지 말고 명시적으로 거부하라.

각 trajectory의 event는 다음 key로 정렬한다.

```text
(arrival_time_s, event_order)
```

동일 timestamp에서는 다음 순서를 사용하라.

```text
gyro propagation first
star-tracker update second
```

최초 direct replay semantics를 다음처럼 잠가라.

1. initial state는 `t0` posterior다.
2. gyro event는 `t_k`에서 이전 filter time부터 `t_k`까지 `dt`를 계산하여 propagation한다.
3. 최초 gyro propagation event는 `t_1 > t_0`에 존재한다.
4. ST event는 gyro event가 같은 timestamp까지 propagation한 뒤 update한다.
5. Gate B1 synthetic ST timestamp는 gyro timestamp의 subset이다.
6. 현재 filter time과 일치하지 않는 ST event는 임의 interpolation하지 않고 fail-loud한다.
7. invalid gyro event는 Gate B1 replay에서 허용하지 않는다.
8. invalid ST event의 정책을 구현한다면 명시적으로 skip하고 ledger에 기록하되, 최초 nominal generator는 모든 event를 valid로 만든다.

## 5.6 input validation

다음을 모두 검사하라.

- exact dtype
- exact rank/shape
- finite numeric values
- normalized quaternion
- strictly SPD `R_ST`
- nonnegative, finite time
- zero latency
- unique trajectory IDs
- event trajectory IDs가 truth trajectory IDs의 subset 또는 정확한 대응
- event order의 per-trajectory uniqueness와 monotonic processing order
- payload index 범위와 one-to-one consumption
- no object dtype
- no pickle

invalid input을 자동 보정하지 마라.

---

# 6. Deterministic serialization과 semantic hash

## 6.1 serialization

standalone directory artifact를 구현하라. 예:

```text
<artifact_dir>/
  manifest.json
  truth.npz
  events.npz
```

정확한 내부 파일명은 문서화하는 조건으로 조정할 수 있다.

요구사항:

- `allow_pickle=False`
- object array 금지
- JSON은 UTF-8, sorted keys, canonical separators
- load 시 schema/version/hash 검증
- partial/corrupt artifact fail-loud
- legacy `GeneratorOutput` 또는 float32 extras로 변환하지 않음

raw `.npz` ZIP byte hash가 실행 시각 metadata 때문에 달라질 수 있으므로, deterministic 판정은 **canonical semantic hash**로 수행하라.

## 6.2 canonical semantic hash

각 array를 고정 field order로 hashing하라.

hash input에는 최소 다음을 포함하라.

```text
field name
dtype
shape
C-contiguous canonical numeric bytes
canonical manifest JSON
```

숫자 dtype과 byte order를 명시적으로 canonicalize하라. hash는 SHA-256을 사용하라.

다음 hash를 분리해서 제공하라.

```text
truth_hash
sensor_payload_hash
event_order_hash
manifest_hash
dataset_hash
```

동일 seed/config로 두 번 생성하면 모든 semantic hash가 동일해야 한다.

다음 mutation은 관련 hash를 바꿔야 한다.

- config 한 값 변경
- gyro noise seed 변경
- ST noise seed 변경
- event order 변경
- payload 한 값 변경

truth seed와 sensor seed를 분리하라.

- sensor seed만 바꾸면 truth hash는 유지되어야 한다.
- truth seed를 바꾸면 truth hash가 달라져야 한다.
- ST sign-representation seed만 바꾸면 raw sensor hash는 달라질 수 있지만 Gate A core replay의 physical posterior는 동일해야 한다.

## 6.3 manifest identity

manifest에는 최소 다음을 포함하라.

```text
schema_version
generator_id
generator_version
seed_policy_version
convention_id
full resolved synthetic config
master seed와 파생 stream seed
trajectory IDs와 split seed
Python version
NumPy version
SciPy version
Gate A core source fingerprint
mekf_events source fingerprint
unit_st_synthetic source fingerprint
```

Basilisk version은 Gate B1에는 포함하지 않는다. Gate B2에서 simulator identity로 추가한다.

source fingerprint 계산이 자기 자신을 포함하여 순환 문제가 생기지 않도록, 생성 시점의 file bytes SHA-256을 manifest field로 기록하되 dataset semantic hash 계산 규칙을 명확히 문서화하라.

---

# 7. Trajectory identity와 split

trajectory ID는 `int64`이며 generator index만 임시 row identity로 사용하지 마라. generator namespace, master seed, trajectory index에서 안정적으로 파생하거나 동등한 충돌 방지 방식을 사용하라.

다음 함수를 구현하라.

```text
split_trajectory_ids(...)
select_trajectories(...)
```

요구사항:

- split은 whole trajectory 단위
- train/val/test ID 교집합은 공집합
- 세 split의 합집합은 원본 ID 집합과 동일
- 동일 seed/config는 동일 split
- 입력 ID 순서가 달라도 동일한 split 결과
- event/window 단위 무작위 split 금지
- 중복 trajectory ID fail-loud
- 너무 적은 trajectory 수 또는 잘못된 fraction fail-loud

split 결과와 split hash를 manifest/report에 기록하라.

---

# 8. Synthetic UNIT-ST generator

`bench/tasks/generator/unit_st_synthetic.py`에 Basilisk 비의존 analytic generator를 구현하라.

## 8.1 config

명시적인 frozen dataclass 또는 동등한 immutable config를 사용하라.

최소 항목:

```text
num_trajectories
duration_s
gyro_rate_hz
star_tracker_rate_hz
master_seed
truth_seed namespace
gyro_noise_seed namespace
st_noise_seed namespace
st_sign_seed namespace
split_seed namespace
truth initial attitude/rate/bias range
gyro white-noise standard deviation
star-tracker tangent-noise standard deviation
star-tracker R
```

모든 최초 수치는 제품 사양이 아니라 다음으로 metadata에 표시하라.

```text
representative_normalized_UNIT-ST
```

flight-grade 또는 특정 제품 성능이라고 주장하지 마라.

rate는 Gate B1에서 integer-aligned schedule을 요구한다.

```text
gyro_rate_hz / star_tracker_rate_hz가 양의 정수
```

ST timestamp는 gyro timestamp subset이어야 한다.

## 8.2 analytic truth

각 trajectory는 최소 다음을 갖는다.

- deterministic initial `q_NB`
- deterministic constant 또는 명시적으로 정의한 body angular rate
- deterministic constant gyro bias
- exact quaternion exponential propagation

Gate B1에서는 temperature drift, bias random walk, maneuver event, outage를 넣지 마라.

truth generation은 Gate A convention을 그대로 사용하되 MEKF propagation/update 수학을 복사하지 말고 검증된 quaternion helper를 import하여 사용하라.

## 8.3 gyro measurement

다음을 사용하라.

```text
omega_m = omega_true + b_g_true + n_g
```

- unit: rad/s
- frame: body
- independent per-trajectory deterministic noise stream
- event at each `t_k`, `k >= 1`
- no missing gyro in nominal Gate B1

## 8.4 star tracker measurement

truth quaternion에 right-local tangent Gaussian perturbation을 적용하라.

```text
q_ST = q_true ⊗ Exp_q(n_ST)
```

- `n_ST` unit: rad
- `R_ST = sigma_ST^2 I` 또는 config에 명시한 SPD matrix
- low-rate timestamp는 gyro grid subset
- optional deterministic raw sign flip stream으로 `q` 또는 `-q` representation을 생성
- physical attitude는 sign flip 전후 동일
- no outage/false solution in nominal Gate B1

## 8.5 seed isolation

master seed에서 최소 다음 stream을 이름으로 분리하라.

```text
truth
gyro_noise
star_tracker_noise
star_tracker_sign
trajectory_split
```

model/estimator ID를 seed derivation에 포함하지 마라.

동일 generated artifact는 모든 estimator가 공통으로 사용할 수 있어야 한다.

---

# 9. Direct-core replay

replay utility는 `bench/tasks/generator/mekf_events.py` 또는 `unit_st_synthetic.py` 중 dependency 방향이 자연스러운 위치에 구현하라.

MEKF math는 중복 구현하지 말고 다음 Gate A API를 호출하라.

```text
MEKFState
propagate_state
star_tracker_update
quat_geodesic_angle
```

## 9.1 replay 입력

replay의 public API는 최소 다음만 받는다.

```text
typed event stream
trajectory_id
initial MEKFState
initial_time_s
fixed nominal Q_c
```

truth table, true attitude, true bias, event label, oracle scale을 입력으로 받지 마라.

ST `R`은 ST payload에서 읽는다.

public signature와 source inspection test로 `truth`, `oracle`, `label`, `future` 입력 부재를 확인하라.

## 9.2 replay 출력

최소 다음을 immutable 또는 defensive-copy result로 반환하라.

```text
processed trajectory_id
processed event count
state time history
posterior q_NB history
posterior b_g history
posterior P history
event sensor code history
ST residual history
ST S history 또는 update result evidence
final MEKFState
```

Gate C 공식 metric을 구현하지는 말라. 테스트에서 `quat_geodesic_angle`로 truth와 비교하는 것은 허용한다.

## 9.3 replay invariants

- 입력 event arrays를 수정하지 않음
- prior state를 수정하지 않음
- 같은 event stream을 두 번 replay하면 결과 동일
- serialization round trip 후 결과 동일
- ST raw sign만 반대로 한 physical-equivalent stream은 posterior DCM/bias/P가 동일
- 모든 output finite
- quaternion norm 유지
- 각 posterior P가 symmetric/SPD
- event processing 순서가 schema ordering과 동일

---

# 10. 필수 tests

## 10.1 `tests/test_mekf_events.py`

최소 다음을 검증하라.

1. exact dtypes/shapes
2. invalid dtype/rank/shape fail-loud
3. payload index mismatch/range error fail-loud
4. normalized ST quaternion 요구
5. non-SPD `R_ST` fail-loud
6. zero-latency exact equality
7. nonzero latency explicit rejection
8. deterministic `(arrival_time,event_order)` sorting
9. same timestamp gyro-before-ST
10. serialization round trip
11. no pickle/object arrays
12. canonical semantic hash equality after round trip
13. payload/order/config mutation changes hash
14. corrupted manifest/hash rejection

## 10.2 `tests/test_unit_st_synthetic.py`

최소 다음을 검증하라.

1. same seed/config repeated generation produces identical hashes
2. sensor seed change preserves truth hash and changes sensor hash
3. truth seed change changes truth hash
4. ST sign seed change may change raw sensor hash but preserves physical measurement
5. trajectory IDs unique/int64/stable
6. event schedule/rate counts exact
7. ST timestamps subset of gyro timestamps
8. all Gate B1 events zero-latency
9. gyro equation sign/unit locked
10. ST right-local noise construction locked
11. representative config metadata present
12. whole-trajectory split disjointness/union/determinism
13. input ID order independence
14. different split seed changes split without changing dataset hashes

## 10.3 `tests/test_mekf_replay.py`

최소 다음을 검증하라.

1. zero-noise exact initial-state trajectory replay matches analytic truth within float64 tolerance
2. constant gyro bias and low-rate ST deterministic smoke remains finite and bounded
3. same replay twice produces identical state/evidence arrays
4. serialization round trip replay equivalence
5. q/-q ST representation stream gives physically identical posterior/bias/P
6. long event sequence quaternion norm and P SPD
7. malformed event order or unaligned ST time fail-loud
8. replay public API has no truth/oracle/event-label input
9. event/truth input arrays are not mutated
10. Gate A state immutability remains intact

performance threshold를 연구 성능 주장처럼 설정하지 마라. zero-noise exactness와 finite/SPD/determinism을 Gate B1 합격 기준으로 사용한다. bias convergence의 정량 acceptance는 후속 UNIT-ST validation에서 별도 잠근다.

## 10.4 property sweep

고정 seed를 사용하여 최소 다음을 별도 evidence log로 수행하라.

```text
10개 dataset seeds
각 seed당 최소 4 trajectories
same-seed regeneration
serialization round trip
sign-representation paired replay
split disjointness
```

모든 semantic hash와 replay invariant 결과를 요약하라.

---

# 11. Required test commands

구현 후 신규 Gate B1 tests:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_mekf_events.py \
  tests/test_unit_st_synthetic.py \
  tests/test_mekf_replay.py
```

Gate A regression:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_mekf_conventions.py \
  tests/test_mekf_core.py
```

legacy subset:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_basilisk_imu_generator.py \
  tests/test_basilisk_mrp_ekf.py \
  bench/tests/test_generator_contract_tg0.py \
  bench/tests/test_adcs_event_metrics.py
```

다음을 금지한다.

```text
skip/xfail로 실패 은폐
tolerance 완화
expected value 변경
pseudo-inverse/jitter/eigenvalue clipping 추가
repository-wide test 기대값 수정
```

---

# 12. Required documents

## 12.1 `P1A_EVENT_SCHEMA_CONTRACT.md`

최소 포함:

- 목적/입력 근거/결정 상태/TBD/다음 Gate
- 모든 field의 dtype/shape/unit/frame
- event ordering과 replay semantics
- zero-latency 제한
- validation policy
- serialization format
- semantic hash algorithm
- manifest identity
- truth/sensor/estimator boundary

## 12.2 `P1A_SYNTHETIC_UNIT_ST_CONTRACT.md`

최소 포함:

- analytic truth equation
- gyro/ST measurement equation
- representative config와 가정 지위
- seed stream separation
- sign representation policy
- trajectory ID/split policy
- 아직 구현하지 않은 Basilisk/latency/outage 항목

## 12.3 `P1A_GATE_B1_TEST_MATRIX.md`

각 test에 다음 열을 사용하라.

| Test ID | Contract | Test function | Input | Expected | Tolerance | Result | Evidence |

## 12.4 `P1A_GATE_B1_VALIDATION_REPORT.md`

최소 포함:

- generated/modified paths
- exact commands
- pre/post Gate A and legacy results
- new B1 test results
- semantic hash examples
- seed-isolation evidence
- serialization/replay equivalence
- split disjointness
- property sweep
- dirty-tree integrity
- blocking issue
- Gate B1 decision

---

# 13. Dirty-tree final integrity

실행 종료 시 다음을 수행하라.

1. 시작 dirty path/status/content fingerprint와 종료 상태 비교
2. exact allowlist 변경만 agent-owned로 분류
3. `artifacts/benchmark_write_control/**` 신규 파일은 concurrent external ledger로 분리
4. allowlist 밖 source/config/test 변경은 실패
5. staged diff가 없어야 함
6. allowlist files만 개별 whitespace check
7. agent-only patch/stat/changed-path list 생성

최소 evidence 파일:

```text
experiments/phase1a/agent_logs/03A_agent_only.patch
experiments/phase1a/agent_logs/03A_agent_only_stat.txt
experiments/phase1a/agent_logs/03A_changed_paths.txt
experiments/phase1a/agent_logs/03A_status_after.txt
experiments/phase1a/agent_logs/03A_dirty_integrity_check.tsv
experiments/phase1a/agent_logs/03A_concurrent_change_ledger.tsv
experiments/phase1a/agent_logs/03A_allowlist_whitespace.txt
experiments/phase1a/agent_logs/03A_baseline_gate_a_before.txt
experiments/phase1a/agent_logs/03A_baseline_legacy_before.txt
experiments/phase1a/agent_logs/03A_gate_b1_tests.txt
experiments/phase1a/agent_logs/03A_gate_a_after.txt
experiments/phase1a/agent_logs/03A_legacy_after.txt
experiments/phase1a/agent_logs/03A_property_sweep.txt
```

기존 dirty 파일을 자동 복원하지 마라. 예상하지 않은 변경이 있으면 경로와 before/after hash를 보고하고 다음으로 종료하라.

```text
BLOCKED_UNINTENDED_CHANGE
```

---

# 14. Gate B1 합격 조건

다음이 모두 충족되어야 한다.

```text
Gate A baseline before: PASS
legacy baseline before: PASS
Typed schema validation: PASS
Zero-latency/order: PASS
Serialization round trip: PASS
Canonical semantic hash: PASS
Seed isolation: PASS
Trajectory-level split: PASS
Synthetic UNIT-ST determinism: PASS
Direct-core replay equivalence: PASS
Truth-leakage boundary: PASS
Quaternion norm/P-SPD replay safety: PASS
Gate A regression after: PASS
legacy regression after: PASS
Dirty-tree integrity: PASS
Unexpected source changes: 0
```

최종 판정 형식:

```text
Status: PASS_GATE_B1 또는 BLOCKED_*/FAIL_GATE_B1

Schema: PASS/FAIL
Zero latency/order: PASS/FAIL
Serialization/hash: PASS/FAIL
Seed isolation: PASS/FAIL
Trajectory split: PASS/FAIL
Synthetic UNIT-ST: PASS/FAIL
Direct replay: PASS/FAIL
Truth boundary: PASS/FAIL
Numerical/replay safety: PASS/FAIL
Gate A regression: PASS/FAIL
Legacy regression: PASS/FAIL
Dirty-tree integrity: PASS/FAIL
Gate B1: GO/STOP
```

Gate B1이 GO여도 Gate B2를 시작하지 말고 종료하라.

---

# 15. 최종 응답에 반드시 포함할 내용

- status와 Gate B1 GO/STOP
- 생성/수정 파일
- schema field/dtype 요약
- event ordering 및 zero-latency 증거
- semantic hash와 round-trip 증거
- same-seed 및 seed-isolation 증거
- trajectory split 교집합 증거
- direct-core replay 결과
- truth leakage 검사
- Gate A/legacy pre/post test 결과
- dirty-tree integrity와 concurrent external ledger
- 아직 남은 Gate B2/C/D 항목
- commit/push를 수행하지 않았다는 확인

Gate B2, Basilisk, metric, runner로 자동 진행하지 마라.
