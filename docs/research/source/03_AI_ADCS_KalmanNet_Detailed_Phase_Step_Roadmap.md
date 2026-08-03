# AI-ADCS / KalmanNet 상세 연구 계획 — Phase–Step Roadmap

> 목적: 연구를 문제 정의부터 시뮬레이션, MEKF, KalmanNet baseline, oracle/learned context, SNN, 실제 데이터, closed-loop 검증까지 단계적으로 진행하기 위한 실행 계획이다.  
> 원칙: 앞 단계의 핵심 가설이 통과하지 않으면 다음 단계의 복잡한 모델로 넘어가지 않는다.  
> 범위: FPGA 구현은 제외한다. SNN은 알고리즘적 기능과 sparsity를 평가하되 hardware 효과는 별도 후속 과제로 둔다.

---

## 0. 전체 진행 구조

```text
Phase 0  문제·모델·센서·평가 명세 확정
    ↓
Phase 1  Truth/Sensor Simulator + Classical MEKF 검증
    ↓
Phase 2  동일 MEKF shell의 Neural Baseline 구축
    ↓
Phase 3  문제 존재성·관측성·식별성 검증
    ↓
Phase 4  Oracle Structured Context 검증
    ↓
Phase 5  Classical Adaptive/Robust Baseline 비교
    ↓
Phase 6  Learned ANN Context Estimator
    ↓
Phase 7  Split/Structured Neural MEKF 확정 및 Ablation
    ↓
Phase 8  SNN Fast-Context 전환
    ↓
Phase 9  실제 센서 데이터 검증
    ↓
Phase 10 Closed-loop ADCS 검증 및 논문 패키지
```

각 Phase는 다음 네 가지를 갖는다.

- **목표**: 무엇을 확정하거나 검증하는가
- **Steps**: 실제 수행 항목
- **산출물**: 코드·문서·데이터·표
- **Gate**: 다음 Phase로 넘어갈 조건

---

# Phase 0. 연구 문제와 모델 명세 확정

## Phase 0 목표

AI architecture를 구현하기 전에 연구가 풀 문제, 위성·환경 truth model, 센서 모델, MEKF 수학, context 의미, 데이터 분할, 비교 기준을 고정한다.

Phase 0가 완료되면 다음 질문에 한 문장 또는 수식으로 답할 수 있어야 한다.

1. 무엇을 추정하는가?
2. 어떤 센서가 어떤 역할로 들어가는가?
3. truth에는 어떤 오차가 있고 estimator는 무엇을 모르는가?
4. neural module은 필터의 어디를 바꾸는가?
5. SoW/context는 무엇을 의미하는가?
6. 어떤 결과가 나와야 가설이 지지되는가?

---

## P0-S0. 기존 사실과 제약 등록

### 수행

- 기존 인수인계 문서의 내용을 다음으로 분류한다.
  - 확인된 결과
  - 문헌 근거
  - 미검증 가설
  - 기각/보류 가정
- 다음 사항을 고정 제약으로 등록한다.
  - 불규칙 IMU 결측은 현재 핵심 문제가 아님
  - 단순 measurement enhancement는 뚜렷한 최종 개선이 확인되지 않음
  - Split-KalmanNet, Adaptive context, SNN이라는 큰 방향은 유지하되 내부 구조는 수정 가능
  - FPGA 구현은 현재 연구 계획에서 제외

### 산출물

- `P0_00_EVIDENCE_REGISTER.md`
- `P0_00_DEPRECATED_ASSUMPTIONS.md`

### Gate

모든 팀원이 “확인된 것”과 “아이디어”를 같은 목록에서 구분할 수 있어야 한다.

---

## P0-S1. 연구 범위와 기여 문장 확정

### 결정해야 할 것

1. 자세 추정만 주 기여로 둘지
2. closed-loop 자세제어를 최종 검증으로 포함할지
3. 저가 gyro만 적응할지, 외부센서 reliability도 함께 적응할지
4. main contribution에서 SNN을 필수로 둘지 확장으로 둘지

### 권장 결정

```text
주 기여       = 자세제어 시스템에 사용되는 적응형 자세 추정기
최종 검증     = 동일 controller를 이용한 closed-loop 비교
적응 대상     = gyro propagation uncertainty + 외부센서 measurement reliability
SNN의 지위    = ANN 구조 검증 후의 기능적 확장
```

### 작성할 문장

#### 문제 문장

> 시간에 따라 변하는 저가 gyro의 propagation uncertainty와 외부 자세센서의 measurement reliability 때문에 고정 공분산 또는 고정 gain mapping 기반 자세 추정기의 성능과 일관성이 저하된다.

#### 목표 문장

> MEKF의 tangent-space 구조를 유지하면서 structured fast/slow context를 추정해 \(Q/R\) 또는 gain/reliability를 적응시키는 neural attitude estimator를 개발한다.

### 산출물

- `P0_01_RESEARCH_SCOPE_AND_CLAIMS.md`
- contribution 1문장, problem 1문장, non-goal 목록

### Gate

- estimator와 controller 범위가 구분됨
- SNN이 없어도 남는 최소 기여가 정의됨

---

## P0-S2. 위성·궤도·환경 truth model 확정

### 결정 항목

#### Spacecraft

- 대표 6U 모델 또는 실제 대상 위성 모델
- 질량과 관성모멘트 \(J\)
- body frame 정의
- 센서/actuator mounting frame
- product of inertia 포함 여부

#### Orbit/environment

- LEO 궤도 파라미터
- 궤도 state를 estimator가 알고 있는지
- 지구 자기장과 태양벡터 모델
- eclipse
- 외란 토크

#### Dynamics

- truth는 rigid-body dynamics 사용
- estimator는 1차로 gyro-driven kinematic propagation 사용
- commanded maneuver와 disturbance를 구분

### 권장 1차 설정

- truth: rigid-body spacecraft dynamics
- estimator: Kinematic MEKF
- orbit state: 알려져 있다고 가정
- eclipse: 포함
- commanded torque/MTQ state: onboard-known telemetry로 제공 가능
- position estimation 문제: 제외

### 산출물

- `P0_02_TRUTH_MODEL_SPEC.md`
- frame diagram
- truth state list
- truth-only parameter table

### Gate

동일한 truth trajectory에서 모든 센서 출력을 재생성할 수 있는 명세가 완성됨.

---

## P0-S3. 센서 구성과 역할 확정

### 권장 센서 시나리오

| Scenario | 구성 | 목적 |
|---|---|---|
| S0 | gyro + star tracker | MEKF/바이어스 단위 검증 |
| S1 | gyro + mag + sun + low-rate ST | 주 연구 시나리오 |
| S2 | gyro + mag | eclipse/ST outage stress |
| S3 | gyro + sun 또는 gyro + ST outage variants | 센서별 reliability ablation |

### 센서별 결정 표

#### Gyro/IMU

- gyro만 core에 사용할지
- temperature channel 사용 여부
- accelerometer의 역할
- 실제 MTi-2 계열 실측/데이터시트 파라미터를 사용할지

권장:

- gyro + temperature 사용
- accelerometer는 core attitude update에서 제외
- accelerometer는 vibration/context proxy 후보로만 보류

#### Magnetometer

- 3D vector 또는 normalized direction
- hard/soft iron 포함 여부
- MTQ interference
- validity/saturation

#### Sun sensor

- abstract sun-vector output부터 시작
- eclipse/FOV/validity flag 포함
- 추후 실제 CSS photodiode model 확장

#### Star tracker

- truth가 아니라 noisy low-rate attitude measurement
- small-angle error model
- update rate, latency, outage, quality flag

### 산출물

- `P0_03_SENSOR_SUITE_AND_ROLES.md`
- `P0_03_SENSOR_INTERFACE_TABLE.csv` 또는 `.md`

### Gate

각 센서에 대해 measurement equation, rate, validity, estimator role이 정해짐.

---

## P0-S4. 센서 오차 모델 계층화

### Tier 0 — 정상 baseline

- gyro white noise
- gyro bias random walk
- mag/sun/ST Gaussian noise
- fixed alignment
- fixed rates

### Tier 1 — 핵심 연구 regime

- gradual gyro bias/noise drift
- temperature-dependent gyro bias residual
- gyro vibration/noise burst
- magnetometer interference
- star tracker outage/reliability degradation
- slow + fast simultaneous event

### Tier 2 — OOD/고난도

- scale factor
- axis misalignment
- heavy-tailed outlier
- saturation
- unseen transition timing/magnitude
- unseen event combination

### 분류 규칙

각 오차는 다음으로 지정한다.

- explicit state/calibration
- \(Q_t\) 변화
- \(R_t\) 변화
- reliability/gate 변화
- truth-only mismatch

### 산출물

- `P0_04_SENSOR_ERROR_CATALOG.md`
- error-to-filter-target matrix
- parameter source column: 실측/데이터시트/문헌/가정

### Gate

“이 오차를 왜 Q/R/gate/state 중 해당 위치에 넣는가”를 설명할 수 있음.

---

## P0-S5. MEKF 수학 계약 확정

### 권장 상태

\[
\bar{x}=(q,b_g),
\qquad
\delta x=[\delta\theta^\top,\delta b_g^\top]^\top.
\]

### 결정 항목

- quaternion ordering
- frame convention
- active/passive rotation
- left/right multiplicative error
- gyro propagation equation
- bias process model
- mag/sun vector measurement function
- star tracker attitude residual
- error injection
- covariance reset
- sensor update ordering
- asynchronous update handling

### Loss/metric 계약

- attitude loss: SO(3) geodesic/log-map
- bias loss
- NIS/NEES 사용 조건
- quaternion sign treatment

### 단위 테스트 설계

- zero motion
- constant rate
- known bias
- small-angle update
- large initial attitude
- quaternion sign flip invariance
- update 후 reset consistency

### 산출물

- `P0_05_MEKF_MATH_CONTRACT.md`
- notation table
- frame/convention test vectors

### Gate

독립 구현 두 개가 동일 test vector에서 같은 propagation/update 결과를 내야 한다.

---

## P0-S6. Neural insertion point와 backbone 정책 결정

### 후보 A — Direct Split-KalmanNet

\[
K_t=G_{1,t}H_t^\top G_{2,t}
\]

역할:

- 기존 backbone 재현
- direct gain learning baseline
- branch는 latent factor로 해석

### 후보 B — Structured adaptive MEKF

```text
context → Q_t, R_t, gate → MEKF covariance recursion → K_t
```

역할:

- 해석성과 SPD consistency
- oracle context 실험
- proposed 또는 fallback

### 권장 정책

두 경로를 모두 유지하되 최종 proposed 결정은 Phase 4~7 결과로 미룬다.

### 산출물

- `P0_06_NEURAL_INSERTION_OPTIONS.md`
- interface specification
- baseline/proposed status table

### Gate

각 후보가 무엇을 입력받고 무엇을 출력하며 어떤 claim이 가능한지 명확함.

---

## P0-S7. SoW/context 계약 확정

### 1차 physical oracle 후보

#### Slow/process

\[
z_s=[\log\alpha_{bRW},\log\alpha_{thermal},\log\alpha_{noise-floor}]
\]

#### Fast/process/measurement

\[
z_f=[\log\alpha_{gyro-burst},\log\alpha_{mag},\log\alpha_{sun},\log\alpha_{ST},p_{outlier}]
\]

실제 최종 dimension은 최소화한다. 서로 독립적으로 변화시키고 필터에서 다른 의미를 갖는 항목만 남긴다.

### 결정 항목

- scalar/vector
- shared/branch-specific
- physical/latent
- supervised/end-to-end
- update rate
- range/normalization
- modulation target: \(Q,R,gate,hidden state,gain\)

### 입력 후보 구분

#### onboard available

- raw gyro/derivative
- temperature
- mag norm and innovation
- sun/ST residual and quality
- control telemetry
- previous correction

#### oracle only

- true noise multiplier
- true bias
- event label
- true \(Q/R\)

### 산출물

- `P0_07_CONTEXT_CONTRACT.md`
- context dimension table
- oracle vs deployable feature table

### Gate

모든 context dimension의 물리 의미, target, label availability가 정해짐.

---

## P0-S8. 데이터셋·분할·시나리오 계약

### trajectory 변수

- initial attitude/rate
- orbit phase
- maneuver timing
- temperature profile
- vibration/interference timing
- sensor outage
- noise realization

### split

- train/validation/test trajectory 분리
- IID test와 OOD test 분리
- normalization은 train에서만 계산
- identical pre-generated measurements 사용

### OOD 분류

- magnitude OOD
- timing OOD
- combination OOD
- initial-condition OOD
- orbit OOD
- simulation-to-real

### 산출물

- `P0_08_DATASET_PROTOCOL.md`
- scenario ID naming rule
- seed policy
- split manifest schema

### Gate

동일 trajectory window가 split 사이에 중복되지 않도록 자동 검사할 수 있음.

---

## P0-S9. 지표·성공 기준·중단 기준 확정

### 기본 지표

- attitude geodesic RMSE/peak/percentile
- bias RMSE
- convergence/recovery time
- divergence rate
- NIS/NEES where valid
- innovation whiteness
- context estimation error
- SNN detection latency/spike rate

### Phase 0 종료 리뷰 질문

- oracle context가 무엇인지 명확한가?
- 센서 시나리오가 관측성 문제를 분리하는가?
- proposed와 baseline이 동일 shell을 사용하는가?
- 어떤 결과면 SNN을 중단할지 정했는가?

### 산출물

- `P0_09_METRICS_AND_GATES.md`
- `P0_DESIGN_REVIEW_CHECKLIST.md`

### Phase 0 Exit Gate

아래가 모두 있어야 Phase 1로 이동한다.

- [ ] scope/claim 문서
- [ ] truth model spec
- [ ] sensor spec
- [ ] error catalog
- [ ] MEKF math contract
- [ ] context contract
- [ ] dataset protocol
- [ ] metric/stop criteria

---

# Phase 1. Truth/Sensor Simulator와 Classical MEKF 검증

## 목표

AI 없이도 물리적으로 일관된 sensor-level attitude estimation benchmark를 만든다.

## P1-S1. Truth dynamics 구현·검증

- rigid-body truth propagation
- known torque cases
- quaternion norm and energy sanity check
- orbit/environment vector logging

### 산출물

- truth trajectory dataset
- analytic/simple-case validation report

## P1-S2. 센서별 unit model 구현

- gyro noise/bias
- magnetometer vector
- sun vector and eclipse
- star tracker small-angle noise, rate, latency, outage

### 검증

- noise statistics
- bias evolution
- frame transform
- timestamp alignment
- validity behavior

## P1-S3. Kinematic MEKF 구현

순서:

1. gyro + star tracker
2. gyro + mag + sun
3. asynchronous low-rate ST 추가
4. gyro + mag stress

## P1-S4. MEKF numerical tests

- small/large initial attitude
- constant bias convergence
- long trajectory
- update reset consistency
- covariance SPD
- sign invariance

## P1-S5. tuned/mismatched baseline 생성

- tuned \(Q/R\)
- under/overestimated \(Q\)
- under/overestimated \(R\)
- time-varying truth + fixed estimator

### Phase 1 산출물

- `MEKF_BASELINE_REPORT.md`
- validated simulation configs
- regression unit tests

### Phase 1 Exit Gate

- 정상 조건에서 MEKF가 안정적으로 수렴
- known bias와 sensor noise에서 예상 동작
- long-horizon divergence 없음
- sensor failure/validity 처리 정상

실패하면 neural model로 넘어가지 않는다.

---

# Phase 2. 동일 MEKF shell의 Neural Baseline 구축

## 목표

기존 KalmanNet 계열과 proposed model을 동일한 상태, 센서, trajectory, metric에서 비교할 수 있는 최소 공통 runner를 구축한다.

## P2-S1. 공통 model interface

```text
predict(state, gyro, dt)
update(prior, measurement, sensor_meta)
reset(sequence)
log_internal_state()
```

## P2-S2. baseline 구현 우선순위

1. classical MEKF
2. MEKF-KalmanNet
3. MEKF-Split-KalmanNet
4. oracle-context adaptive baseline

MAML/unsupervised/Flex 계열은 핵심 구조가 안정된 후 추가한다.

## P2-S3. MEKF-KNet feature 재정의

- quaternion subtraction 사용 금지
- tangent-space state difference
- innovation normalization
- update magnitude
- sensor-type aware input

## P2-S4. Split-KNet baseline

- original direct gain path
- latent branch logging
- no physical covariance claim
- alternating training 여부 실험 가능하도록 config화

## P2-S5. 공정 비교 검사

- 동일 measurement tensors
- 동일 sequence length
- 동일 train/val/test
- 동일 knowledge of \(f,h,H\)
- parameter count 기록

### Phase 2 산출물

- common runner
- baseline reproduction report
- model card per baseline

### Phase 2 Exit Gate

- 모든 모델이 동일 sensor-level dataset을 처리
- baseline 결과가 seed 반복에서 안정적
- proposed 없이도 비교 프레임이 완성

---

# Phase 3. 문제 존재성·관측성·식별성 검증

## 목표

neural adaptation이 필요한 문제가 실제로 존재하는지, context가 원인을 구분할 수 있는지 먼저 확인한다.

## P3-S1. Problem-existence test

시나리오:

- stationary matched
- stationary mismatched
- slow drift
- fast event
- combined slow + fast

확인:

- fixed MEKF와 stationary-trained neural filter의 error 증가
- recovery delay
- consistency degradation

문제가 약하면 adaptation architecture를 확장하지 않는다.

## P3-S2. Observability test

- gyro + ST
- gyro + mag + sun
- gyro + mag
- 다양한 orbit phase/initial attitude/motion

확인:

- gyro+mag의 조건별 observability
- network가 timing prior를 외우는지
- unseen initial/orbit 조건 성능

## P3-S3. Q/R 원인 분리 intervention

### A

gyro process uncertainty만 증가, measurement 고정

### B

measurement noise만 증가, gyro 고정

### C

gyro bias step vs magnetometer bias/outlier

### D

commanded maneuver vs unknown disturbance

### E

large initial error vs measurement degradation

가능하면 A/B가 비슷한 innovation RMS를 만들도록 조정한다.

## P3-S4. Split branch 진단

- 각 intervention에서 branch hidden/output 반응
- branch swap/scale ambiguity
- branch ablation
- physical interpretation 가능 여부

### Phase 3 산출물

- `PROBLEM_AND_IDENTIFIABILITY_REPORT.md`
- observability scenario matrix
- branch interpretation limits

### Phase 3 Exit Gate

- adaptation이 필요한 성능 저하가 존재
- main sensor scenario의 관측성이 설명 가능
- context 분리 가능 범위와 불가능 범위가 명시됨

---

# Phase 4. Oracle Structured Context 검증

## 목표

context estimator를 학습하기 전에 “올바른 context가 주어졌을 때 필터가 정말 좋아지는가”를 검증한다.

## P4-S1. Oracle scalar context

- 원 AKNet과 비교 가능한 process/measurement ratio 또는 단일 reliability

## P4-S2. Oracle vector context

- gyro process scale
- bias RW scale
- mag/sun/ST reliability
- outlier probability

## P4-S3. Fast/slow context

- slow only
- fast only
- dual simultaneous
- context update rate 변화

## P4-S4. modulation target 비교

1. shared gain modulation
2. branch-specific Split modulation
3. explicit \(Q/R\) scaling
4. reliability gate
5. hybrid \(Q/R+gate\)

## P4-S5. sensitivity test

- context noise
- delay
- bias
- quantization
- wrong context

### 핵심 판단

- oracle에서도 개선이 없으면 context 구조를 기각
- scalar와 vector 차이가 없으면 vector dimension 축소
- direct gain과 \(Q/R\) 차이가 작으면 해석성이 높은 구조 우선

### Phase 4 산출물

- `ORACLE_CONTEXT_ABLATION_REPORT.md`
- final context candidate shortlist

### Phase 4 Exit Gate

최소 하나의 structured context 방식이 fixed/classical baseline 대비 transition 또는 consistency에서 반복적으로 개선됨.

---

# Phase 5. Classical Adaptive/Robust Baseline 비교

## 목표

AI가 실제로 필요한지 검증하고, 고전적 adaptive filtering을 복잡하게 재구현하는 데 그치지 않도록 한다.

## P5-S1. adaptive covariance matching

- innovation covariance matching
- moving-window \(R\) estimation
- adaptive \(Q\) scale

## P5-S2. robust update

- residual gating
- Huber/robust weighting
- sensor validity based update skip

## P5-S3. change detector

- EMA statistics
- CUSUM 또는 간단한 change-point detector

## P5-S4. 비교

- steady-state error
- peak error
- recovery time
- false adaptation
- computational complexity

### Phase 5 산출물

- `CLASSICAL_ADAPTIVE_BASELINE_REPORT.md`

### Phase 5 Exit Gate

proposed context가 해결해야 할 gap이 명확함. classical method가 충분하면 neural contribution을 축소하거나 다른 강점이 필요하다.

---

# Phase 6. Learned ANN Context Estimator

## 목표

oracle-only 정보를 제거하고 실제 onboard에서 가능한 입력으로 context 또는 reliability를 추정한다.

## P6-S1. feature set 정의

### Process-side

- gyro increment
- angular-rate derivative
- temperature
- bias estimate
- control telemetry
- previous correction

### Measurement-side

- innovation
- normalized innovation energy
- mag norm mismatch
- sensor validity/quality
- residual history

## P6-S2. simple baseline부터 시작

1. linear/EMA estimator
2. small MLP
3. single GRU
4. dual-timescale ANN

## P6-S3. supervision 방식

- direct physical context supervision
- multi-task: context + attitude loss
- end-to-end latent context

physical/latent 결과를 구분한다.

## P6-S4. fast/slow 구조

- separated time constants
- slow smoothness regularization
- fast sparsity/event loss
- branch decorrelation

## P6-S5. OOD test

- unseen magnitude
- unseen timing
- unseen combination
- unseen initial/orbit condition

### Phase 6 산출물

- `ANN_CONTEXT_MODEL_REPORT.md`
- feature ablation table
- oracle gap analysis

### Phase 6 Exit Gate

- ANN context가 classical/latent baselines보다 의미 있는 개선
- oracle 성능과의 gap이 설명 가능
- trajectory memorization 증거 없음

---

# Phase 7. 최종 Neural MEKF 구조 확정 및 핵심 Ablation

## 목표

직접 gain Split-KNet과 structured \(Q/R\)-adaptive MEKF 중 최종 proposed 구조를 결정하고 논문 핵심 기여를 고정한다.

## P7-S1. 후보 비교

### Model A

MEKF-Split-KalmanNet direct gain

### Model B

context-modulated Split-KalmanNet

### Model C

structured context → \(Q/R\) → MEKF

### Model D

structured \(Q/R\) + robust gate hybrid

## P7-S2. 필수 ablation

1. KNet vs Split-KNet
2. no context vs scalar
3. scalar vs vector
4. single vs dual timescale
5. shared vs branch-specific
6. oracle vs learned
7. raw sensor vs residual vs telemetry
8. process-only vs measurement-only
9. fast removal
10. slow removal
11. direct gain vs explicit \(Q/R\)
12. context delay/noise

## P7-S3. 안정성·일관성

- long horizon
- covariance boundedness
- NIS/NEES where valid
- worst-case transition
- divergence rate

## P7-S4. final claim freeze

결과에 따라 claim 수준을 확정한다.

- physical context adaptation
- latent context modulation
- direct gain robustness
- covariance-consistent adaptive MEKF

### Phase 7 산출물

- `FINAL_ARCHITECTURE_DECISION.md`
- full ablation report
- paper-ready main tables

### Phase 7 Exit Gate

최종 proposed 구조가 단순한 조합이 아니라 각 구성요소의 필요성이 ablation으로 입증됨.

---

# Phase 8. SNN Fast-Context 전환

## 목표

ANN에서 검증된 fast context 기능을 SNN으로 옮기고, 정확도뿐 아니라 sparse/event-driven 특성을 평가한다.

## P8-S1. SNN 역할 선택

우선순위:

1. fast change detector
2. measurement reliability gate
3. fast \(Q/R\) scale estimator
4. 전체 context estimator

## P8-S2. encoding 비교

- delta/event encoding
- threshold crossing
- direct current input + membrane state
- rate coding은 비교 baseline으로 제한

## P8-S3. ANN/SNN 동일 조건 비교

- 동일 input history
- 유사 parameter budget
- 동일 output/context target
- 동일 dataset

## P8-S4. 평가

- attitude/bias accuracy
- event detection latency
- false alarm
- spike rate
- active update ratio
- event/operation count

## P8-S5. 중단 판단

SNN이 ANN/GRU보다 sparse하지 않거나 latency/accuracy에서 이점이 없으면 핵심 기여에서 제외한다.

### Phase 8 산출물

- `SNN_CONTEXT_EVALUATION.md`
- ANN/SNN trade-off curves

### Phase 8 Exit Gate

SNN의 기능적 장점이 최소 하나의 측정 가능한 지표로 확인됨.

---

# Phase 9. 실제 센서 데이터 검증

## 목표

시뮬레이션에서 학습한 구조가 실제 저가 센서의 bias, 온도, vibration, alignment 문제에서 유지되는지 확인한다.

## P9-S1. 실험 장비와 reference 정의

- low-cost IMU
- reference/high-grade gyro
- 별도 attitude ground truth: rate table, optical tracker, camera/encoder 등
- temperature logging
- time synchronization
- sensor-frame alignment

고가 gyro 적분값을 장시간 attitude truth로 단독 사용하지 않는다.

## P9-S2. 데이터 수집 시나리오

- static multi-temperature
- constant-rate rotation
- multi-axis maneuver
- slow thermal drift
- vibration/interference event
- repeated runs for turn-on variation

## P9-S3. calibration and characterization

- Allan deviation
- bias vs temperature
- scale/misalignment
- noise distribution
- residual correlation

## P9-S4. simulation-to-real protocol

- simulation pretraining
- zero/few-shot transfer
- limited calibration
- OOD real test

## P9-S5. 평가

- gyro/bias error
- attitude error
- context/reliability behavior
- false adaptation
- long-horizon drift

### Phase 9 산출물

- real dataset manifest
- `REAL_SENSOR_VALIDATION_REPORT.md`

### Phase 9 Exit Gate

적어도 일부 핵심 regime에서 proposed model의 개선 또는 한계가 재현 가능하게 확인됨.

---

# Phase 10. Closed-loop ADCS 검증 및 논문 패키지

## 목표

추정 성능 향상이 실제 제어 시스템에서 의미가 있는지 확인하고, 연구 주장과 증거를 완성한다.

## P10-S1. 동일 controller 연결

- 동일 control law
- 동일 actuator limits
- estimator만 변경

## P10-S2. 시나리오

- nadir/sun pointing
- slew maneuver
- sensor outage
- magnetic interference
- long coast

## P10-S3. 제어 지표

- pointing RMSE
- peak pointing error
- settling time
- control effort
- wheel saturation
- failure/divergence behavior

## P10-S4. 논문 구조

1. problem and sensor regimes
2. MEKF and context formulation
3. identifiability/observability limits
4. proposed architecture
5. benchmark and baselines
6. oracle → ANN → SNN evidence chain
7. real/closed-loop validation
8. limitations

## P10-S5. reproducibility package

- configs
- seeds
- split manifests
- model cards
- experiment registry
- metric definitions

### 최종 Exit Gate

모든 주요 claim이 대응하는 controlled experiment와 ablation을 갖는다.

---

# 11. 단계별 핵심 의사결정 표

| Decision | 결정 시점 | 권장 기본안 | 재검토 조건 |
|---|---|---|---|
| 추정 vs 제어 | P0-S1 | 추정 주 기여 | closed-loop를 직접 제안할 때 |
| Kinematic vs Dynamic MEKF | P0-S5 | Kinematic | torque-model mismatch가 핵심일 때 |
| 주 센서 구성 | P0-S3 | gyro+mag+sun+low-rate ST | mission sensor 제약이 다를 때 |
| Split direct gain vs explicit Q/R | P7 | 둘 다 비교 후 결정 | oracle/consistency 결과 |
| scalar vs vector context | P4 | 최소 vector 후보 | scalar와 성능 차이 없을 때 |
| fast/slow | P4/P6 | intervention 통과 시 유지 | 분리 효과가 없을 때 |
| physical vs latent context | P6 | physical 우선 | label이 불충분할 때 latent 병행 |
| SNN 적용 | P8 | fast context만 | ANN 구조가 유효하고 event sparsity가 있을 때 |

---

# 12. 가장 중요한 Gate 요약

## Gate A — MEKF가 먼저 정상이어야 한다

AI 이전에 sensor-level MEKF가 안정적으로 동작하지 않으면 모든 neural 결과의 원인을 해석할 수 없다.

## Gate B — 문제 자체가 있어야 한다

시간가변 조건에서도 tuned/adaptive classical filter가 충분하면 복잡한 context model의 필요성이 약하다.

## Gate C — oracle context가 유효해야 한다

올바른 context를 줘도 좋아지지 않으면 context estimator를 학습할 이유가 없다.

## Gate D — ANN이 유효해야 SNN으로 간다

SNN은 알고리즘적 문제를 해결하는 마법이 아니라 구현 방식이다.

## Gate E — 동일 shell과 동일 데이터가 유지되어야 한다

representation, sensor, trajectory가 다르면 개선 원인을 분리할 수 없다.

---

# 13. 즉시 착수용 최소 실행 묶음

Phase 0 전체가 끝나기 전에 코드를 크게 늘리지 말고, 다음 작은 묶음을 우선 수행한다.

## Immediate Package A — 결정 문서

1. 연구 범위 1페이지
2. sensor role matrix
3. MEKF convention 1페이지
4. context dimension 후보표
5. baseline/proposed 구분표

## Immediate Package B — 최소 수학 검증

1. \([q,b_g]\) Kinematic MEKF 수식 확정
2. gyro + ST unit scenario
3. mag/sun vector Jacobian test
4. quaternion injection/reset test

## Immediate Package C — 최소 problem test

1. fixed noise
2. gyro uncertainty step
3. mag reliability step
4. slow + fast simultaneous
5. identical innovation RMS A/B pair

이 세 묶음의 결과가 나온 뒤 neural architecture의 layer와 SNN neuron을 결정한다.

---

# 14. 연구 완료 기준

다음 조건을 모두 충족하면 연구가 완결된 것으로 본다.

- [ ] sensor-level MEKF benchmark가 검증됨
- [ ] time-varying uncertainty 문제의 존재가 확인됨
- [ ] oracle context의 이점이 입증됨
- [ ] classical adaptive baseline보다 neural context의 추가 가치가 있음
- [ ] fast/slow와 branch-specific 요소의 ablation이 있음
- [ ] Split branch의 해석 한계를 명시함
- [ ] learned context가 onboard-available feature만 사용함
- [ ] trajectory-level OOD에서 안정적임
- [ ] SNN을 주장할 경우 ANN 대비 기능적 이점이 있음
- [ ] 실제 센서 또는 강한 simulation-to-real 검증이 있음
- [ ] 자세제어 적용성을 주장할 경우 closed-loop 결과가 있음
