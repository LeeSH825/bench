# AI-ADCS / KalmanNet 연구 행동 지침

> 목적: 앞으로의 구현, 실험, 문헌 검토, 결과 해석, 논문 작성에서 일관된 판단 기준으로 사용한다.  
> 적용 범위: 위성 자세 추정용 MEKF, KalmanNet 계열, Split-KalmanNet, context/SoW, ANN/SNN. FPGA 구현은 현재 범위에서 제외한다.  
> 사용법: 새로운 아이디어나 실험을 추가하기 전에 본 지침의 관련 규칙과 gate를 먼저 확인한다.

---

## 1. 최상위 원칙

### RULE 1 — 연구의 주 기여는 자세 추정기로 고정한다

- 기본 표현: **자세제어 시스템에 사용되는 자세 추정기**
- estimator 성능만 평가한 상태에서 “자세제어 성능을 개선했다”고 쓰지 않는다.
- 자세제어 주장을 하려면 동일 controller와 actuator 조건에서 closed-loop 결과를 제시한다.

### RULE 2 — 확인된 사실, 가설, 해석을 항상 구분한다

문서와 결과에는 다음 표지를 사용한다.

- **[확인]** 실제 데이터 또는 실험으로 확인
- **[문헌]** 논문이나 공식 자료가 직접 지지
- **[분석]** 수식·구조로부터 도출한 논리적 해석
- **[가설]** 향후 실험이 필요한 주장
- **[결정]** 연구자가 선택해 고정한 설계
- **[보류]** 현재 근거 부족으로 주력 방향에서 제외

확인되지 않은 내용을 완료된 결과처럼 쓰지 않는다.

### RULE 3 — AI를 넣기 전에 상태공간모델을 먼저 고정한다

다음이 확정되지 않은 상태에서는 neural architecture를 확장하지 않는다.

- nominal state와 error state
- process/measurement equation
- frame과 quaternion convention
- 센서별 역할
- truth model과 estimator model의 차이
- baseline \(Q_0,R_0\)
- sampling과 latency

### RULE 4 — 동일한 문제를 푸는 모델끼리만 직접 비교한다

- proposed가 MEKF라면 primary baseline도 MEKF shell 위에 둔다.
- 기존 MRP-EKF 결과는 legacy baseline으로 따로 표시한다.
- 센서 입력, state dimension, model knowledge, trajectory가 다르면 직접 우열을 주장하지 않는다.

---

## 2. 연구 범위와 용어 규칙

### 2.1 사용 권장 용어

- 자세 추정기 / attitude estimator
- Kinematic MEKF
- local attitude error / tangent-space error
- propagation uncertainty
- measurement reliability
- structured context
- oracle context / estimated context
- latent context
- sensor regime change
- robust measurement update

### 2.2 사용에 주의할 용어

#### “자세제어 모델”

controller까지 포함하지 않으면 “자세제어용 자세 추정 모델”로 쓴다.

#### “SoW”

물리적으로 정의된 값이나 명시적 label이 있을 때 사용한다. end-to-end latent vector라면 “latent context”로 쓴다.

#### “process branch / measurement branch”

Split-KalmanNet의 branch를 편의상 그렇게 부를 수 있으나, 실제 \(Q\)와 \(R\)을 유일하게 학습한다고 단정하지 않는다.

#### “공분산”

대칭, 양의 정부호성, frame, calibration이 확인되지 않은 neural output을 공분산이라고 부르지 않는다. 필요하면 “covariance-like latent factor”라고 쓴다.

#### “저전력”과 “저지연”

SNN을 사용했다는 이유만으로 주장하지 않는다. operation/event count, latency 또는 실제 측정이 있어야 한다.

---

## 3. 모델링 지침

### RULE 5 — truth, sensor, estimator를 분리한다

모든 시뮬레이션은 다음 세 층으로 문서화한다.

```text
Truth spacecraft/environment
    → Sensor output model
    → Estimator model
```

각 파라미터는 다음 중 하나로 표시한다.

- truth-only
- estimator-known
- nominally known but mismatched
- context input으로 제공
- neural network가 추정
- 평가에만 사용

### RULE 6 — 1차 핵심 필터는 Kinematic MEKF를 우선한다

권장 1차 상태:

\[
\bar x=(q,b_g),
\qquad
\delta x=[\delta\theta^\top,\delta b_g^\top]^\top.
\]

Dynamic MEKF \((q,\omega,b_g)\)는 torque/inertia mismatch가 핵심 연구문제로 확정될 때 후속 확장한다.

### RULE 7 — MEKF convention을 코드와 문서에 한 번만 정의한다

반드시 고정할 항목:

- scalar-first 또는 scalar-last quaternion
- active/passive rotation
- \(I\to B\) 또는 \(B\to I\)
- left/right multiplicative error
- body/inertial angular rate
- error injection 순서
- reset Jacobian
- quaternion sign handling
- sensor frame to body frame transform

모든 module은 이 정의를 가져다 쓰며 개별적으로 재정의하지 않는다.

### RULE 8 — 센서의 역할은 필터 수식 기준으로 정한다

Kinematic MEKF 권장 역할:

```text
Gyro          → propagation input
Magnetometer  → vector measurement update
Sun sensor    → vector measurement update
Star tracker  → low-rate attitude measurement update
Temperature   → calibration/context input
Control telemetry → known exogenous context input
```

센서라는 이유만으로 모두 measurement branch에 넣지 않는다.

---

## 4. 센서 오차 처리 지침

### RULE 9 — 오차를 세 종류로 나눈다

#### A. Mean/model error

- bias
- temperature-dependent bias
- scale factor
- axis misalignment
- hard/soft iron
- mounting misalignment

처리: explicit state, calibration, 또는 명시적 correction model

#### B. Stochastic uncertainty

- white noise
- bias random walk
- time-varying noise floor
- model uncertainty

처리: \(Q_t\), \(R_t\), gain adaptation

#### C. Gross error / invalidity

- outlier
- saturation
- sensor glitch
- blinding
- eclipse/FOV invalid

처리: reliability gate, robust likelihood, update skip/down-weight

한 종류의 오차를 다른 종류의 메커니즘으로 억지로 해결하지 않는다.

### RULE 10 — gyro bias와 bias uncertainty를 구분한다

- \(b_g\): 상태 추정 대상
- \(Q_b\): bias가 얼마나 빠르게 변하는지
- temperature-bias mean model: calibration 대상
- calibration residual uncertainty: context 대상 가능

### RULE 11 — accelerometer는 지상과 같은 중력 자세센서로 가정하지 않는다

궤도상 accelerometer를 사용할 경우 목적을 명시한다.

- specific force 측정
- maneuver/vibration proxy
- context auxiliary feature
- 특수한 thrust 구간의 관측

목적이 없으면 1차 core attitude update에서는 제외한다.

---

## 5. Split-KalmanNet 및 covariance 지침

### RULE 12 — Split branch를 실제 \(Q/R\)로 자동 해석하지 않는다

\[
K=G_1H^\top G_2
\]

에서 \(G_1,G_2\)는 최종 state loss만으로 유일하게 식별되지 않는다. 또한 innovation covariance는

\[
S=HP^-H^\top+R
\]

이므로 innovation-side factor에 prior와 measurement 정보가 섞인다.

허용되는 해석:

- latent prior-side gain factor
- latent innovation-side gain factor

추가 제약·supervision 없이 금지되는 해석:

- 실제 \(P^-\)
- 실제 \(Q\)
- 실제 \(R\)
- 실제 \(S^{-1}\)

### RULE 13 — NEES/NIS는 explicit SPD covariance가 있을 때만 핵심 지표로 사용한다

필요 조건:

- 정의된 tangent frame
- symmetric positive definite \(P,S\)
- correct reset/transport
- dimension에 맞는 자유도

implicit neural factor에 대해 무리하게 consistency를 주장하지 않는다.

### RULE 14 — 직접 gain 방식과 structured covariance 방식을 모두 비교한다

- Direct Split-KNet: 원 구조 baseline
- Structured adaptive MEKF: neural context → \(Q_t,R_t\), MEKF → \(K_t\)

최종 구조는 성능뿐 아니라 다음을 함께 보고 결정한다.

- consistency
- long-horizon stability
- interpretation
- failure behavior
- parameter count

---

## 6. SoW/context 지침

### RULE 15 — context의 의미를 구현 전에 명시한다

각 context dimension마다 다음을 기록한다.

| 항목 | 질문 |
|---|---|
| 이름 | 무엇을 나타내는가 |
| 단위/범위 | physical scale인가 probability인가 |
| time scale | fast/slow 중 어디인가 |
| target | Q, R, gate, modulation 중 무엇인가 |
| label | simulator에서 ground truth가 존재하는가 |
| onboard availability | 실제 입력만으로 추정 가능한가 |

### RULE 16 — oracle context를 먼저 검증한다

순서:

1. fixed baseline
2. oracle scalar context
3. oracle vector context
4. oracle fast/slow context
5. learned ANN context
6. learned SNN context

oracle에서도 개선되지 않는 context 구조는 learned estimator로 넘어가지 않는다.

### RULE 17 — physical context와 latent context를 구분한다

- physical context: 실제 simulator parameter 또는 정의된 reliability와 대응
- latent context: end-to-end 성능을 위한 내부 표현

latent context의 dimension을 “gyro bias level”이나 “vibration intensity”라고 해석하려면 별도 intervention evidence가 필요하다.

### RULE 18 — innovation만으로 원인 분리가 된다고 가정하지 않는다

innovation 증가는 다음 원인에서 모두 발생할 수 있다.

- propagation error
- bias estimation error
- 초기 자세 오차
- measurement noise
- magnetic disturbance
- model error
- frame misalignment

따라서 process/measurement context를 나누려면 branch별 보조 입력과 controlled intervention이 필요하다.

### RULE 19 — 알려진 telemetry를 숨은 SoW로 다시 추론하지 않는다

실제 onboard에서 사용할 수 있는 다음 정보는 직접 입력 후보로 둔다.

- commanded torque
- reaction-wheel speed
- magnetorquer on/off/current
- sensor validity flag
- star tracker quality flag
- temperature

알려진 원인을 네트워크가 residual만 보고 추측하게 만들지 않는다.

---

## 7. fast/slow 지침

### RULE 20 — 시간척도는 이름이 아니라 구조와 데이터로 정의한다

- slow branch에는 긴 time constant 또는 낮은 update bandwidth
- fast branch에는 짧은 time constant와 event responsiveness
- 센서 실측/모델의 Allan deviation, thermal response, vibration interval을 기준으로 범위를 정한다.

### RULE 21 — branch collapse를 항상 검사한다

필수 점검:

- branch activation variance
- branch 간 correlation
- slow-only/fast-only ablation
- branch permutation 가능성
- one-branch-zero 실험
- combined event에서의 역할 변화

### RULE 22 — slow와 fast를 분리했다는 주장은 intervention으로 검증한다

최소 시나리오:

1. slow drift only
2. fast event only
3. slow + fast simultaneous
4. 같은 innovation RMS를 만드는 서로 다른 원인
5. unseen event timing

---

## 8. SNN 지침

### RULE 23 — ANN에서 유효하지 않은 구조를 SNN으로 구현하지 않는다

SNN 단계 진입 조건:

- oracle context가 유효
- learned ANN context가 baseline을 개선
- fast/slow 또는 branch-specific 요소의 ablation 효과가 확인

### RULE 24 — SNN의 기능을 한정한다

우선 후보:

- fast change detector
- reliability gate
- sparse context updater
- outlier probability estimator
- \(Q/R\) scale estimator

전체 gain network를 spiking으로 바꾸는 것은 후순위다.

### RULE 25 — 공정한 ANN/SNN 비교를 한다

공통 조건:

- 동일 입력 history
- 동일 target/context dimension
- 유사 parameter budget
- 동일 train/test trajectory
- 동일 latency 정의

평가:

- attitude/bias error
- detection latency
- false alarm
- spike rate
- active update ratio
- event/operation count

---

## 9. 데이터와 실험 지침

### RULE 26 — trajectory 단위로 분할한다

window 단위 무작위 분할을 금지한다.

분리 대상:

- initial attitude/rate
- orbit phase
- maneuver schedule
- temperature profile
- disturbance timing
- sensor outage
- noise realization

### RULE 27 — 동일한 pre-generated sensor realization을 사용한다

모델별 실행 중 센서 noise를 새로 생성하지 않는다. 비교 모델은 동일한 truth와 measurement를 사용한다.

### RULE 28 — IID와 OOD를 분리해 보고한다

- IID noise realization
- unseen magnitude
- unseen transition timing
- unseen combination
- unseen orbit/attitude condition
- simulation-to-real

OOD라는 한 단어로 모두 묶지 않는다.

### RULE 29 — matched stationary 성능을 희생하지 않는지 확인한다

adaptive model은 변화 조건에서만 좋아지고 정상 조건에서 불필요한 진동이나 과적응을 만들 수 있다. stationary matched scenario는 항상 포함한다.

### RULE 30 — long-horizon stability를 짧은 window 성능과 별도로 평가한다

- quaternion norm
- bias drift
- covariance boundedness
- hidden-state saturation
- divergence rate

---

## 10. 비교군 지침

### 필수 계층

1. tuned MEKF
2. mismatched fixed-\(Q/R\) MEKF
3. classical adaptive covariance MEKF
4. robust/gated MEKF
5. MEKF-KalmanNet
6. MEKF-Split-KalmanNet
7. oracle-context adaptive model
8. learned ANN context model
9. learned SNN context model

모든 모델을 한 번에 구현할 필요는 없지만, 논문의 최종 비교 논리는 위 계층을 충족해야 한다.

### 적응 budget 공개

- weight update 여부
- labeled state 사용 여부
- oracle parameter 사용 여부
- online history 길이
- support samples
- parameter count

이를 숨긴 채 성능 숫자만 비교하지 않는다.

---

## 11. 지표 지침

### 핵심 추정 지표

- attitude geodesic RMSE
- peak/percentile attitude error
- gyro bias RMSE
- convergence time
- divergence rate

### 적응 지표

- event detection latency
- transient peak error
- recovery time
- false adaptation rate
- context estimation error

### 일관성 지표

- NIS/NEES: explicit covariance가 있을 때
- innovation whiteness
- empirical coverage

### SNN 지표

- spike rate
- active neuron/update ratio
- event count
- accuracy-latency-sparsity trade-off

quaternion component MSE만으로 최종 결론을 내리지 않는다.

---

## 12. 금지하거나 보류할 기본 전제

다음을 기본 가정으로 다시 사용하지 않는다.

1. 실제 데이터에 불규칙 결측이 많다.
2. pseudo-measurement를 넣으면 새로운 정보가 생긴다.
3. measurement MSE가 줄면 attitude MSE도 자동으로 줄어든다.
4. 저가 IMU 문제는 단일 Gaussian variance 차이다.
5. Split branch가 실제 process/measurement covariance를 자동 분리한다.
6. learned context는 자동으로 물리적 SoW다.
7. fast/slow 네트워크는 이름대로 역할을 나눈다.
8. SNN은 자동으로 저전력·저지연이다.
9. proposed MEKF와 기존 direct-state EKF를 같은 조건으로 간주할 수 있다.
10. star tracker measurement를 simulation ground truth로 간주한다.

---

## 13. 실험 제안서 작성 템플릿

새 실험은 아래 양식을 채운 후 시작한다.

```markdown
# Experiment ID / Name

## 1. 검증 가설
한 문장으로 작성.

## 2. 변경 변수
한 번에 바꾸는 핵심 변수.

## 3. 고정 조건
truth trajectory, sensors, initial condition, model knowledge.

## 4. 비교 모델
동일 shell인지, parameter/adaptation budget은 무엇인지.

## 5. 입력 가능 정보
onboard available / oracle only를 구분.

## 6. 평가 지표
accuracy, consistency, adaptation, long horizon.

## 7. 성공 기준
수치 또는 명확한 방향성.

## 8. 실패 시 해석
어떤 가설이 기각되고 무엇은 남는가.

## 9. 재현 정보
seed, config, commit, dataset ID.
```

---

## 14. 의사결정 절차

새로운 architecture나 feature를 추가할 때 다음 순서로 판단한다.

1. **문제와 직접 연결되는가?**
2. **고전적 또는 더 단순한 방법으로 해결 가능한가?**
3. **oracle 입력에서 유효한가?**
4. **실제 onboard에서 관측 가능한 입력으로 추정 가능한가?**
5. **ablation으로 기능을 분리할 수 있는가?**
6. **동일 조건의 baseline보다 의미 있는 개선이 있는가?**
7. **정상 조건과 장시간 안정성을 해치지 않는가?**
8. **논문에서 과장 없이 설명 가능한가?**

한 단계라도 “아니오”이면 구현 규모를 줄이거나 보류한다.

---

## 15. 단계별 중단 기준

### context 구조 중단

- oracle context에서도 개선 없음
- scalar/vector/fast-slow 차이가 없음
- context가 입력 timing만 외움

### Split interpretation 중단

- branch intervention과 출력의 대응이 재현되지 않음
- factor가 불안정하거나 consistency 해석 불가

### SNN 중단

- ANN보다 accuracy/latency가 나쁨
- spike가 조밀해 event-driven 이점 없음
- 동일 성능에서 모델/연산 이점 없음

### 실험 설계 중단

- train/test trajectory leakage 발견
- baseline의 MEKF convention이 다름
- 모델별 sensor realization이 다름

중단은 연구 실패가 아니라 더 약한 주장을 제거하는 과정으로 기록한다.

---

## 16. fallback 우선순위

1. **Structured adaptive MEKF** — explicit \(Q/R\), reliability gate
2. **ANN dual-timescale context MEKF**
3. **Context-modulated Split-KalmanNet**
4. **SNN fast-context hybrid**

SNN 또는 Split branch의 물리 해석이 실패해도 1~2단계가 성립하도록 연구를 설계한다.

---

## 17. 구현 전 체크리스트

- [ ] 연구 출력이 자세 추정인지 제어인지 명시했는가
- [ ] state와 error state를 확정했는가
- [ ] quaternion/frame convention 문서가 있는가
- [ ] 센서별 measurement equation이 있는가
- [ ] truth와 estimator mismatch 표가 있는가
- [ ] 오차를 mean/covariance/outlier로 분류했는가
- [ ] context dimension과 target을 정의했는가
- [ ] oracle experiment가 먼저 설계되어 있는가
- [ ] identical trajectory baseline이 준비되어 있는가
- [ ] success/stop criterion이 있는가

---

## 18. 결과 해석 전 체크리스트

- [ ] 개선이 MEKF 전환 때문인지 AI 때문인지 분리했는가
- [ ] parameter count 증가 효과를 배제했는가
- [ ] train timing memorization 가능성을 점검했는가
- [ ] stationary matched 성능을 확인했는가
- [ ] long-horizon divergence를 확인했는가
- [ ] physical context와 latent context를 구분했는가
- [ ] covariance라고 부를 수 있는 조건을 충족했는가
- [ ] oracle/estimated context를 구분해 보고했는가
- [ ] 실패 사례와 worst-case를 포함했는가

---

## 19. 논문 작성 전 체크리스트

- [ ] 문제 정의가 센서 모델과 일치하는가
- [ ] contribution이 단순 조합이 아니라 문제-구조 대응으로 설명되는가
- [ ] 자세추정과 자세제어 표현이 구분되는가
- [ ] Split branch의 물리적 의미를 과장하지 않았는가
- [ ] SNN 저전력 주장을 근거 없이 쓰지 않았는가
- [ ] 실제 데이터에서 확인되지 않은 결측 문제를 다시 들고 오지 않았는가
- [ ] 모든 핵심 주장에 대응하는 ablation이 있는가
- [ ] baseline의 정보·학습 budget을 공개했는가
- [ ] OOD 유형을 구분했는가
- [ ] 연구 한계와 fallback을 명시했는가

---

## 20. 현재 권장 기본 결정값

Phase 0에서 별도 근거로 변경하기 전까지 다음을 기본안으로 사용한다.

```text
주 기여          = ADCS용 자세 추정기
기본 필터        = Kinematic MEKF
nominal state    = [q, b_g]
error state      = [δθ, δb_g]
단위 검증 센서   = gyro + star tracker
주 연구 센서     = gyro + magnetometer + sun sensor + low-rate star tracker
stress 센서      = gyro + magnetometer
adaptation target= gyro-side Q scale + external-sensor R/reliability
Split-KNet       = direct-gain backbone baseline
proposed 후보    = structured context → Q/R/gate → MEKF
context 순서     = oracle → ANN → SNN
SNN 1차 역할     = fast change/reliability estimator
데이터 분할      = trajectory-level
```

이 기본안을 변경할 때는 변경 이유, 기대 이점, 추가 실험 비용, 비교 공정성 영향을 decision log에 기록한다.
