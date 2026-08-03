# AI-ADCS / KalmanNet 연구 평가서 — 사람용 설명본

> 작성 목적: 지금까지 진행한 연구를 처음 접하는 연구자도 전체 맥락, 강점, 허점, 개선 방향을 이해할 수 있도록 정리한다.  
> 검토 범위: 위성 자세 **추정**, KalmanNet·Split-KalmanNet·Adaptive KalmanNet, MEKF, SoW/context, SNN 적용 가능성. FPGA 구현은 본 평가에서 제외한다.  
> 기반 자료: `AI_ADCS_KalmanNet_Research_Handoff(1).md`에 기록된 확인 결과와 연구 가정, 그리고 그에 대한 구조적·수학적 검토.  
> 주의: 본 문서에서 **확인된 결과**, **논리적 분석**, **향후 검증이 필요한 가설**을 구분한다.

---

## 0. 핵심 결론

이 연구의 큰 방향은 타당하고 연구 가치도 있다. 특히 다음 문제는 실제적이며 KalmanNet 계열과도 잘 맞는다.

> 저가 IMU와 외부 자세센서의 오차 특성이 시간에 따라 변할 때, 고정된 잡음 공분산이나 고정된 gain mapping을 사용하는 자세 추정기는 성능과 일관성을 잃을 수 있다. 따라서 현재 센서·환경 상태를 나타내는 context를 추정하고, 그 context에 따라 필터의 신뢰도 배분을 적응시키는 구조가 필요하다.

그러나 현재 연구는 아직 하나의 완성된 알고리즘이라기보다 다음 가설들이 이어진 상태다.

```text
저가 센서·환경의 변화
    → slow/fast context 또는 SoW
    → Split-KalmanNet의 두 branch 조절
    → Kalman gain 변화
    → MEKF 자세 추정 성능 향상
    → 향후 SNN을 통한 효율적인 context 추정
```

이 연결고리 중 가장 취약한 부분은 다음 두 가지다.

1. **Split-KalmanNet의 두 branch가 실제 process uncertainty와 measurement uncertainty를 자동으로 분리한다고 해석한 점**
2. **residual과 센서 신호로부터 얻은 learned context가 실제 물리적 SoW라고 자동으로 간주한 점**

둘 다 학습만으로 보장되지 않는다. 따라서 연구를 강하게 만들려면, “neural network가 알아서 분리한다”는 전제에서 벗어나 다음 방향으로 바꾸는 것이 좋다.

> **MEKF의 기하학적 구조와 공분산 일관성을 유지하면서, gyro propagation uncertainty와 외부센서 measurement reliability를 명시적으로 정의하고, 이들이 서로 다른 시간척도로 변할 때 structured context로 추정·반영하는 적응형 neural MEKF**

Split-KalmanNet은 버릴 필요가 없다. 다만 **유력 backbone이자 비교 기준**으로 두고, 최종 제안 구조가 직접 gain을 생성할지, 물리적으로 정의된 \(Q_t\), \(R_t\), reliability gate를 생성할지는 oracle 실험과 baseline 비교를 거쳐 결정해야 한다.

---

## 1. 현재 연구를 한 문장으로 다시 정의하면

현재 연구의 중심은 “저가 IMU 신호를 고가 IMU처럼 복원하는 것”이 아니다. 보다 정확한 정의는 다음과 같다.

> **저가 gyro의 시간가변 propagation uncertainty와 magnetometer·sun sensor·star tracker의 시간가변 measurement reliability가 존재하는 위성 자세 추정에서, 필터가 현재 상황에 맞게 모델과 센서를 신뢰하도록 적응시키는 문제**

이 정의가 중요한 이유는 저가 센서의 오차가 모두 단순한 백색잡음 증가로 표현되지 않기 때문이다.

- gyro bias와 온도 의존성은 측정 평균을 바꾼다.
- scale factor와 축 오정렬은 측정모델을 바꾼다.
- white noise와 random walk는 공분산을 바꾼다.
- magnetic interference와 순간 outlier는 Gaussian covariance만으로 다루기 어렵다.
- star tracker outage와 sun sensor eclipse는 센서의 유효성 자체가 변하는 문제다.

따라서 연구는 센서 값을 단순히 “깨끗하게 만드는 전처리”보다 다음 세 기능을 구분해야 한다.

```text
체계적 평균/모델 오차   → calibration 또는 명시적 상태 추정
확률적 불확실성 변화    → Q_t, R_t 또는 gain adaptation
순간 이상치·센서 무효   → reliability gate 또는 robust update
```

---

## 2. 지금까지 확인된 사실과 그 의미

### 2.1 불규칙 결측은 현재 핵심 문제가 아니다

실제 확인한 데이터에서는 센서 값이 중간중간 불규칙하게 비는 현상이 없었고 sampling은 정상적으로 들어왔다. 따라서 missing mask, imputation, sparse observation을 연구의 중심으로 둘 근거가 약해졌다.

이 결과는 부정적인 결과가 아니라 문제를 정확히 좁힌 결과다. 즉 현재의 핵심은 “측정이 없는 문제”가 아니라 “측정은 계속 들어오지만 품질과 오차 특성이 변하는 문제”다.

### 2.2 단순 measurement enhancement는 최종 성능을 뚜렷하게 개선하지 못했다

저가 IMU 값을 별도 네트워크로 보정한 뒤 Split-KalmanNet에 넣는 간단한 실험에서는 뚜렷한 최종 성능 향상이 확인되지 않았다.

이 결과로부터 확정할 수 있는 것은 다음 정도다.

- 현재 사용한 단순 전처리 방식은 충분한 효과를 보이지 않았다.
- 센서 값 MSE 감소가 상태 추정 MSE 감소를 보장하지 않는다.
- smoothing 또는 point-wise denoising은 bias drift, 온도 변화, 시간상관 오차를 제대로 처리하지 못할 수 있다.
- 전처리 네트워크가 innovation의 크기와 시간구조를 왜곡해 필터 내부 적응을 방해했을 가능성이 있다.

반대로 이 결과만으로 “센서 calibration은 필요 없다”고 결론 내리면 안 된다. 단순 denoising과 물리적으로 구조화된 bias·scale·temperature calibration은 서로 다른 문제다.

### 2.3 fast/slow context는 아직 유망한 가설이지 확인된 결과가 아니다

- slow 후보: gyro bias 변화, 온도 drift, 장기 noise-floor 변화
- fast 후보: vibration burst, magnetic interference, 순간 outlier, sensor invalidity

이 분류는 물리적으로 이해하기 쉽지만, 두 neural branch를 만들었다고 해서 네트워크가 자동으로 이 역할을 나누지는 않는다. 이 점은 향후 반드시 검증해야 한다.

---

## 3. 연구의 강점

## 3.1 실제 데이터와 실험 결과에 따라 문제를 수정했다

초기 가정이었던 불규칙 결측과 단순 measurement enhancement를 고집하지 않고, 실제 확인 결과에 따라 문제를 저가 센서의 시간가변 오차로 재정의했다. 이는 연구 설계에서 매우 중요한 장점이다.

## 3.2 완전한 black-box가 아니라 물리 필터 구조를 유지하려 한다

KalmanNet 계열은 상태전이와 측정모델을 버리지 않고, 불확실성이 들어가는 gain 계산 일부를 학습한다. 위성 자세 추정처럼 물리 모델과 안정성이 중요한 분야에서 순수 RNN보다 설명 가능성과 구조적 안정성을 확보하기 유리하다.

## 3.3 process 측과 measurement 측의 변화가 다르다는 점을 인식했다

저가 gyro의 문제와 magnetometer·sun sensor·star tracker의 문제는 필터에서 같은 위치에 들어가지 않는다. Split 구조를 고려한 것은 이 차이를 반영하려는 시도라는 점에서 타당하다.

## 3.4 oracle → ANN → SNN으로 단계화하려는 방향이 좋다

SNN을 처음부터 최종 해법으로 가정하지 않고, 먼저 oracle context의 유효성, 그다음 ANN estimator, 마지막으로 SNN 전환을 검증하려는 방향은 연구 실패 원인을 분리하는 데 매우 유리하다.

## 3.5 SNN이 실패해도 남는 fallback 연구가 있다

fast/slow structured context와 adaptive MEKF가 유효하다면 SNN이 기대한 이점을 보이지 않더라도 연구 자체는 유지될 수 있다. 이는 연구 리스크 관리 측면에서 강점이다.

---

## 4. 핵심 허점과 개선 방향

아래 항목은 억지로 단점을 만든 것이 아니라, 현재 구조가 논문이나 구현으로 넘어갈 때 실제로 문제가 될 가능성이 큰 부분이다.

---

## 4.1 연구 범위가 자세 추정과 자세제어 사이에서 혼재되어 있다

### 문제

현재 설계의 직접 출력은 quaternion, gyro bias, 필요 시 angular rate와 같은 **상태 추정값**이다. 이것만으로는 자세제어 알고리즘을 제안했다고 볼 수 없다.

### 무시할 경우 생기는 문제

- 논문 제목과 기여가 실제 구현 범위보다 커질 수 있다.
- estimator 성능 향상인지 controller 개선인지 구분하기 어렵다.
- reaction wheel, magnetorquer, control law, actuator saturation 등의 평가가 빠진 상태에서 “자세제어 성능”을 주장하게 될 수 있다.

### 개선

주 기여는 다음으로 고정하는 것이 좋다.

> **자세제어 시스템에 사용되는 적응형 위성 자세 추정기**

closed-loop 제어는 최종 검증으로 두고, 동일한 controller에 각 estimator를 연결해 pointing error, settling time, control effort를 비교한다.

---

## 4.2 MEKF는 기존 EKF에 단순히 하나를 추가하는 문제가 아니다

### 문제

MEKF에서는 quaternion 자체의 4차원 덧셈 오차가 아니라, 3차원 local attitude error를 사용한다.

권장 1차 상태는 다음과 같다.

\[
\bar{x}_t=(q_t,b_{g,t}),
\qquad
\delta x_t=
\begin{bmatrix}
\delta\theta_t\\
\delta b_{g,t}
\end{bmatrix}
\in\mathbb{R}^{6}.
\]

업데이트 후에는

\[
\hat q_t^+
=
\operatorname{Exp}_q(\widehat{\delta\theta}_t)
\otimes \hat q_t^-
\]

처럼 곱셈으로 자세 오차를 주입하고, local error를 다시 0으로 reset한다.

### 무시할 경우 생기는 문제

- quaternion component MSE와 실제 회전오차가 일치하지 않는다.
- gain의 state dimension과 Jacobian이 잘못 정의될 수 있다.
- MEKF reset 후 RNN hidden state가 어느 tangent frame의 정보를 담는지 불분명해진다.
- proposed model만 MEKF이고 baseline은 기존 direct-state EKF이면 공정 비교가 되지 않는다.

### 개선

Phase 0에서 다음을 수학적 계약으로 고정해야 한다.

- quaternion 순서와 회전 방향
- inertial-to-body 또는 body-to-inertial convention
- left 또는 right multiplicative error
- error injection과 reset Jacobian
- gyro가 propagation input인지 measurement인지
- 각 센서의 measurement function과 Jacobian
- tangent-space loss와 covariance 정의

그리고 주요 baseline도 동일한 MEKF shell 위에 구현해야 한다.

---

## 4.3 gyro, magnetometer, sun sensor, star tracker의 필터 내 역할이 아직 확정되지 않았다

### 문제

센서의 물리적 종류와 필터에서의 수학적 역할이 혼동될 수 있다.

Kinematic MEKF를 사용할 경우 권장 역할은 다음과 같다.

```text
Gyro          → attitude propagation input
Magnetometer  → body-frame reference vector update
Sun sensor    → body-frame reference vector update
Star tracker  → low-rate absolute attitude update
Temperature   → calibration/context 보조 입력
Actuator telemetry → 알려진 context 보조 입력
```

gyro는 센서지만 필터 안에서는 보통 propagation에 사용되므로, gyro white noise와 bias random walk는 주로 process-side uncertainty에 들어간다.

### 개선

Phase 0에서 센서별로 다음 표를 완성해야 한다.

| 항목 | 내용 |
|---|---|
| truth quantity | 시뮬레이터가 생성하는 실제 물리량 |
| measurement equation | 센서가 출력하는 값 |
| error model | bias, noise, scale, misalignment, outlier 등 |
| rate/latency | sampling, timestamp, 지연 |
| validity | eclipse, blinding, FOV, saturation 등 |
| estimator role | propagation 또는 update |
| context role | SoW 입력으로 사용할 수 있는 정보 |

---

## 4.4 저가 IMU 오차를 전부 gain adaptation으로 처리할 수는 없다

### 문제

다음 오차들은 서로 성격이 다르다.

#### 평균 또는 모델을 바꾸는 오차

- constant/temperature-dependent gyro bias
- scale factor
- axis misalignment
- magnetometer hard/soft iron
- sensor-to-body frame misalignment

#### 공분산으로 나타내기 적합한 오차

- gyro white noise
- bias random walk 또는 Gauss–Markov variation
- measurement noise floor 변화
- process-model uncertainty

#### robust 처리 대상

- magnetic outlier
- 순간 saturation
- sensor glitch
- star tracker false solution

### 무시할 경우 생기는 문제

네트워크가 모든 오차를 단순히 \(R_t\) 증가로 처리해 측정을 무시하는 방향으로 학습할 수 있다. 그러면 단기 MSE는 낮아질 수 있지만 bias는 교정되지 않고 관측 정보도 낭비된다.

### 개선

다음 원칙을 적용한다.

```text
mean/model error   → explicit state 또는 calibration
stochastic change  → Q_t/R_t adaptation
gross outlier      → gate 또는 robust likelihood
```

특히 gyro bias 값 자체와 bias random-walk intensity를 구분해야 한다.

- \(b_g\): MEKF가 추정하는 상태
- \(Q_b\): bias가 얼마나 빠르게 변하는지를 나타내는 불확실성

---

## 4.5 gyro + magnetometer만 사용할 때의 관측성 문제가 연구 효과와 섞일 수 있다

### 문제

한 시점의 magnetometer vector는 자세의 세 자유도를 모두 독립적으로 제공하지 않는다. 예상 자기장 벡터 \(\hat m^b\)에 대한 선형화 Jacobian은 대략

\[
H_\theta=-[\hat m^b]_\times
\]

이며 rank가 2다. 자기장 방향 주위의 회전은 그 순간 직접 관측되지 않는다.

시간에 따른 자기장 방향 변화와 위성 운동을 통해 전체 구간에서는 관측성이 확보될 수 있지만, 궤도·운동 조건에 따라 성능이 크게 달라진다.

### 무시할 경우 생기는 문제

- neural network가 관측 불가능한 정보를 “추정”한 것처럼 보일 수 있다.
- 실제로는 고정된 궤도 위상이나 maneuver schedule을 외운 것일 수 있다.
- 초기 자세와 궤도 위상이 바뀌면 성능이 급락할 수 있다.

### 개선

센서 시나리오를 분리한다.

1. **MEKF 단위 검증:** gyro + star tracker
2. **주 연구:** gyro + magnetometer + sun sensor + low-rate star tracker
3. **제한 운용 stress test:** gyro + magnetometer

그리고 초기 자세, 궤도 위상, maneuver timing, 자기장 profile을 trajectory 단위로 분리·무작위화한다.

---

## 4.6 Split-KalmanNet의 두 branch를 실제 \(Q\)와 \(R\)처럼 해석할 수 없다

### 문제

Split-KalmanNet의 gain을 개념적으로

\[
K_t=G_{1,t}H_t^\top G_{2,t}
\]

로 표현하더라도, \(G_1\)과 \(G_2\)가 실제 공분산을 유일하게 나타낸다는 보장은 없다.

고전 필터의 innovation covariance는

\[
S_t=H_tP_t^-H_t^\top+R_t
\]

이므로, innovation-side factor는 measurement noise \(R_t\)만이 아니라 prior uncertainty \(P_t^-\)도 포함한다.

또한 임의의 양수 \(c\)에 대해

\[
(cG_1)H^\top(c^{-1}G_2)
=G_1H^\top G_2
\]

이므로 최종 gain이 같아도 두 factor의 scale은 유일하지 않다.

### 무시할 경우 생기는 문제

- “prior branch가 실제 process uncertainty를 학습했다”고 과도하게 해석할 수 있다.
- “measurement branch가 실제 sensor reliability를 학습했다”고 주장하기 어렵다.
- implicit factor로 NEES/NIS를 계산하거나 covariance consistency를 주장하면 근거가 약해진다.

### 개선

두 경로를 모두 비교하는 것이 좋다.

#### 경로 A — 원 Split-KalmanNet

- 직접 gain factor를 학습
- 원 구조 재현과 성능 baseline으로 사용
- branch는 covariance 자체가 아닌 **latent gain factor**로 해석

#### 경로 B — structured adaptive MEKF

neural context가 SPD가 보장된 \(Q_t\), \(R_t\) 또는 scale을 출력하고, MEKF가 \(P_t^-\), \(S_t\), \(K_t\)를 명시적으로 계산한다.

예:

\[
Q_t=L_{Q,0}\operatorname{diag}(e^{z_{Q,t}})L_{Q,0}^{\top},
\]

\[
R_t=L_{R,0}\operatorname{diag}(e^{z_{R,t}})L_{R,0}^{\top}.
\]

이 경로는 해석성과 필터 일관성이 더 좋다.

---

## 4.7 현재 proposed SoW는 원래의 scalar SoW와 다른 개념이다

### 문제

원래 Adaptive KalmanNet의 scalar SoW와 달리, 현재 연구가 원하는 것은 residual, IMU, temperature, telemetry에서 추정하는 fast/slow vector context다.

end-to-end loss만 사용하면 이 값은 물리적 SoW가 아니라 단지 latent code일 수 있다.

### 무시할 경우 생기는 문제

- context 값의 변화가 실제 bias·vibration·measurement reliability를 의미한다고 잘못 해석할 수 있다.
- 서로 다른 원인이 비슷한 innovation을 만들 때 원인을 구분하지 못한다.
- 네트워크가 센서 상태가 아니라 trajectory timing을 encode할 수 있다.

### 개선

먼저 물리적으로 정의된 oracle context를 사용한다.

예:

\[
z_{P,t}=
[\log\alpha_{g,t},\log\alpha_{b,t}],
\]

\[
z_{R,t}=
[\log\alpha_{m,t},\log\alpha_{s,t},\log\alpha_{ST,t},p_{\text{outlier},t}].
\]

그다음 learned model에서 다음 두 가지를 명확히 구분한다.

- **supervised/semi-supervised context:** 물리 label에 대응
- **latent context:** 성능을 위한 내부 표현이며 물리적 의미를 주장하지 않음

---

## 4.8 fast/slow branch가 자동으로 역할을 나누지 않는다

### 문제

두 recurrent branch에 서로 다른 이름을 붙이는 것만으로 하나가 slow drift, 다른 하나가 fast event를 담당하지는 않는다.

가능한 실패 형태는 다음과 같다.

- 두 branch가 같은 값을 학습
- 한 branch만 사용되고 다른 branch가 collapse
- 역할이 뒤바뀜
- slow branch가 explicit bias state를 대신 암묵적으로 저장
- fast branch가 모든 문제를 measurement rejection으로 처리

### 개선

1. 시간상수 범위를 구조적으로 분리한다.
2. slow context에는 시간 변화 억제 regularization을 둔다.
3. fast context에는 sparse activation 또는 event supervision을 둔다.
4. simulator의 실제 변화 label을 이용해 auxiliary supervision을 적용한다.
5. slow-only, fast-only, 동시 변화 intervention 실험을 수행한다.
6. 두 branch의 상관과 ablation을 확인한다.

---

## 4.9 SNN 적용은 기능적 이유가 필요하다

### 문제

SNN을 사용했다는 사실만으로 신규성, 저전력, 저지연이 보장되지 않는다. 고정 주기 IMU를 여러 spike timestep으로 반복 변환하면 오히려 지연과 계산량이 늘 수 있다.

### 개선

SNN의 역할을 전체 Kalman gain 생성이 아니라 다음 중 하나로 좁히는 것이 좋다.

- abrupt change detector
- reliability gate
- \(Q/R\) scale estimator
- outlier probability estimator
- event-triggered context updater

그리고 반드시 동일 조건에서 비교한다.

- moving-window/EMA
- CUSUM 또는 classical change detector
- 작은 MLP
- 작은 GRU
- ANN dual-timescale model
- SNN model

SNN의 주장은 실제 hardware를 다루기 전까지 다음 수준으로 제한한다.

- spike sparsity
- active update ratio
- context detection latency
- 연산 이벤트 수
- 동일 성능에서의 모델 크기

---

## 4.10 데이터 분할과 공정 비교가 잘못되면 연구 전체가 무효화될 수 있다

### 위험

동일 trajectory를 겹치는 window로 나누어 train과 test에 넣으면 네트워크가 물리적 적응이 아니라 궤도 위상과 event timing을 외울 수 있다.

### 개선

반드시 trajectory 단위로 분리한다.

- initial attitude
- initial angular rate
- orbit phase
- maneuver schedule
- temperature profile
- vibration timing
- noise realization
- sensor outage interval

또한 proposed model과 baseline은 다음을 공유해야 한다.

- 동일 truth trajectory
- 동일 sensor realization
- 동일 estimator knowledge
- 동일 MEKF convention
- 동일 metric
- 동일 adaptation budget

---

## 5. 권장 최종 구조

## 5.1 기본 필터

1차 권장 모델은 **6차원 error-state Kinematic MEKF**다.

\[
\delta x=
[\delta\theta^\top,\delta b_g^\top]^\top.
\]

- gyro: propagation
- magnetometer: vector update
- sun sensor: vector update
- star tracker: low-rate absolute attitude update
- temperature와 actuator telemetry: context 보조 입력

## 5.2 structured adaptation

### process-side context

- gyro noise scale
- bias random-walk scale
- vibration-induced propagation uncertainty

### measurement-side context

- magnetometer reliability
- sun sensor reliability
- star tracker reliability
- outlier probability 또는 measurement gate

```text
Gyro / temperature / control telemetry
    → process-context estimator
    → Qgyro, Qbias scale

Mag/Sun/ST residual / validity / quality
    → measurement-context estimator
    → R scale, reliability gate

Q_t, R_t
    → classical MEKF covariance recursion
    → K_t
    → multiplicative attitude correction
```

## 5.3 Split-KalmanNet의 위치

Split-KalmanNet은 다음 역할을 맡는 것이 좋다.

- 기존 연구와의 연속성을 유지하는 backbone baseline
- 직접 gain learning 방식의 대표 모델
- structured \(Q/R\)-adaptive MEKF와 비교할 모델

oracle 실험에서 직접 gain modulation이 명백하게 유리할 경우 proposed 구조에 유지할 수 있다. 반대로 structured \(Q/R\) 방식이 비슷한 정확도에서 일관성·해석성이 좋다면 최종 proposed model을 그쪽으로 옮기는 것이 합리적이다.

## 5.4 SNN의 위치

가장 설득력 있는 1차 선택은 다음이다.

- slow context: long-memory ANN/GRU 또는 느린 recurrent state
- fast context: LIF 기반 SNN change/reliability detector
- 전체 필터: MEKF 또는 neural MEKF 유지

즉 모든 부분을 spiking으로 바꾸기보다, **변화가 드문 fast context update를 event-driven으로 처리하는 hybrid 구조**가 우선 후보가 된다.

---

## 6. 연구 가설을 검증 가능한 형태로 다시 쓰기

### H1 — 문제 존재성

시간가변 gyro uncertainty와 외부센서 reliability 변화가 동시에 존재하면 fixed-covariance MEKF와 stationary-trained KalmanNet의 성능과 일관성이 저하된다.

### H2 — structured context의 유용성

oracle \(Q/R\) 또는 reliability context를 제공하면 fixed/adaptive baseline보다 transition peak error와 recovery time이 개선된다.

### H3 — scalar보다 vector context가 필요한가

gyro propagation uncertainty와 외부센서 reliability가 독립적으로 변할 때 scalar SoW보다 branch-/sensor-specific vector context가 유리하다.

### H4 — dual-timescale이 필요한가

slow drift와 fast event가 함께 발생하는 조건에서 dual-timescale context가 single-timescale context보다 steady-state error와 adaptation latency를 동시에 줄인다.

### H5 — learned context의 실현 가능성

실제 onboard에서 사용할 수 있는 sensor/residual/telemetry feature만으로 oracle context 성능의 의미 있는 비율을 재현할 수 있다.

### H6 — SNN의 기능적 이점

SNN fast-context estimator가 동일한 정보와 유사한 parameter budget을 가진 ANN/GRU 대비 정확도를 유지하면서 더 sparse한 update와 충분히 짧은 detection latency를 제공한다.

---

## 7. 주장 수준과 필요한 증거

| 주장 | 필요한 최소 증거 |
|---|---|
| time-varying uncertainty가 문제다 | fixed baseline의 성능 저하와 일관성 악화 |
| SoW가 유용하다 | oracle context 실험 |
| vector SoW가 필요하다 | scalar/vector controlled ablation |
| fast/slow 분리가 유용하다 | slow-only, fast-only, combined intervention |
| branch-specific 구조가 process/measurement를 분리한다 | 원인 분리 실험과 branch ablation; 단순 latent factor 해석 금지 |
| learned context가 물리적 의미를 가진다 | context label 또는 intervention correlation |
| SNN이 유리하다 | ANN/GRU/classical detector와 동등 조건 비교 |
| 자세제어에 유리하다 | 동일 controller를 사용한 closed-loop 결과 |
| 실제 적용 가능하다 | 실제 센서 또는 hardware-in-the-loop 데이터에서 OOD 검증 |

---

## 8. 우선순위별 위험 평가

| 우선순위 | 위험 | 심각도 | 개선 가능성 |
|---:|---|---:|---:|
| 1 | MEKF와 센서 역할이 확정되지 않음 | 매우 큼 | 높음 |
| 2 | Split branch를 실제 Q/R로 과해석 | 매우 큼 | 높음 |
| 3 | SoW의 물리적 정의와 식별성 부족 | 매우 큼 | 높음 |
| 4 | truth/sensor/estimator model이 분리되지 않음 | 큼 | 매우 높음 |
| 5 | gyro+mag 관측성과 학습 prior가 혼재 | 큼 | 높음 |
| 6 | fast/slow branch 역할 붕괴 가능성 | 큼 | 중간~높음 |
| 7 | SNN의 역할과 비교 기준 부족 | 중간~큼 | 높음 |
| 8 | trajectory leakage와 baseline 불공정 | 매우 큼 | 매우 높음 |
| 9 | 자세추정과 자세제어 용어 혼재 | 중간 | 매우 높음 |

---

## 9. 최종 평가

이 연구는 폐기해야 할 방향이 아니다. 오히려 다음 조건을 충족하면 연구 문제가 훨씬 선명해진다.

1. MEKF와 센서 모델을 Phase 0에서 먼저 확정한다.
2. systematic error, stochastic uncertainty, outlier를 구분한다.
3. Split-KalmanNet branch를 물리 공분산으로 자동 해석하지 않는다.
4. SoW를 oracle physical context부터 검증한다.
5. 직접 gain learning과 structured \(Q/R\)-adaptation을 공정하게 비교한다.
6. dual-timescale의 필요성을 intervention 실험으로 입증한다.
7. ANN 구조가 먼저 유효할 때만 SNN으로 넘어간다.
8. 동일 MEKF shell, 동일 trajectory, 동일 metric으로 baseline을 비교한다.

이 조건을 만족할 때 가장 강한 연구 설명은 다음과 같다.

> **저가 gyro의 propagation uncertainty와 vector/absolute attitude sensor의 reliability가 서로 다른 시간척도로 변화하는 위성 자세 추정 환경에서, MEKF의 tangent-space 구조와 공분산 일관성을 유지하면서 structured context를 online 추정·반영하고, fast reliability change를 SNN으로 효율적으로 검출하는 적응형 neural attitude estimator**

이 설명은 단순한 “Split-KalmanNet + Adaptive KalmanNet + SNN의 조합”보다 문제, 구조, 실험, 기여가 명확하다.
