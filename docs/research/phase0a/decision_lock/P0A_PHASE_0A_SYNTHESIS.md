# Phase 0A 종합 판단 — 연구·센서·MEKF·Context 의사결정 잠금

> 작성일: 2026-07-30
> 표지 규칙: **[확인]** 실험·실측, **[문헌]** 논문·공식 자료, **[분석]** 수식·구조 해석, **[가설]** 검증 대상, **[결정]** 설계 선택, **[보류]** 근거 부족·후속 범위

| 항목 | 내용 |
|---|---|
| 목적 | Phase 1 구현 전에 연구 질문, 정보 경계, 센서 역할, 수학 convention, neural insertion과 즉시 검증 Gate를 사람이 이해할 수 있는 한 문서로 통합한다. |
| 입력 근거 | S0–S4, E1–E4; 세부 근거는 `P0A_REFERENCE_REGISTER.md` |
| 결정 상태 | 핵심 설계는 LOCK, 실제 하드웨어 수치와 일부 성능 임계값은 PROVISIONAL/TBD |
| 남은 TBD | 위성 mass/inertia, 실제 기준 IMU·mag·sun·ST 제품, 실측 기반 잡음 파라미터, 최종 통계 합격 임계값 |
| 다음 Gate | 사용자 승인 항목을 잠근 뒤 `Phase 1A`에서 Basilisk truth/sensor와 UNIT-ST MEKF를 최소 구현·검증 |


## 1. 한 문장 연구 정의

> **[결정]** 본 연구는 저가 gyro와 외부 자세센서의 시간가변 오차·신뢰도 변화 아래에서, 위성의 body-to-inertial 자세 quaternion과 gyro bias를 안정적이고 일관되게 추정하도록 `MEKF + structured uncertainty context`를 결합하는 **ADCS용 적응형 자세 추정기**를 연구한다.

추정 대상은

\[
\bar x_t=(q_{NB,t},\,b_{g,t}),\qquad
\delta x_t=[\delta\theta_t^\top,\,\delta b_{g,t}^\top]^\top\in\mathbb R^6
\]

이다. 여기서 `q_NB`는 body 좌표를 inertial 좌표로 능동 회전시키는 scalar-first Hamilton quaternion이다. 자세제어기는 연구의 주 기여가 아니며, 동일 controller·actuator를 고정한 후 estimator만 바꾸는 closed-loop 검증은 최종 적용성 평가다.

## 2. 현재 근거로 내릴 수 있는 핵심 판단

### 2.1 문제 방향

- **[확인]** 실제 데이터에서 불규칙 IMU 결측은 핵심 문제가 아니었다.
- **[확인]** 단순 measurement enhancement를 Split-KalmanNet 앞에 추가한 실험에서 뚜렷한 최종 자세 추정 개선은 확인되지 않았다. 보존된 수치가 없으므로 개선량을 재구성하지 않는다.
- **[분석]** 따라서 “측정값을 더 깨끗하게 만들면 상태 오차도 자동으로 줄어든다”는 경로보다, bias·drift·가변 noise·sensor validity를 필터 내부의 state, `Q/R`, gate로 분리하여 다루는 편이 연구 질문과 더 직접적으로 연결된다.

### 2.2 필터와 센서 역할

```text
Basilisk truth spacecraft/environment
        ↓  true attitude/rate, sun, magnetic field, eclipse, events
parameterized sensor output layer
        ↓  gyro, mag, CSS/sun vector, ST quaternion, temperature, flags
onboard estimator
        ↓  q_NB, gyro bias, P, innovation, NIS, context estimate
```

- **[결정]** gyro는 propagation input이다.
- **[결정]** magnetometer는 body magnetic-vector update, sun sensor는 CSS constellation에서 재구성한 body sun-vector update, star tracker는 low-rate absolute-attitude update다.
- **[결정]** temperature와 알려진 actuator telemetry는 calibration/context 보조 입력이다.
- **[결정]** accelerometer는 1차 자세 update에서 제외하고 vibration proxy 후보로만 둔다.

### 2.3 오차 처리 원칙

| 오차 종류 | 기본 처리 위치 | 예 |
|---|---|---|
| 평균·모델 오차 | explicit state 또는 calibration | gyro bias, temperature mean-bias model, 고정 scale/misalignment calibration |
| zero-mean stochastic uncertainty 변화 | `Q_t/R_t` adaptation | gyro white-noise PSD, bias-RW intensity, mag/sun/ST inlier noise |
| gross outlier·invalidity | validity/reliability gate 또는 robust update | eclipse, ST outage/false solution, mag spike/saturation |

**[결정]** 동일 현상을 동시에 mean correction과 covariance inflation과 gate에 중복 배치하지 않는다. 특히 temperature-dependent mean bias는 calibration/state 문제이며, calibration 후 잔여 불확실성만 `Q_b` 또는 gyro process scale로 보낸다.

## 3. 잠글 공통 수학 계약

- quaternion ordering: **scalar-first `[w,x,y,z]`**
- attitude direction: **`q_NB`, body → inertial active rotation**
- vector prediction: `v^B=C_{BN}(q)v^N=R_{NB}(q)^T v^N`
- multiplicative error: **right-multiplicative local error**

\[
q_{NB}^{true}=\hat q_{NB}\otimes\delta q(\delta\theta)
\]

- propagation:

\[
\hat q_{k+1}^{-}=\operatorname{normalize}\left(
\hat q_k^{+}\otimes\operatorname{Exp}_q((\omega_{m,k}-\hat b_{g,k})\Delta t)
\right)
\]

- first-order error dynamics:

\[
\delta\dot\theta=-[\hat\omega]_\times\delta\theta-\delta b_g-w_g,
\qquad
\delta\dot b_g=w_b
\]

- body-vector Jacobian for residual `y-h(q)`:

\[
H_v=\begin{bmatrix}[\hat v^B]_\times & 0\end{bmatrix}.
\]

- injection/reset:

\[
\hat q^+=\hat q^-\otimes\operatorname{Exp}_q(\widehat{\delta\theta}),\quad
\hat b_g^+=\hat b_g^-+\widehat{\delta b_g},
\]

\[
G_{reset}\simeq\operatorname{blkdiag}
\left(I-\tfrac12[\widehat{\delta\theta}]_\times, I\right).
\]

**[분석]** 이 convention은 right-error MEKF의 gyro error dynamics와 body-vector 측정 Jacobian을 간결하게 만들며, Solà의 local angular-error injection/reset 계열과 일치한다. Basilisk가 제공하는 `sigma_BN`은 simulator–estimator 어댑터에서 `C_BN`을 얻은 뒤 전치하여 `R_NB`로 변환한다. 이 경계는 test vector로 검증하며 이름만 보고 부호를 추정하지 않는다.

## 4. 권장 Truth/Sensor 시나리오

| 시나리오 | 구성 | 목적 | 상태 |
|---|---|---|---|
| `UNIT-ST` | gyro + star tracker | quaternion, bias, asynchronous update의 단위 검증 | LOCK |
| `MAIN-FUSION` | gyro + magnetometer + CSS-derived sun vector + low-rate ST | 주 연구 및 sensor-specific reliability 비교 | LOCK |
| `STRESS-MAG` | gyro + magnetometer | eclipse/ST 부재 및 제한 관측성 stress | LOCK |

**[결정]** 1차 orbit은 제품 독립적인 near-polar circular LEO representative configuration으로 둔다. 임시 기본값은 `h=550 km`, `i=97.6°`, `e=0`이며 이는 임무 사실이 아니라 benchmark 가정이다. RAAN과 initial true anomaly는 trajectory 단위로 무작위화한다. 위성은 CoM 중심 principal-axis rigid cuboid로 두고 mass·dimensions로 inertia를 계산하되 실제 수치는 대상 bus/CAD가 확정될 때 교체한다.

**[결정]** Tier 0은 prescribed/단순 rigid motion으로 MEKF를 검증한다. Tier 1에 gravity-gradient와 bounded unknown torque pulse를 순차 도입한다. aerodynamic torque, SRP torque, residual magnetic-dipole torque, full structural reaction-wheel vibration은 geometry/hardware가 필요한 Tier 2로 미룬다. 다만 RW vibration의 센서 영향은 Tier 1에서 variance-burst proxy로 먼저 시험한다.

## 5. Split-KalmanNet과 structured 경로의 지위

### 경로 A — Direct Split-KalmanNet

\[
K_t=G_{1,t}H_t^\top G_{2,t}.
\]

- **[문헌]** Split 구조는 prior-side와 innovation-side factor를 별도 RNN으로 학습한다.
- **[분석]** `G1`과 `G2`는 scale ambiguity가 있고 innovation covariance에는 prior와 measurement uncertainty가 함께 들어가므로, 실제 `P^-`, `Q`, `R`, `S^{-1}`로 자동 해석할 수 없다.
- **[결정]** 기존 연구 연속성을 위한 direct-gain backbone/baseline으로 유지한다.

### 경로 B — Structured adaptive MEKF

```text
physical/estimated context
  → positive Q block scales + sensor-specific R scale + reliability gate
  → explicit MEKF covariance recursion
  → K
  → multiplicative injection/reset
```

- **[결정]** proposed 후보이자 해석 가능한 fallback이다.
- **[분석]** SPD를 보장할 수 있고 NIS/NEES·innovation consistency를 평가할 수 있다.
- **[가설]** 정확한 oracle context에서도 개선이 없다면 learned context 및 SNN 확장을 중단한다.

## 6. 최소 oracle context 계약

persistent dense vector를 처음부터 크게 만들지 않고, **event-local 4-value interface**를 우선한다.

\[
c_t=\left[
\log\alpha_{g,t},
\log\alpha_{b,t},
\log\alpha_{R,j,t},
\rho_{j,t}
\right]
\]

- `α_g`: gyro-driven attitude process uncertainty scale
- `α_b`: gyro bias random-walk intensity scale
- `α_R,j`: 현재 update sensor `j`의 **inlier** measurement variance scale
- `ρ_j`: 현재 sensor의 validity/outlier reliability; sensor identity `j`는 metadata

이로써 각 event에서 차원은 4로 유지하면서 mag/sun/ST의 독립 변화를 sensor identity로 분리한다. dense batch 표현이 필요할 때만 `2+2N_s`로 펼친다. temperature mean bias는 별도 context 차원으로 두지 않고 calibration 후 잔여 불확실성을 `α_b/α_g`에 반영한다.

검증 순서는 `oracle scalar → oracle event-local vector → ANN estimator → 기능이 확인된 fast channel만 SNN`이다.

## 7. Phase 1 진입 최소 Gate

1. `P0_05_MEKF_CONVENTION_TEST_VECTORS.md`의 quaternion/DCM/주입/reset test가 모두 통과한다.
2. mag·sun analytic Jacobian이 central finite difference와 설정 tolerance 내에서 일치한다.
3. `UNIT-ST` 정상 조건에서 attitude와 constant gyro bias가 수렴하고 장시간 quaternion norm, covariance symmetry/SPD가 유지된다.
4. matched fixed-noise 조건에서 tuned MEKF와 oracle-scaled MEKF가 통계적으로 동일한 정상 성능을 보인다. oracle이 정상조건에서 부당한 이득을 가져서는 안 된다.
5. process-uncertainty step과 measurement-reliability step에서 fixed mismatched MEKF의 peak/recovery/consistency 저하가 재현된다.
6. oracle Q/R/gate가 적어도 하나의 transition metric을 반복적으로 개선해야 ANN context 학습으로 간다.
7. 모든 train/validation/test는 trajectory-level로 분리되고 동일 sensor realization을 estimator 간 공유한다.

## 8. 최종 판단

**[결정]** Phase 0A에서 가장 합리적인 연구 구조는 다음이다.

```text
6D Kinematic MEKF shell
  ├─ classical fixed/tuned/robust baselines
  ├─ direct-gain Split-KalmanNet baseline
  └─ structured Q/R/reliability-adaptive MEKF proposed candidate
          ↑
    oracle → ANN → selected fast channel SNN
```

연구의 신규성은 “SNN을 썼다”가 아니라, **시간가변 process uncertainty와 sensor-specific measurement reliability를 물리적 정보 경계 안에서 분리하고, tangent-space MEKF에서 공정하게 검증하는 것**에서 확보해야 한다.

## 9. 사용자 승인이 필요한 항목 — 최대 8개

| # | 권장 선택 | 대안 | 권장 이유 | 변경 시 재작성 범위 | 지금 확정? | Phase 1 후 가능? |
|---:|---|---|---|---|---|---|
| 1 | `q_NB`, scalar-first Hamilton, right error | `q_BN` 또는 left error | 수식·test가 가장 직접적이며 일반 라이브러리와 호환 | Math Contract, 모든 Jacobian·adapter·test | **예** | 아니오 |
| 2 | 6D Kinematic MEKF `[q,b_g]` | dynamic MEKF에 `ω`, torque state 추가 | 센서 uncertainty 효과를 dynamics mismatch와 분리 | Truth/MEKF/context 전반 | **예** | 아니오 |
| 3 | MAIN sun input은 CSS constellation → WLS body sun vector | 이상적 direct sun-vector sensor만 사용 | 실제 센서 validity/FOV를 보존하면서 MEKF interface는 단순 | Sensor spec, Basilisk adapter | **예** | 일부 가능 |
| 4 | near-polar circular LEO 대표값을 provisional 사용 | 특정 임무궤도 즉시 고정 | 자기장·일식 변화가 있고 임무 미정 상태에서 재현 가능 | Truth config와 OOD split | 아니오—provisional 승인 | 예 |
| 5 | MTi-2-5A-T는 후보 자산으로만 두고 실측 우선 | 제조사 typical 값을 baseline truth로 사용 | suffix/firmware/실물 편차 및 환경 의존성 방지 | Sensor parameter manifest | 아니오 | 예, hardware characterization 후 |
| 6 | event-local 4-value oracle context | 처음부터 8D dense sensor vector | 최소 차원·중복 억제·sensor identity 재사용 | Context contract와 oracle runner | **예** | vector 확장은 가능 |
| 7 | Split direct-gain은 baseline, structured 경로는 proposed 후보 | Split을 곧바로 최종 proposed로 고정 | branch 물리 해석 과장을 피하고 consistency 분석 가능 | Neural insertion·논문 claim | **예** | oracle 결과로 최종 지위 변경 가능 |
| 8 | Phase 1 pilot 후 통계 효과량 임계값을 최종 잠금 | 지금 임의의 개선률 고정 | noise scale·trajectory 길이 미확정 상태의 허위 정밀성 방지 | Immediate test의 수치 Gate만 | 아니오 | **예** |

## 10. 다음 실행 제목

**Phase 1A — Basilisk Truth/Sensor Simulator 및 Gyro+Star-Tracker Kinematic MEKF 최소 구현·검증**
