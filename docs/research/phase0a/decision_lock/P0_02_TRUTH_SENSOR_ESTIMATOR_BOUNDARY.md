# P0_02 Truth–Sensor–Estimator 정보 경계

> 작성일: 2026-07-30
> 표지 규칙: **[확인]** 실험·실측, **[문헌]** 논문·공식 자료, **[분석]** 수식·구조 해석, **[가설]** 검증 대상, **[결정]** 설계 선택, **[보류]** 근거 부족·후속 범위

| 항목 | 내용 |
|---|---|
| 목적 | simulation truth, 센서가 내보내는 데이터, onboard estimator가 아는 정보, oracle label과 평가 전용 값을 분리하여 data leakage를 막는다. |
| 입력 근거 | S0–S4, E1; `P0A_REFERENCE_REGISTER.md` |
| 결정 상태 | LOCK — 정보 등급과 leakage 규칙 고정 |
| 남은 TBD | 실제 OBC에서 확보 가능한 actuator telemetry와 각 센서 quality flag의 구체 인터페이스 |
| 다음 Gate | Phase 1 데이터 schema가 이 표의 `estimator-known/oracle/evaluation-only` 구분을 강제하고 자동 leakage 검사를 통과 |


## 1. 세 층의 계약

```text
Truth spacecraft/environment model
  ├─ q_NB,true, ω_true, orbit, sun, magnetic field, disturbances, event states
  ↓
Sensor output model
  ├─ timestamped measurements, validity/quality, latency, saturation/fault behavior
  ↓
Onboard estimator model
  ├─ nominal model, measurements, allowed telemetry, estimated state/covariance/context
  └─ true state, true event cause, true Q/R는 보지 못함
```

- **[결정] Truth**는 물리적 상태와 sensor-error generator의 hidden state를 모두 가진다.
- **[결정] Sensor layer**는 truth를 실제 장치가 출력할 법한 값·timestamp·flag로 변환한다. sensor output에 포함되지 않은 true parameter는 estimator에 전달하지 않는다.
- **[결정] Estimator layer**는 sensor packet, nominal orbit/environment model, 사전에 calibration된 상수, 실제로 연결된 telemetry만 사용한다.
- **[결정] Oracle runner**만 hidden noise scale·event label을 사용하여 `Q/R/gate` 상한 성능을 측정한다. Oracle 결과를 deployable estimator 결과와 혼용하지 않는다.

## 2. 정보 경계 표

표 안의 기호는 `예`, `아니오`, `간접`, `조건부`를 사용한다. `Estimator가 직접 앎`은 코드가 입력으로 받는다는 뜻이며, 상태로 추정한다는 뜻과 다르다.

| 물리량/파라미터 | Truth에 존재 | 센서 출력에 반영 | Estimator가 직접 앎 | Context 입력 가능 | Oracle label만 가능 | 평가 전용 |
|---|---|---|---|---|---|---|
| true attitude `q_NB,true` | 예 | ST·mag·sun에 간접 반영 | 아니오 | **금지** | 아니오 | 예 |
| true angular rate `ω_true^B` | 예 | gyro에 반영 | 아니오 | **금지**; gyro measurement는 가능 | 아니오 | 예 |
| true gyro bias `b_g,true` | 예 | gyro에 가산 | 아니오; `b_g`를 상태로 추정 | **금지**; 추정 bias는 가능 | oracle 진단에는 가능 | 예 |
| gyro bias random-walk intensity `S_b(t)` | 예 | bias trajectory 통계에 간접 반영 | nominal `S_b0`만 앎 | 실제 추정 feature로는 직접 금지 | 예: `α_b` label | 예 |
| gyro white-noise multiplier `α_g(t)` | 예 | gyro sample variance에 반영 | nominal noise만 앎 | 직접 입력 금지 | 예: `α_g` label | 예 |
| temperature-induced mean bias `b_T(T)` | 예 | gyro mean에 반영 | calibration map이 있으면 nominal map만 앎 | measured temperature는 가능; true mean-bias label 직접 입력 금지 | calibration 학습 label로 조건부 | 예 |
| gyro scale-factor matrix | 예 | gyro measurement에 반영 | 사전 calibration 값만 조건부 | raw true matrix 금지 | calibration 실험 label로 조건부 | 예 |
| gyro axis misalignment/cross-axis matrix | 예 | gyro measurement에 반영 | nominal mounting/calibration만 조건부 | raw true matrix 금지 | calibration 실험 label로 조건부 | 예 |
| true Earth magnetic field `m_true^N` | 예 | magnetometer에 반영 | 아니오 | **금지** | 아니오 | 예 |
| onboard magnetic-field model `m_model^N(r,t)` | 별도 truth와 비교 가능 | 센서 출력에는 직접 없음 | 예; orbit/time model에서 계산 | model residual feature는 조건부 | 아니오 | 예 |
| magnetic disturbance `d_mag^B` | 예 | magnetometer에 가산 | 아니오 | true disturbance 금지; measured norm/residual은 가능 | interference label 가능 | 예 |
| true inertial sun vector `s_true^N` | 예 | CSS/sun output에 반영 | onboard ephemeris model을 별도 계산 | true vector 직접 입력 금지; model vector는 가능 | 아니오 | 예 |
| eclipse/FOV validity | 예 | CSS counts와 validity에 반영 | **sensor validity flag는 예** | 예; 실제 onboard flag로 허용 | hidden eclipse geometry는 oracle 가능 | 예 |
| star-tracker inlier noise scale | 예 | ST quaternion scatter에 반영 | nominal accuracy만 앎 | true scale 금지; quality/innovation은 가능 | 예: `α_R,ST` label | 예 |
| star-tracker latency | 예 | packet timestamp/arrival time에 반영 | packet timestamp로 예 | 예; age-of-measurement 허용 | hidden injected delay는 oracle 진단 | 예 |
| star-tracker quality flag | sensor generator에 존재 | packet에 포함 | **예, 장치가 제공한다고 명세한 경우** | 예 | 아니오 | 예 |
| star-tracker outage | 예 | packet absence/invalid flag | packet absence 또는 flag로 예 | 예; availability feature | event cause label은 oracle | 예 |
| commanded torque `τ_cmd` | 예 | actuator telemetry에 조건부 반영 | 연결된 telemetry이면 예 | 예; exogenous context | 아니오 | 예 |
| unknown disturbance torque `τ_dist` | 예 | 자세·rate에 간접 반영 | 아니오 | true torque 금지; innovation/rate change는 가능 | disturbance label 가능 | 예 |
| reaction-wheel speed | RW truth를 쓰면 예 | wheel telemetry로 조건부 | 실제 버스 연결 시 예 | 예; vibration proxy/known regime | hidden imbalance force는 oracle | 예 |
| magnetorquer state/current | MTQ truth를 쓰면 예 | telemetry로 조건부 | 실제 버스 연결 시 예 | 예; mag update gate의 강한 보조정보 | hidden residual dipole은 oracle | 예 |
| event label (`thermal`, `vibration`, `interference` 등) | 예 | 보통 직접 출력되지 않음 | 아니오 | **deployable input 금지** | 예 | 예 |
| 실제 continuous/discrete `Q_t` | sensor-error generator에서 정의 | 결과 통계에만 반영 | nominal `Q_0`만 앎 | 직접 입력 금지 | oracle label 가능 | 예 |
| 실제 sensor-specific `R_{j,t}` | sensor generator에서 정의 | measurement scatter에 반영 | nominal `R_{j,0}`만 앎 | 직접 입력 금지 | oracle label 가능 | 예 |

## 3. 실제 onboard에서 사용할 수 있는 정보

**[결정] 허용 기본 입력**

- gyro, magnetometer, CSS/sun-vector, star-tracker measurement와 timestamp
- packet validity, saturation, quality, age-of-measurement처럼 실제 인터페이스에 존재하는 metadata
- measured temperature
- estimator의 prior/posterior state, bias estimate, innovation, normalized innovation energy, previous correction, covariance diagonal/trace
- known orbit/time에서 계산한 onboard sun·magnetic reference model
- 실제 연결되어 들어오는 commanded torque, reaction-wheel speed, magnetorquer current/state

**[결정] 조건부 입력**

- accelerometer high-pass/RMS vibration feature: 실제 accelerometer sample에서 계산할 때만 허용
- reaction-wheel imbalance proxy: wheel speed와 사전 model에서 계산할 수 있을 때만 허용
- star-tracker quality: 선택 제품이 해당 flag를 제공할 때만 허용

## 4. simulation에서만 가능한 oracle 정보

- injected `α_g(t)`, `α_b(t)`, `α_R,j(t)`
- true outlier/inlier indicator와 event cause
- true bias, scale/misalignment, magnetic interference vector
- unknown disturbance torque
- truth attitude/rate와 true sensor error realization

이 정보는 `oracle_context/` 또는 `evaluation/` namespace에만 저장하고, deployable feature tensor와 다른 파일 또는 schema group에 둔다.

## 5. Network input으로 쓰면 data leakage가 되는 정보

1. true attitude/rate/bias 또는 그 미래값
2. injected event start/end와 event class
3. actual `Q_t`, `R_t`, noise multiplier
4. true magnetic/sun vector를 estimator model error 없이 직접 제공하는 값
5. future innovation, future sensor validity, future actuator command
6. train/test에 걸친 trajectory ID·orbit phase·seed를 모델이 식별할 수 있는 encoding
7. test trajectory로 계산한 normalization statistics

**[분석]** oracle context는 “학습 input으로 허용된 정보”가 아니라 “해당 uncertainty가 완벽히 알려졌을 때 가능한 상한”이다. Oracle 결과가 좋더라도 실제 feature로 재현되지 않으면 deployable contribution으로 주장할 수 없다.

## 6. Ground-truth 평가에만 사용할 정보

- SO(3) geodesic attitude error
- true gyro-bias error
- NEES의 true tangent error
- event-conditioned peak/recovery metrics
- context target error와 detection delay
- divergence 원인 분류

## 7. 데이터 schema 권장안

```yaml
truth/:
  q_NB_true, omega_B_true, b_g_true, orbit_state, sun_N, mag_N,
  disturbance_torque, injected_event_state
sensor/:
  gyro, mag, css_raw, sun_vector_B, star_quaternion_NB,
  temperature, timestamp, arrival_time, validity, quality
estimator_allowed/:
  reference_sun_N, reference_mag_N, command_torque, wheel_speed, mtq_state
oracle/:
  alpha_g, alpha_b, alpha_R_by_sensor, inlier_indicator, event_label
estimate/:
  q_NB_hat, b_g_hat, P, innovation, S, NIS, gate, context_hat
```

`estimator_allowed/`와 `oracle/`는 API와 파일 permission 수준에서 분리한다.

## 8. Gate checklist

- [ ] deployable runner가 `truth/`와 `oracle/` key를 열지 못한다.
- [ ] oracle runner는 동일 sensor realization을 재사용하고 `Q/R/gate`만 다르게 설정한다.
- [ ] timestamp와 arrival time을 구분한다.
- [ ] 모든 telemetry는 “실제 OBC 제공 가능” 근거 열을 갖는다.
- [ ] normalization은 train trajectory에서만 산출한다.
- [ ] evaluation script만 truth state에 접근한다.
