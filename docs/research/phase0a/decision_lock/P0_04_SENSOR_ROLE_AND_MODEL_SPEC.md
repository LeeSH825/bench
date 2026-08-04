# P0_04 센서 역할 및 모델 명세

> 작성일: 2026-07-30
> 표지 규칙: **[확인]** 실험·실측, **[문헌]** 논문·공식 자료, **[분석]** 수식·구조 해석, **[가설]** 검증 대상, **[결정]** 설계 선택, **[보류]** 근거 부족·후속 범위

| 항목 | 내용 |
|---|---|
| 목적 | gyro/IMU, magnetometer, sun sensor, star tracker, temperature, optional accelerometer의 측정식·rate·오차·필터 역할을 고정한다. |
| 입력 근거 | S0–S4, E1, E4; `P0_03_TRUTH_MODEL_SPEC.md`, `P0_05_MEKF_MATH_CONTRACT.md` |
| 결정 상태 | LOCK — 센서 역할·측정 형태·scenario; PROVISIONAL — rate/latency representative values; TBD-BY-HARDWARE — 오차 수치 |
| 남은 TBD | 실제 sensor product, mounting, data rate, timestamp/quality format, 실측 noise/bias/temperature characterization |
| 다음 Gate | 각 sensor unit test가 분포·frame·timestamp·validity를 통과하고 UNIT-ST/MAIN-FUSION/STRESS-MAG packet을 재생 가능하게 생성 |


## 1. 공통 표기

- `C_AB`: B-frame 좌표를 A-frame 좌표로 변환
- `S_j`: sensor `j`의 raw measurement frame
- `B`: spacecraft body frame
- `N`: inertial frame
- `sat(·)`: component/range saturation
- `A_j`: scale, non-orthogonality, cross-axis sensitivity를 포함하는 calibration matrix
- 모든 packet은 `measurement_time`, `arrival_time`, `validity`, optional `quality`를 가진다.

**[결정]** estimator core에는 nominal mounting/calibration으로 body frame에 변환된 측정을 전달한다. raw `S_j` packet과 calibrated `B` packet은 모두 log하여 calibration/mounting 오류를 분리한다.

## 2. 대표 sampling/latency profile

다음은 제품 사양이 아니라 **[결정: Phase 1 representative configuration]**이다.

| 채널 | 기본 rate | Tier-0 latency | Tier-1 latency/async | 비고 |
|---|---:|---:|---:|---|
| gyro | 100 Hz | 0 | timestamp jitter optional | propagation clock |
| magnetometer | 10 Hz | 0 | 0–1 sample jitter | vector update |
| CSS raw | 10 Hz | 0 | independent head validity | WLS input |
| reconstructed sun vector | 5 Hz | 0 | CSS window/WLS delay 기록 | tangent vector update |
| star tracker | 1 Hz | 0 | 0.1 s representative + outage | absolute attitude update |
| temperature | 1 Hz | 0 | slow asynchronous | calibration/context |
| accelerometer proxy | 100 Hz | 0 | optional | vibration feature only |

실제 제품이 정해지면 rate와 latency를 교체하고 모든 trajectory를 재생성한다. 알고리즘 간에는 동일 packet stream을 공유한다.

## 3. 3축 gyro/IMU

### 3.1 Measurement equation

Raw sensor-frame model:

\[
y_g^{S_g}=
\operatorname{sat}\!\left(
A_g C_{S_gB}\omega_{true}^B
+b_{g,r}(t)+b_{g,T}(T(t))+v_g(t)+v_{vib}(t)
\right).
\]

Estimator body-frame input after nominal calibration:

\[
\omega_m^B=\hat C_{BS_g}\hat A_g^{-1}
\left(y_g^{S_g}-\hat b_{g,T}(T_m)\right).
\]

Tier-0 core model:

\[
\omega_m^B=\omega_{true}^B+b_g^B+v_g^B,
\qquad
\dot b_g^B=w_b^B
\]

또는 finite-correlation GM model을 선택할 때

\[
\dot b_g=-\tau_b^{-1}b_g+w_b.
\]

**[결정]** Phase 1A는 random-walk bias를 기본으로 한다. finite `τ_b` GM model은 실측 Allan/온도 시험이 지원할 때 profile로 추가한다.

### 3.2 Specification table

| 항목 | 내용 |
|---|---|
| truth quantity | true body angular rate `ω_true^B`, true bias hidden state, temperature, vibration regime |
| measurement equation | 위 식; baseline `ω_m=ω+b_g+v_g` |
| output dimension/unit | `3`, rad/s; raw device unit가 deg/s이면 ingest 시 SI로 변환 |
| baseline error | zero-mean white noise + constant/slow bias + bias RW |
| time-varying error | `S_g(t)` noise-floor scale, `S_b(t)` bias-RW scale, vibration burst |
| calibration/model error | temperature mean bias, scale factor, non-orthogonality, sensor-to-body mounting |
| outlier/invalidity | saturation, clipping, quantization, packet validity; missing은 core issue가 아님 |
| rate/latency | representative 100 Hz, timestamped; hardware TBD |
| estimator role | **propagation input**, not ordinary attitude measurement update |
| context role | measured temperature/rate increment/vibration feature allowed; true bias/noise scale is oracle only |
| source status | architecture: [문헌]/[결정]; numeric values: 실측 > Allan/static/rotation > official datasheet > literature > assumption |

### 3.3 Bias와 uncertainty의 구분

- `b_g`: mean error; explicit MEKF state
- `S_b` 또는 `Q_b`: bias가 얼마나 빨리 움직이는지; process covariance
- `b_{g,T}(T)`: temperature-dependent mean; calibration model
- `b_{g,T}-\hat b_{g,T}`의 잔여: `S_b/S_g` scale 또는 model-uncertainty context
- `v_vib`: inlier variance burst이면 `α_g`; clipping/spike이면 gate/fault handling

### 3.4 Xsens MTi-2-5A-T 사용 정책

- **[확인]** 실제 자산 후보로 제품명이 알려져 있다.
- **[문헌]** 제조사 공식 leaflet/documentation은 nominal range·noise·bias stability·interface 확인 출처다.
- **[결정]** suffix/firmware/설정이 일치하는 공식 자료와 실측 전에는 numeric truth profile에 자동 복사하지 않는다.
- 필요한 시험: warm-up 포함 정지시험, Allan deviation, multi-temperature soak, rate-table multi-axis scale/misalignment, vibration/actuator-on test.

## 4. Magnetometer

### 4.1 Measurement equation

\[
y_m^{S_m}=\operatorname{sat}\!\left(
A_m C_{S_mB}C_{BN}(q_{NB})m_{true}^N
+b_m^{S_m}+d_m^{S_m}(t)+v_m^{S_m}
\right).
\]

Nominal body-frame update:

\[
y_m^B=C_{BN}(\hat q)m_{model}^N+v_m^B.
\]

`d_m`은 spacecraft current, MTQ, hard/soft-iron residual, transient interference를 포함할 수 있다.

### 4.2 Specification table

| 항목 | 내용 |
|---|---|
| truth quantity | WMM magnetic vector `m_true^N`과 spacecraft magnetic disturbance |
| measurement equation | 위 raw 3D vector 식 |
| output dimension/unit | `3`, tesla 또는 nT; estimator 내부 SI 단위 고정 |
| baseline error | calibrated additive Gaussian inlier noise, exact mounting |
| time-varying error | inlier noise scale increase, slowly varying field-model residual |
| calibration/model error | hard iron `b_m`, soft iron/scale `A_m`, axis/mounting misalignment |
| outlier/invalidity | MTQ interference, spike/stuck/saturation, norm anomaly |
| rate/latency | representative 10 Hz; hardware TBD |
| estimator role | body magnetic-vector measurement update |
| context role | mag norm mismatch, innovation, MTQ state allowed; true disturbance/`R_t` oracle only |
| source status | Basilisk base model [문헌]; spacecraft interference wrapper [결정/가설]; 수치 TBD |

### 4.3 Raw vector vs normalized direction

- **[결정]** MAIN-FUSION과 STRESS-MAG의 1차 update는 calibrated **raw 3D vector**와 `R_m∈R^{3×3}`를 사용한다.
- 장점: field magnitude와 norm anomaly 정보를 보존하며 WMM model과 직접 비교할 수 있다.
- 제한: 순간 Jacobian rank는 2이고 field-model magnitude error가 residual에 들어온다.
- **[보류]** normalized-direction update는 ablation으로 유지한다. 이 경우 covariance를 normalization Jacobian으로 운반하고 2D tangent residual을 써야 한다.

## 5. Coarse Sun Sensor / sun-vector channel

### 5.1 Raw CSS head

CSS head `i`의 이상적 출력:

\[
z_i=\operatorname{sat}\left(
k_i f_e\,\max(0,n_i^{B\top}s_{true}^B)+b_i+v_i
\right),
\quad s_{true}^B=C_{BN}(q)s^N.
\]

여기서 `f_e`는 eclipse illumination factor, `n_i^B`는 CSS head boresight다. FOV, Kelly factor, ADC quantization/fault는 sensor layer에서 반영한다.

### 5.2 Reconstructed body sun vector

Basilisk CSS WLS 또는 동등한 estimator로

\[
y_s^B=\operatorname{normalize}(\hat s_{WLS}^B)
\]

를 생성한다. core MEKF는 unit vector의 2D tangent residual을 사용한다.

### 5.3 Specification table

| 항목 | 내용 |
|---|---|
| truth quantity | inertial Sun vector, body attitude, eclipse factor, CSS head geometry |
| measurement equation | raw cosine/FOV response → WLS body unit sun vector |
| output dimension/unit | raw: `N_CSS` ADC/count; core update: unit vector `3`, residual `2` rad-equivalent |
| baseline error | head gain/bias calibrated, WLS angular scatter |
| time-varying error | partial illumination, head-specific noise/gain drift |
| calibration/model error | boresight, gain, bias, nonlinear cosine response |
| outlier/invalidity | total eclipse, insufficient illuminated heads, blinding/FOV, stuck/saturated head |
| rate/latency | raw 10 Hz, WLS output 5 Hz representative |
| estimator role | body sun-vector measurement update; invalid이면 skip |
| context role | WLS residual/condition number, illuminated head count, eclipse/validity allowed; true sun/error scale oracle only |
| source status | CSS/WLS/eclipse architecture [문헌 E1]; numeric geometry/noise TBD |

### 5.4 R determination

제품 수치가 없을 때 `R_s`를 임의 각도 하나로 선언하지 않고 다음 순서로 정한다.

1. CSS raw noise/gain/FOV profile을 명시
2. attitude/sun-angle grid에서 Monte Carlo WLS 수행
3. tangent residual covariance `R_s(geometry)` 추정
4. Tier 0은 대표 고도각 구간의 fixed covariance 사용
5. Tier 1에서 condition/illumination에 따른 oracle/estimated scale 적용

## 6. Star tracker

### 6.1 Measurement equation

Known mounting을 body attitude로 보정한 quaternion packet:

\[
q_{ST,NB}=q_{true,NB}\otimes\operatorname{Exp}_q(\eta_{ST}),
\qquad \eta_{ST}\sim\mathcal N(0,R_{ST}).
\]

packet은 scalar-first로 canonicalize하지 않아도 되지만 update 전에

\[
\text{if }\hat q^Tq_{ST}<0,\qquad q_{ST}\leftarrow -q_{ST}.
\]

처럼 estimate와 같은 hemisphere로 정렬한다.

### 6.2 Specification table

| 항목 | 내용 |
|---|---|
| truth quantity | true `q_NB`, ST mounting, availability/inlier state |
| measurement equation | right-multiplicative small-angle noise on body attitude quaternion |
| output dimension/unit | quaternion `4`, unitless; MEKF residual `3`, rad |
| baseline error | zero-mean tangent Gaussian inlier noise |
| time-varying error | inlier accuracy degradation, tracking quality change |
| calibration/model error | boresight/mounting misalignment, time-tag offset |
| outlier/invalidity | outage, invalid flag, false solution, blinding |
| rate/latency | 1 Hz, Tier-1 0.1 s representative delay; hardware TBD |
| estimator role | low-rate absolute-attitude measurement update |
| context role | quality flag, age, innovation allowed; true accuracy/outlier label oracle only |
| source status | Basilisk quaternion sensor/noise/timestamp [문헌 E1]; outage/false-solution wrapper [결정] |

### 6.3 Latency handling

- Tier 0: zero latency, measurement time=arrival time.
- Tier 1: packet에 original time tag를 유지한다.
- 권장 baseline: state/covariance buffer로 measurement time까지 rollback/repropagate 또는 fixed-lag update.
- **금지:** old measurement를 현재 시각 measurement로 가장하여 update.

## 7. Temperature channel

\[
T_m=T_{true}+b_T^{sensor}+v_T.
\]

| 항목 | 내용 |
|---|---|
| truth quantity | sensor-local temperature profile |
| measurement equation | bias/noise가 있는 scalar temperature |
| output dimension/unit | `1`, °C 또는 K; calibration domain과 단위 명시 |
| baseline error | slow sensor noise/bias |
| time-varying error | thermal lag, spatial gradient |
| calibration/model error | gyro bias-temperature map mismatch |
| outlier/invalidity | implausible range/stuck channel |
| rate/latency | representative 1 Hz |
| estimator role | 직접 attitude update 없음; gyro mean-bias calibration |
| context role | onboard auxiliary feature 허용 |
| source status | profile/수치 TBD; 실제 chamber test 우선 |

**[결정]** temperature 그 자체를 `Q/R`로 직접 해석하지 않는다. 먼저 `\hat b_{g,T}(T_m)`를 제거하고, 남는 불확실성만 context target에 반영한다.

## 8. Optional accelerometer / vibration proxy

Raw specific-force model:

\[
y_a^{S_a}=A_a C_{S_aB}f_{specific}^B+b_a+v_a+v_{vib}.
\]

- **[결정]** 궤도상에서 accelerometer를 지상 중력방향 자세센서처럼 사용하지 않는다.
- core attitude measurement update에는 넣지 않는다.
- 가능한 context feature: high-pass RMS, spectral-band energy, jerk, saturation flag.
- true vibration label이나 injected event ID는 oracle only다.

## 9. Sensor scenario 명세

### 9.1 `UNIT-ST`

| 항목 | 설정 |
|---|---|
| 구성 | gyro 100 Hz + ST 1 Hz |
| 목적 | quaternion convention, bias observability/convergence, async update, latency unit |
| baseline errors | gyro white + bias/RW, ST tangent Gaussian |
| 제외 | mag/sun/environment observability, neural model |
| primary outputs | attitude geodesic error, bias error, norm, `P`, ST innovation/NIS |

### 9.2 `MAIN-FUSION`

| 항목 | 설정 |
|---|---|
| 구성 | gyro + mag + CSS→WLS sun + low-rate ST |
| 목적 | sensor-specific reliability와 process/measurement context |
| normal availability | mag/sun/ST 모두 timestamped; eclipse에서 sun invalid |
| Tier-1 events | gyro drift/noise, vibration burst, mag interference, ST outage/degradation, overlapping slow+fast |
| primary outputs | sensor-specific innovation/NIS, gate, attitude/bias, recovery |

### 9.3 `STRESS-MAG`

| 항목 | 설정 |
|---|---|
| 구성 | gyro + magnetometer |
| 목적 | 제한된 instantaneous observability와 long-horizon/OOD robustness |
| 제한 | 한 시점 vector Jacobian rank 2; orbit/motion profile에 의존 |
| 금지된 주장 | neural model이 관측 불가능한 자유도를 센서 정보만으로 복원했다는 해석 |
| 필수 randomization | orbit phase, initial attitude/rate, magnetic profile, event timing |

## 10. Parameter provenance template

각 numeric parameter는 아래 priority와 metadata를 가진다.

1. 실측 characterization
2. Allan deviation / static / rate-table / chamber test
3. exact product·suffix의 official datasheet
4. official application note
5. peer-reviewed literature range
6. researcher-set representative assumption

```yaml
parameter:
value:
unit:
source_level: measured|allan|datasheet|appnote|literature|assumption
source_id:
product_and_suffix:
temperature_and_range:
uncertainty:
sensitivity_sweep:
```

## 11. 제품 확정 후 교체할 parameter profile

| sensor | 제품이 정해지면 교체·확정할 항목 | Phase 1 representative 허용 | 결과 민감도 |
|---|---|---|---|
| gyro/IMU | full-scale range, output rate/bandwidth, noise density/ARW, bias instability/RW/GM, warm-up, temperature map, scale/misalignment, saturation/quantization, timestamp | rate는 현재 representative 사용; 오차 크기는 dimensionless normalized base profile 가능 | attitude drift·bias convergence·context 필요성에 매우 높음 |
| magnetometer | range, noise, hard/soft-iron calibration, mounting, temperature, MTQ/current coupling, saturation/fault semantics | rate와 normalized covariance profile 가능 | STRESS-MAG와 interference gate에 매우 높음 |
| CSS/sun | head count/boresight, FOV, cosine response, gain/bias, ADC, saturation, WLS algorithm/quality | ideal vector unit test + abstract CSS constellation 가능 | eclipse/FOV availability와 `R_s`에 높음 |
| star tracker | attitude accuracy/covariance axes, output convention, rate, latency, FOV/blinding, quality flags, lost-in-space/reacquisition, false-solution behavior | 1 Hz/0.1 s와 normalized tangent covariance를 assumption으로 사용 가능 | UNIT-ST convergence와 outage/recovery에 매우 높음 |
| temperature | location, rate, accuracy, lag, synchronization | slow scalar profile 가능 | calibration residual과 `α_b` 식별에 중간~높음 |
| accelerometer | bandwidth/range/noise, mounting, actuator vibration coupling | context proxy를 비활성화한 채 진행 가능 | optional detector 성능에만 영향 |

**[결정]** 제품 교체는 measurement equation과 estimator role을 바꾸지 않는 `sensor_profile` 교체로 설계한다. 단, output convention·frame·latency semantics가 다르면 adapter와 관련 convention tests를 반드시 다시 실행한다.

## 12. Gate

- [ ] sensor mean/variance/PSD가 config와 통계적으로 일치한다.
- [ ] `S_j→B` mounting transform이 basis-vector test를 통과한다.
- [ ] ST `q`와 `-q` packet이 동일 update를 만든다.
- [ ] eclipse/FOV/outage에서 invalid update가 실행되지 않는다.
- [ ] latency packet을 현재 시각으로 잘못 적용하지 않는다.
- [ ] MTQ/interference true state가 deployable input으로 누출되지 않는다.
- [ ] 실제 numeric parameter마다 source-level metadata가 있다.
