# P0_03 위성·궤도·환경 Truth Model 명세

> 작성일: 2026-07-30  
> 표지 규칙: **[확인]** 실험·실측, **[문헌]** 논문·공식 자료, **[분석]** 수식·구조 해석, **[가설]** 검증 대상, **[결정]** 설계 선택, **[보류]** 근거 부족·후속 범위

| 항목 | 내용 |
|---|---|
| 목적 | Basilisk에서 재현할 1차 spacecraft/orbit/environment truth와 Tier 0–2 난도 도입 순서를 고정한다. |
| 입력 근거 | S0–S4, E1; `P0_02_TRUTH_SENSOR_ESTIMATOR_BOUNDARY.md` |
| 결정 상태 | LOCK — frame, truth/estimator 분리, Tier 순서; PROVISIONAL — representative bus/orbit 수치 |
| 남은 TBD | 대상 위성 CAD·mass properties, 실제 궤도, sensor/actuator mounting, mission maneuver profile |
| 다음 Gate | 동일 config/seed에서 truth trajectory와 모든 sensor output을 재생성하고 analytic Tier-0 case를 통과 |


## 1. Truth model의 역할

**[결정]** truth는 estimator와 동일한 kinematic model을 복사하는 단순 signal generator가 아니라, 다음을 독립적으로 생성하는 기준층이다.


\[
\mathcal T_t=\{r^N,v^N,q_{NB},\omega^B,\dot\omega^B,
T,\tau_{cmd},\tau_{dist},m^N,s^N,\chi_{eclipse},\text{sensor hidden states}\}.
\]

Estimator는 이 중 sensor packet과 허용된 nominal model/telemetry만 본다.

## 2. 좌표계와 강체 convention

| 기호 | 정의 | 축/원점 | 사용 |
|---|---|---|---|
| `N` | inertial reference frame | Phase 1에서는 Earth-centered inertial 계열 | orbit, sun, attitude truth |
| `B` | spacecraft body frame | CoM 원점, nominal principal axes | attitude state, angular rate, torque, inertia |
| `S_j` | sensor `j` frame | 센서 proof-mass/optical axes | raw sensor model |
| `A_k` | actuator `k` frame | wheel/MTQ mounting axes | telemetry와 disturbance proxy |

일반 DCM `C_AB`는 B-frame 성분을 A-frame 성분으로 변환한다. 자세 state `q_NB`는 B → N 능동 회전이며 `C_BN(q)=R_NB(q)^T`이다.

**[결정]** body frame은 우선 principal-axis frame으로 두어

\[
J_B=\operatorname{diag}(J_x,J_y,J_z)
\]

를 사용한다. product of inertia는 실제 CAD가 들어오는 hardware-specific profile에서만 활성화한다.

## 3. 대표 위성 크기·질량·관성모멘트 결정 방법

### 3.1 권장 기본안

- **[결정]** 제품 독립 `6U-class parameterized rigid cuboid`를 기본 profile로 둔다.
- **[보류]** 특정 제조사 bus의 mass/inertia를 사실처럼 채우지 않는다.
- **[결정]** config에는 `mass`, `Lx,Ly,Lz`, optional `J_B`를 모두 저장한다. `J_B`가 없으면 균질 cuboid 식으로 생성한다.

\[
J_x=\frac{m}{12}(L_y^2+L_z^2),\quad
J_y=\frac{m}{12}(L_x^2+L_z^2),\quad
J_z=\frac{m}{12}(L_x^2+L_y^2).
\]

| 값 | 현재 상태 | 필요한 자료 | 획득 방법 | 임시 representative 사용 | 민감도 |
|---|---|---|---|---|---|
| mass `m` | TBD-BY-HARDWARE | 대상 bus mass budget | CAD/질량측정/시스템 budget | 가능; config value로 명시 | dynamic truth와 GG torque에 중간~높음, kinematic MEKF 수식에는 낮음 |
| dimensions `Lx,Ly,Lz` | TBD-BY-HARDWARE | bus envelope | CAD/기구도 | 가능 | inertia와 aero/SRP torque에 높음 |
| inertia `J_B` | TBD-BY-HARDWARE | mass properties | CAD 또는 pendulum test | cuboid 근사 가능 | maneuver/angular acceleration에 높음 |
| CoM offset | Tier 2 | mass property | CAD/test | Phase 1A는 0 | force-induced torque에 높음 |

**[결정]** sensitivity profile은 nominal inertia와 각 축 `×0.5`, `×2` perturbation을 포함한다. 이 값은 hardware uncertainty의 사실값이 아니라 estimator가 dynamics를 사용하지 않는 조건에서도 truth motion diversity를 확인하기 위한 연구자 설정이다.

## 4. Orbit와 environment

### 4.1 Orbit 종류

**[결정: PROVISIONAL]** 1차 benchmark는 circular near-polar LEO:

```yaml
altitude: 550 km          # representative research setting, mission fact 아님
eccentricity: 0
inclination: 97.6 deg     # representative near-polar setting
RAAN: trajectory-randomized
argument_of_perigee: 0    # circular orbit에서 기준 convention
true_anomaly_0: trajectory-randomized
```

- Tier 0: two-body Earth gravity로 orbit state를 전파하거나 짧은 unit case에서는 prescribed position을 사용한다.
- Tier 1: Earth rotation, SPICE Sun ephemeris, Basilisk WMM, eclipse를 사용한다.
- Tier 2/OOD: altitude, inclination, RAAN, eccentricity, initial phase를 train 범위 밖으로 이동한다. 장기간 궤도정밀도가 연구 결과를 제한할 때 J2/spherical harmonics를 추가한다.

### 4.2 Orbit initial phase randomization

- `RAAN ~ U(0,2π)`와 `ν_0 ~ U(0,2π)`를 trajectory 단위로 뽑는다.
- 동일 trajectory의 모든 estimator는 같은 orbit과 sensor realization을 공유한다.
- train/validation/test는 seed와 orbit phase가 겹치지 않는다.
- OOD-orbit test는 IID 범위와 별도 manifest를 사용한다.

### 4.3 Sun, eclipse, magnetic field

- **[문헌]** Basilisk의 eclipse module은 spacecraft/planet/Sun geometry에서 illumination factor를 제공하고, WMM module은 위치·시간 기반 Earth magnetic field를 계산할 수 있다.
- **[결정]** true inertial Sun vector는 SPICE/ephemeris 기반으로 생성한다.
- **[결정]** eclipse factor `f_e∈[0,1]`를 truth에 저장한다. CSS output은 이 factor와 각 head의 FOV를 반영한다.
- **[결정]** Earth magnetic truth는 WMM을 기본으로 한다. Tier 0 analytic Jacobian test는 고정 inertial vector를 사용한다.
- **[결정]** estimator는 Tier 0에서 truth와 같은 reference vector를 사용하고, Tier 1 mismatch에서는 update time/position quantization, 낮은 차수 model 또는 additive model error를 명시적으로 준다.

## 5. Initial attitude·angular-rate distribution

### 5.1 Truth initial state

| profile | true `q_NB,0` | true `ω_0^B` | 용도 |
|---|---|---|---|
| `analytic-zero` | identity | 0 | zero-motion unit test |
| `analytic-rate` | identity | 지정 축의 constant rate | propagation unit test |
| `nominal-random` | Haar-uniform on SO(3) | 방향 uniform, magnitude `U(0,1 deg/s)` | 일반 trajectory diversity |
| `stress-rate` | Haar-uniform on SO(3) | magnitude `U(1,5 deg/s)` | high-rate/OOD |

수치 범위는 **[결정: representative]**이며 실제 임무 운용률이 정해지면 교체한다.

### 5.2 Estimator initial error

- nominal: random axis + angle `U(0,10 deg)`
- moderate: `U(10,60 deg)`
- large-error regression: fixed `120 deg`
- sign/antipodal test: `q`와 `-q`
- near-π stress: `179 deg`는 별도 reset/initialization test로만 사용

**[분석]** small-error covariance를 사용하는 MEKF의 정상 update는 local regime을 전제로 한다. 큰 초기오차 시험은 필터의 global initialization 보장을 의미하지 않으며, 필요하면 coarse attitude initialization을 별도 전처리로 둔다.

## 6. Attitude motion profile

### Tier-0 prescribed/analytic profiles

1. zero motion
2. single-axis constant angular rate
3. known constant gyro bias with zero/constant rate
4. piecewise-smooth rate profile
5. small commanded slew with exact truth

### Tier-1 rigid-body profiles

강체 방정식:

\[
J\dot\omega^B+\omega^B\times J\omega^B
=\tau_{cmd}^B+\tau_{dist}^B+\tau_{env}^B.
\]

- coast segment
- known commanded torque/slew segment
- unknown bounded torque pulse
- slow thermal drift와 fast sensor event가 겹치는 segment

**[결정]** Kinematic MEKF는 torque/inertia를 propagation에 사용하지 않는다. `τ_cmd`, wheel speed, MTQ state는 실제로 알려진 경우 context 보조 입력일 뿐이며, `τ_dist`는 hidden truth다.

## 7. Commanded maneuver와 unknown disturbance의 구분

| 항목 | Truth | Estimator/Context | 평가 |
|---|---|---|---|
| commanded torque/slew schedule | 정확히 기록 | 실제 telemetry가 있다고 가정한 profile에서만 입력 허용 | maneuver 중 error·recovery |
| control law internal state | closed-loop 단계에서만 | Phase 1에는 없음 | 최종 적용성 시험 |
| unknown disturbance torque | truth hidden state | 직접 입력 금지 | process mismatch/event label |
| sensor vibration caused by actuator | 별도 sensor-error injection | wheel speed/command만 보조 feature | detector latency/reliability |

## 8. Actuator telemetry 정책

- **Phase 1A UNIT-ST:** actuator model/telemetry 없음.
- **MAIN-FUSION normal:** commanded slew flag/torque를 optional onboard-known context로 기록한다.
- **magnetorquer scenario:** MTQ on/off/current가 실제 OBC에서 알려진다는 조건에서 magnetometer gate feature로 허용한다.
- **reaction-wheel scenario:** wheel speed는 context feature 가능; full wheel imbalance force/torque는 Tier 2 truth-only parameter다.

## 9. Truth–estimator 의도적 mismatch

| mismatch ID | Truth | Estimator | Tier | 목적 |
|---|---|---|---|---|
| M0 | 동일 reference vector/model | 동일 | 0 | 수학/코드 검증 |
| M1 | time-varying gyro white noise | fixed `S_g0` | 1 | process-scale adaptation 필요성 |
| M2 | time-varying bias RW | fixed `S_b0` | 1 | slow context 필요성 |
| M3 | WMM field + interference | nominal field, fixed `R_mag` | 1 | measurement reliability/gate |
| M4 | eclipse/FOV invalid CSS | validity-aware estimator | 1 | update skip 확인 |
| M5 | ST outage/latency/degraded inlier noise | fixed nominal `R_ST` | 1 | async reliability adaptation |
| M6 | scale/misalignment | nominal identity | 2 | calibration/model-error OOD |
| M7 | false ST solution/heavy tail | Gaussian inlier model | 2 | robust gate |
| M8 | orbit/reference-model OOD | nominal orbit/model | 2 | model mismatch robustness |

## 10. 외란 도입 순서

### Tier 0 — MEKF 기본 검증

- prescribed 또는 simple rigid-body truth
- gyro white noise
- constant gyro bias 또는 bias RW
- nominal mag/sun/ST Gaussian inlier noise
- exact sensor alignment
- zero latency 또는 단순 multirate update
- no physical environmental torque required

### Tier 1 — 핵심 연구 조건

- gradual gyro bias drift / bias-RW scale change
- temperature-dependent mean bias + imperfect calibration residual
- time-varying gyro noise floor and vibration burst
- magnetometer interference
- star-tracker outage, latency, inlier accuracy degradation
- slow + fast event overlap
- **gravity-gradient torque:** inertia가 고정된 rigid-body trajectory부터 포함
- **generic bounded unknown torque pulse:** process/event 구분 시험에 포함
- **reaction-wheel vibration:** full structural model 대신 sensor variance-burst proxy

### Tier 2 — OOD·고난도

- scale factor, axis misalignment, cross-axis sensitivity
- heavy-tailed noise, saturation, quantization edge cases
- false star-tracker solution
- unseen timing/magnitude/combination
- orbit/initial-condition OOD
- aerodynamic torque, SRP torque, residual magnetic-dipole torque
- full reaction-wheel imbalance/flexible vibration model

### Phase 1 외란 채택 판단

| 외란 | Phase 1A | Phase 1 main/Tier 1 | 후속 Tier 2 | 이유 |
|---|---:|---:|---:|---|
| gravity-gradient | 제외 | 포함 | — | inertia가 정해지면 재현 가능하고 저주파 motion diversity 제공 |
| aerodynamic torque | 제외 | 제외 | 포함 | geometry·density·CoM/CoP 의존성이 큼 |
| solar-radiation-pressure torque | 제외 | 제외 | 포함 | optical geometry·CoP 자료 필요 |
| residual magnetic dipole | 제외 | proxy만 가능 | 포함 | 실제 dipole/MTQ 자료 필요 |
| reaction-wheel vibration | 제외 | sensor-noise burst proxy | full model | 구조/imbalance parameter 없이는 허위 정밀성 위험 |
| generic unknown torque pulse | 제외 | 포함 | — | process/event 식별 시험을 가장 통제 가능하게 구성 |

## 11. Basilisk 구성 블록

**[문헌/결정]** 공식 Basilisk 모듈을 다음과 같이 조합한다.

```text
spacecraft rigid body
  + Earth gravity/orbit propagation
  + SPICE Sun/planet ephemeris
  + WMM magnetic field
  + eclipse illumination
  + optional gravity-gradient / extForceTorque
  → sensor modules / custom error wrappers
```

온도 profile, temperature-dependent bias, magnetic interference, false ST solution, event-local noise scale은 기본 sensor module 바깥의 deterministic/replayable error-injection wrapper로 둔다.

## 12. Reproducibility manifest

각 trajectory는 최소 다음을 저장한다.

```yaml
scenario_id:
tier:
software_version:
truth_seed:
sensor_seed_by_type:
orbit_elements:
spacecraft_mass_properties:
initial_q_NB_true:
initial_omega_B_true:
maneuver_schedule:
disturbance_schedule:
temperature_profile_id:
sensor_profile_ids:
event_intervals:
```

## 13. Gate

- [ ] zero/constant-rate analytic truth와 quaternion propagation이 일치한다.
- [ ] rigid-body no-torque case에서 angular momentum/energy sanity check가 통과한다.
- [ ] WMM/sun/eclipse output frame과 timestamp가 검증된다.
- [ ] commanded vs hidden disturbance가 dataset namespace에서 분리된다.
- [ ] 동일 truth seed로 모든 sensor packet을 bitwise 또는 tolerance 내 재생성한다.
- [ ] Tier 0이 통과하기 전 Tier 1 복합 event를 활성화하지 않는다.
