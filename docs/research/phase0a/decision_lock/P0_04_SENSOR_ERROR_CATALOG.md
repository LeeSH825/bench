# P0_04 Sensor Error Catalog

> 작성일: 2026-07-30
> 표지 규칙: **[확인]** 실험·실측, **[문헌]** 논문·공식 자료, **[분석]** 수식·구조 해석, **[가설]** 검증 대상, **[결정]** 설계 선택, **[보류]** 근거 부족·후속 범위

| 항목 | 내용 |
|---|---|
| 목적 | 각 센서 오차를 mean/model, stochastic uncertainty, gross invalidity로 분류하고 state/calibration, Q/R, gate 중 처리 위치를 명시한다. |
| 입력 근거 | S0–S4, E1, E4; `P0_04_SENSOR_ROLE_AND_MODEL_SPEC.md` |
| 결정 상태 | LOCK — 분류 원칙과 Tier; TBD-BY-HARDWARE — 수치 |
| 남은 TBD | 실제 sensor characterization, magnetic/thermal/actuator environment, 제품 validity/quality semantics |
| 다음 Gate | 각 error injector가 단독 intervention test를 갖고, filter target과 oracle label이 1:1로 추적 가능 |


## 1. 분류 원칙

```text
mean/model error
  → explicit state, calibration, or deterministic correction
stochastic uncertainty change
  → Q_t or R_t adaptation
outlier/invalidity
  → reliability gate, robust likelihood, update skip/down-weight
```

**[결정]** 한 error source를 여러 mechanism으로 중복 “해결”하지 않는다. 예를 들어 ST false solution은 `R_ST`를 아주 크게 만드는 inlier-noise 문제로 숨기지 않고 outlier/gate로 분리한다.

## 2. Error-to-filter-target matrix

| ID | 센서/원인 | error type | Truth injection | 기본 estimator 처리 | adaptive target | robust/gate | Tier | source status |
|---|---|---|---|---|---|---|---|---|
| G-01 | constant gyro bias | mean | additive `b_g0` | explicit `b_g` state | 없음 | 없음 | 0 | 실측/가정 |
| G-02 | gyro bias random walk | stochastic process | `db=w_b` | nominal `Q_b` | `α_b` | 없음 | 0/1 | Allan/실측 우선 |
| G-03 | gyro Gauss–Markov bias | mean+stochastic model | `db=-b/τ dt+w_b` | model profile | `α_b` residual | 없음 | 1/2 | 실측 필요 |
| G-04 | temperature mean bias | mean/model | `b_T(T)` | calibration map + bias state | calibration residual만 `α_b/α_g` | temp invalid gate 조건부 | 1 | chamber test |
| G-05 | gyro white-noise floor | stochastic | sample noise PSD | nominal `S_g0` | `α_g` | 없음 | 0/1 | Allan/datasheet |
| G-06 | vibration-induced inlier noise | stochastic burst | colored/variance burst | fixed `S_g0` baseline | fast `α_g` | saturation 시 gate | 1 | shaker/actuator test or assumption |
| G-07 | gyro scale factor | model/calibration | multiplicative matrix | offline calibration | 잔여는 model uncertainty; 1차 context 아님 | gross saturation 별도 | 2 | rate table |
| G-08 | gyro axis misalignment/cross-axis | model/calibration | non-diagonal matrix | offline calibration/mounting | 잔여는 OOD | 없음 | 2 | rate table/CAD |
| G-09 | gyro saturation/clipping | gross invalidity | component clip | validity/fault handling | 없음 | propagation fault mode/gate | 2 | datasheet/test |
| G-10 | gyro quantization | stochastic/nonlinear | discretization | sensor model | 필요 시 `α_g` | 없음 | 2 | datasheet |
| M-01 | magnetometer hard-iron bias | mean/calibration | additive vector | calibration | residual model bias; `R`로 숨기지 않음 | 없음 | 2 | calibration test |
| M-02 | soft-iron/scale/non-orthogonality | model/calibration | matrix | calibration | residual OOD | 없음 | 2 | calibration test |
| M-03 | mag white/inlier noise | stochastic measurement | additive noise | nominal `R_mag` | `α_R,mag` | 없음 | 0/1 | datasheet/test |
| M-04 | slowly varying field-model error | model+stochastic | truth/model vector mismatch | nominal model | bounded `α_R,mag` 또는 explicit model augmentation | 없음 | 1/2 | WMM/model study |
| M-05 | MTQ/electrical interference | gross or regime-dependent | additive field/current coupling | known MTQ state로 update skip/down-weight | inlier residual이면 `α_R,mag` | `ρ_mag` | 1 | system test/assumption |
| M-06 | magnetic spike/stuck/saturation | gross invalidity | spike, hold, clip | fault detector | 없음 | `ρ_mag` hard/soft gate | 1/2 | Basilisk fault/test |
| S-01 | CSS head noise | stochastic | count noise | WLS covariance | `α_R,sun` | 없음 | 0/1 | datasheet/test |
| S-02 | CSS gain/bias | calibration | head-specific scale/bias | calibration | residual geometry-dependent `R` | bad head gate | 1/2 | calibration |
| S-03 | CSS boresight/FOV error | model/calibration | geometry mismatch | mounting calibration | residual OOD | validity | 2 | CAD/test |
| S-04 | eclipse | invalidity | illumination→0 | update skip | 없음 | `ρ_sun=0` | 0/1 | truth geometry |
| S-05 | insufficient heads/poor WLS geometry | reliability | low rank/condition | WLS validity | `α_R,sun` | `ρ_sun` | 1 | computed metadata |
| S-06 | CSS saturation/blinding/stuck | gross invalidity | clip/fault | head rejection/WLS validity | 없음 | `ρ_sun` | 2 | datasheet/test |
| ST-01 | ST nominal tangent noise | stochastic measurement | right quaternion perturbation | nominal `R_ST` | `α_R,ST` | 없음 | 0/1 | datasheet/test |
| ST-02 | ST accuracy degradation | stochastic regime | increased inlier covariance | fixed baseline | `α_R,ST` | low quality can gate | 1 | quality/test/assumption |
| ST-03 | ST latency/time-tag offset | timing/model | delayed packet | buffer/repropagation | age can context | stale-packet gate | 1 | interface/test |
| ST-04 | ST mounting misalignment | calibration | fixed rotation | boresight calibration | residual OOD | 없음 | 2 | alignment test |
| ST-05 | ST outage/blinding | invalidity | no packet/invalid | no update | 없음 | `ρ_ST=0` | 1 | scenario/quality |
| ST-06 | false star solution | gross outlier | large wrong quaternion | robust residual test | inlier `R`와 분리 | `ρ_ST` | 2 | hardware/literature/assumption |
| T-01 | temperature sensor noise/bias | auxiliary measurement | scalar error | calibration/filter feature | context confidence 조건부 | implausible/stuck gate | 1/2 | chamber/test |
| A-01 | accelerometer vibration proxy noise | auxiliary | specific-force/noise | feature only | detector input | feature validity | 1 | test/assumption |
| A-02 | accelerometer interpreted as gravity attitude | modeling misuse | — | **사용 금지** | — | — | rejected | orbital physics |

## 3. 핵심 혼동 방지

### 3.1 `b_g`와 `Q_b`

- `b_g(t)`는 특정 trajectory에서 존재하는 bias mean이며 추정 state다.
- `Q_b`는 bias error process가 얼마나 빨리 확산하는지 나타내는 covariance다.
- true `b_g`가 커졌다는 이유만으로 `Q_b`가 반드시 커지는 것은 아니다.
- bias step처럼 RW model이 설명하지 못하는 event는 explicit jump model 또는 temporary `Q_b` scale/gate 실험으로 별도 표지한다.

### 3.2 Temperature mean과 residual uncertainty

\[
b_g(t)=\hat b_T(T_m(t))+b_{res}(t).
\]

- `\hat b_T`: calibration model
- `b_res`: MEKF bias state가 흡수할 mean residual
- residual dynamics의 세기: `Q_b`/`α_b`
- fast vibration sample noise: `S_g`/`α_g`

### 3.3 Magnetometer bias vs noise 증가

- fixed/slow vector offset: calibration/model error
- zero-mean scatter 증가: `R_mag`
- MTQ-on large deterministic interference, spike, saturation: gate/skip
- 세 경우를 모두 “mag reliability 저하” 하나로만 label하면 context 의미가 붕괴한다.

### 3.4 Star tracker 정확도 저하 vs false/outage

- inlier distribution이 넓어짐: `R_ST` scale
- packet 없음/invalid: gate=0
- plausible-looking large wrong quaternion: outlier probability/robust gate
- latency: timestamp handling; 단순 `R` inflation만으로 해결하지 않는다.

## 4. Tier activation rules

### Tier 0

- `G-01`, `G-02`, `G-05`
- `M-03`, `S-01`, `ST-01`
- exact mounting/calibration, valid packets, simple multirate

### Tier 1

- `G-04`, `G-05` time variation, `G-06`
- `M-03` scale change, `M-05`, `M-06` selected
- `S-04`, `S-05`
- `ST-02`, `ST-03`, `ST-05`
- slow + fast overlap

### Tier 2

- scale/misalignment/cross-axis
- heavy-tail, saturation, false ST solution
- product-specific actuator coupling
- unseen timing/magnitude/combination

## 5. Numeric parameter 결정 표준

| 우선순위 | 근거 | 사용 가능 범위 | 주의 |
|---:|---|---|---|
| 1 | 해당 실물 실측 | baseline/validation | setup·temperature·bandwidth 기록 |
| 2 | Allan/static/rate-table/chamber | sensor process profile | confidence interval 포함 |
| 3 | exact official datasheet | provisional profile | typical/max와 조건 구분 |
| 4 | official app note | model form/범위 | 제품/setting 일치 확인 |
| 5 | peer-reviewed literature | stress range | 제품 차이를 사실로 일반화 금지 |
| 6 | researcher assumption | controlled intervention | 반드시 `[가정]` 및 sensitivity sweep |

## 6. Error injector API contract

```text
inject(truth_quantity, time, hidden_state, config, rng)
  -> measurement, validity, quality, hidden_log
```

- hidden_log는 oracle/evaluation namespace로만 보낸다.
- 각 injector는 `off`, `single-error`, `combined` mode를 가진다.
- event onset/offset은 trajectory manifest에 있지만 deployable runner에는 제공하지 않는다.
- random seed는 sensor와 error source별로 분리한다.

## 7. Gate

- [ ] 모든 error가 mean/Q/R/gate 중 주 처리 위치를 가진다.
- [ ] `b_g`, `Q_b`, temperature mean, residual uncertainty를 별도 로그로 검증한다.
- [ ] single-error intervention에서 의도하지 않은 다른 context label이 바뀌지 않는다.
- [ ] outlier event가 inlier covariance label에 중복 encoding되지 않는다.
- [ ] source 없는 numeric 값은 assumption으로 표시되고 sensitivity sweep이 있다.
