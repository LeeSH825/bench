# P0_00 Evidence Register

> 작성일: 2026-07-30
> 표지 규칙: **[확인]** 실험·실측, **[문헌]** 논문·공식 자료, **[분석]** 수식·구조 해석, **[가설]** 검증 대상, **[결정]** 설계 선택, **[보류]** 근거 부족·후속 범위

| 항목 | 내용 |
|---|---|
| 목적 | 현재 연구에서 확인된 결과, 문헌 구조, 미구현 가설, 유지해야 할 방향과 Phase 0A의 새 결정을 하나의 증거 장부로 분리한다. |
| 입력 근거 | S0–S4, E1–E4 |
| 확인된 내용 | 불규칙 IMU 결측이 핵심 문제가 아니며, 단순 measurement enhancement에서 뚜렷한 최종 자세 추정 개선이 확인되지 않았음 |
| 미확인 내용 | 과거 실험의 정량 로그·seed·configuration, dual-timescale/context/SNN 가설의 실제 효과 |
| 결정 상태 | LOCK — 증거 등급과 현재 상태 고정 |
| 남은 TBD | 과거 실험의 원시 로그·seed·정량 결과, 정확한 Handoff(1) 원본 |
| 다음 Gate | Phase 1에서는 모든 새 결과를 동일 형식으로 append하고, 가설을 [확인]으로 승격할 때 artifact 경로를 기록 |


## 1. 실제 수행·관찰

| ID | 내용 | 표지 | 근거 | 현재 해석 | 상태 |
|---|---|---|---|---|---|
| EV-001 | 실제 센서 데이터에서 불규칙한 IMU 결측이 핵심 문제로 관찰되지 않음 | [확인] | S0, S1, S2 | missing-aware architecture를 정당화하지 못함 | LOCK |
| EV-002 | 단순 measurement enhancement를 Split-KalmanNet 앞에 붙인 시험에서 뚜렷한 최종 자세 추정 개선이 확인되지 않음 | [확인] | S0, S1, S2 | sensor-domain MSE와 state-domain 성능은 동일 목표가 아님 | LOCK |
| EV-003 | 과거 Split-KalmanNet 계열 baseline과 noise 변화 시험 경험이 있음 | [확인] | S1 | 현재 MEKF shell과 동일 조건인지 확인되지 않아 legacy evidence로만 사용 | PROVISIONAL |
| EV-004 | 과거 결과의 정확한 수치·seed·trajectory manifest는 현재 산출물에 보존되어 있지 않음 | [확인] | S1/S2의 명시적 한계 | 정량 성능을 재구성하거나 논문 결과처럼 인용하지 않음 | LOCK |

## 2. 문헌이 직접 지지하는 구조

| ID | 문헌 내용 | 표지 | 본 연구에서 허용되는 해석 | 허용되지 않는 해석 |
|---|---|---|---|---|
| LT-001 | KalmanNet은 상태공간 구조 일부를 유지하면서 neural module로 Kalman gain을 학습 | [문헌] E3 | direct gain-learning baseline | 어떤 환경에서도 classical KF보다 우월하다는 일반화 |
| LT-002 | Split-KalmanNet은 Jacobian과 두 recurrent factor를 이용해 gain을 구성 | [문헌] E3 | split latent factor backbone | 각 factor가 실제 `Q`와 `R`을 유일하게 복원한다는 주장 |
| LT-003 | Adaptive KalmanNet은 context-dependent modulation으로 분포 변화에 적응 | [문헌] E3 | oracle/estimated context 비교의 선행 근거 | scalar context가 본 위성 다중센서 문제에 충분하다는 보장 |
| LT-004 | MEKF/ESKF는 quaternion nominal state와 3D local attitude error를 곱셈으로 결합하고 update 후 reset | [문헌] E2 | 6D tangent covariance, multiplicative injection | quaternion 4개 성분을 독립 additive Gaussian state로 취급 |
| LT-005 | Basilisk는 spacecraft, IMU, magnetometer, CSS, star tracker, eclipse, WMM 등의 공식 모듈 제공 | [문헌] E1 | truth/sensor baseline 구축에 사용 | 모든 연구 오차가 기본 모듈에 이미 구현되어 있다는 주장 |

## 3. 아직 구현·검증하지 않은 가설

| ID | 가설 | 필요한 최소 검증 | 실패 시 조치 | 상태 |
|---|---|---|---|---|
| HY-001 | time-varying gyro process uncertainty와 외부센서 reliability가 fixed MEKF를 의미 있게 악화시킨다 | Package C step tests | adaptation 연구 범위 축소 | TBD-BY-EXPERIMENT |
| HY-002 | 정확한 oracle Q/R/gate가 transition peak와 recovery를 개선한다 | oracle-scaled MEKF | context estimator 학습 중단 | TBD-BY-EXPERIMENT |
| HY-003 | process-side와 measurement-side context를 분리해야 한다 | matched innovation-RMS A/B pair | scalar/shared context 유지 | TBD-BY-EXPERIMENT |
| HY-004 | slow drift와 fast event를 분리한 dual-timescale context가 유리하다 | slow/fast/combined ablation | single-timescale 유지 | TBD-BY-EXPERIMENT |
| HY-005 | onboard feature로 oracle 성능의 의미 있는 비율을 복원할 수 있다 | ANN context estimator | oracle-only 분석으로 한정 | TBD-BY-EXPERIMENT |
| HY-006 | SNN fast detector가 동등 ANN/classical detector 대비 기능적 이점이 있다 | detection latency, event count, accuracy 비교 | SNN 제외 | TBD-BY-EXPERIMENT |
| HY-007 | Split branch가 intervention별로 안정된 역할 분화를 보인다 | branch ablation/scale test | latent baseline으로만 기록 | TBD-BY-EXPERIMENT |

## 4. Abstract/연구계획 때문에 유지할 큰 방향

- **[결정]** 상위 대상은 저가 IMU를 포함한 ADCS 센서 기반 위성 자세 추정이다.
- **[결정]** 기계학습 기반 sensor compensation과 adaptive neural filtering의 연계를 연구하되, 단순 denoiser cascade로 한정하지 않는다.
- **[결정]** 최종 적용성은 동일 controller를 사용한 closed-loop 자세제어 시험으로 확인한다.
- **[결정]** 실제 저가/기준 IMU 자산과 Basilisk simulation을 단계적으로 연결한다.
- **[보류]** SNN·FPGA는 알고리즘 가설 통과 전 main contribution으로 확정하지 않는다.

## 5. Phase 0A에서 새로 잠그는 결정

| ID | 결정 | 표지 | 이유 | 연결 문서 |
|---|---|---|---|---|
| DL-001 | Kinematic MEKF, nominal `[q_NB,b_g]`, error `[δθ,δb_g]` | [결정] | attitude uncertainty와 gyro bias를 최소 상태로 분리 | P0_01, P0_05 |
| DL-002 | scalar-first Hamilton, body-to-inertial active, right error | [결정] | 부호·frame을 unit-test 가능한 단일 계약으로 고정 | P0_05 |
| DL-003 | UNIT-ST / MAIN-FUSION / STRESS-MAG | [결정] | 단위 검증, 주 연구, 제한 관측성 분리 | P0_03, P0_04 |
| DL-004 | mean→state/calibration, stochastic→Q/R, gross→gate | [결정] | 원인과 filter target 중복 방지 | P0_04 error catalog |
| DL-005 | Split direct gain baseline + structured adaptive candidate | [결정] | 연속성과 해석성 동시 보존 | P0_06 |
| DL-006 | oracle event-local context 후 ANN, 필요한 fast channel만 SNN | [결정] | 가설 단계 순서 준수 | P0_07 |
| DL-007 | trajectory-level split와 identical pre-generated measurement | [결정] | leakage·불공정 비교 방지 | P0_01/P0A test |

## 6. 확인되지 않은 정보

- 실제 저가 IMU의 Allan deviation, bias-temperature curve, scale/misalignment matrix
- 기준 IMU·magnetometer·sun sensor·star tracker의 제품과 정확도
- 실제 ADCS bus mass/inertia, sensor mounting, actuator telemetry rate
- 과거 Split-KNet 실험의 원시 결과와 exact configuration
- oracle context가 실제로 성능을 개선하는지
- ANN/SNN이 oracle context를 식별할 수 있는지

## 7. 증거 승격 규칙

`[가설] → [확인]` 승격에는 다음이 모두 필요하다.

1. immutable config/commit ID
2. trajectory manifest와 seed
3. 동일 sensor realization을 사용한 baseline 비교
4. metric definition과 단위
5. raw log 또는 재생 가능한 artifact
6. 실패 사례와 OOD 결과
