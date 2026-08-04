# P0_01 Decision Ledger

> 작성일: 2026-07-30
> 표지 규칙: **[확인]** 실험·실측, **[문헌]** 논문·공식 자료, **[분석]** 수식·구조 해석, **[가설]** 검증 대상, **[결정]** 설계 선택, **[보류]** 근거 부족·후속 범위

| 항목 | 내용 |
|---|---|
| 목적 | Phase 0A의 핵심 설계 질문을 선택지·권장안·파급효과·상태와 함께 잠근다. |
| 입력 근거 | S0–S4, E1–E4 |
| 결정 상태 | LOCK/PROVISIONAL/TBD 상태를 행별로 관리 |
| 남은 TBD | 실제 hardware parameter와 oracle ablation 결과 |
| 다음 Gate | LOCK 항목 승인 후 Phase 1A implementation contract로 변환 |

## Decision Table

| ID | 결정 질문 | 가능한 선택지 | 권장 기본안 | 근거 | 변경 시 파급효과 | 상태 |
|---|---|---|---|---|---|---|
| D01 | 자세 추정 vs 자세제어 범위 | estimator only / controller 공동 제안 | ADCS용 자세 추정이 주 기여; 동일 controller closed-loop는 최종 검증 | S0–S4; estimator 효과를 controller 효과와 분리 | 변경 시 claim·metric·truth actuator 범위 전면 수정 | LOCK |
| D02 | 저가 gyro uncertainty만 vs 외부센서 reliability 포함 | gyro only / gyro+외부센서 | gyro process uncertainty와 mag/sun/ST reliability 모두 | 주 연구 가설이 process/measurement 동시 변화를 포함 | context와 scenario 차원 축소·논문 기여 변경 | LOCK |
| D03 | Kinematic vs Dynamic MEKF | kinematic / rigid-body dynamic | Kinematic MEKF 1차 | gyro propagation으로 sensor uncertainty를 분리; dynamics parameter 의존도 낮음 | state·F/G/Q·truth knowledge 전면 변경 | LOCK |
| D04 | nominal/error state | q only / q+b / q+ω+b | nominal `[q_NB,b_g]`; error `[δθ,δb_g]∈R6` | S0–S4와 MEKF 문헌 | gain/state dimension과 labels 변경 | LOCK |
| D05 | quaternion ordering | scalar-first / scalar-last | scalar-first `[w,x,y,z]` | Hamilton product와 test vector를 명시하기 쉬움 | 모든 conversion·serialization 수정 | LOCK |
| D06 | attitude 방향 | inertial→body / body→inertial | `q_NB`: body→inertial active | 표준 Hamilton kinematics `qdot=1/2 q⊗ω`; body vector는 transpose | Basilisk adapter, 모든 measurement function 부호 변경 | LOCK |
| D07 | active/passive convention | active / passive | `R_NB(q)` active; `C_BN=R_NB^T` coordinate transform | 의미와 행렬 연산을 분리 | frame 문서·test vector 전면 변경 | LOCK |
| D08 | multiplicative error | left / right | right local error `q_true=qhat⊗δq` | gyro local rate와 compact F, Solà local reset 계열 | F/H/injection/reset 부호·순서 변경 | LOCK |
| D09 | 센서별 역할 | 여러 조합 | gyro propagation; mag 3D vector; CSS-derived sun tangent update; ST tangent attitude; T/telemetry context | 물리 역할과 estimator interface 분리 | sensor model·H/R·context input 변경 | LOCK |
| D10 | unit/main/stress 구성 | 임의 단일 구성 / 계층 구성 | UNIT-ST / MAIN-FUSION / STRESS-MAG | 관측성과 failure source를 분리 | dataset·metric matrix 변경 | LOCK |
| D11 | 제품 모델 범위 | 제품 고정 / 완전 추상 | 제품 독립 parameterized model이 core; 실제 제품은 parameter source profile | 제품 미확정 상태의 허위 사양 방지 | 실제 제품 고정 시 parameter manifest만 교체 | PROVISIONAL |
| D12 | sampling 처리 | 강제 동기 / asynchronous multirate | timestamp event queue; 동일 timestamp는 valid measurements stack | 실제 센서 rate/latency와 일치 | runner·buffer·replay 구조 변경 | LOCK |
| D13 | 오차 처리 위치 | 모두 state / 모두 Q/R / 모두 neural | mean→state/calibration; stochastic→Q/R; gross→gate | 식별성과 물리 의미 보존 | state/context dimension 전반 영향 | LOCK |
| D14 | truth와 estimator 지식 | truth 공개 / 완전 blind | 명시적 boundary table; oracle label은 deployment input 금지 | data leakage 방지 | dataset schema·claim 영향 | LOCK |
| D15 | Split와 structured 지위 | Split proposed / structured only | Split direct-gain baseline; structured adaptive proposed 후보 | 기존 연속성과 SPD/consistency를 동시 보존 | 논문 main model은 oracle 결과 후 변경 가능 | LOCK |
| D16 | physical vs latent context | latent only / physical first | physical oracle 우선, latent는 후속 | oracle usefulness를 먼저 검증 | supervision·해석성·stop gate 영향 | LOCK |
| D17 | scalar vs vector context | scalar / dense vector / event-local | scalar baseline→event-local 4-value minimum→필요 시 dense vector | 최소 차원과 독립 sensor intervention 양립 | output head·dataset label 변경 | TBD-BY-EXPERIMENT |
| D18 | shared vs branch/sensor-specific | shared / branch-specific | shared scalar를 먼저 기각/지지 후 process+active-sensor specific | 복잡도를 근거 없이 늘리지 않음 | neural architecture 후속 영향 | TBD-BY-EXPERIMENT |
| D19 | single vs dual timescale | single / dual | single baseline 후 slow `α_b`와 fast `α_g/ρ` 분리 ablation | dual-timescale 필요성 자체가 가설 | state memory·training sequence 변경 | TBD-BY-EXPERIMENT |
| D20 | SNN 1차 기능 | full gain / context / detector | fast reliability/change detector 또는 event updater | 기능적 이유가 가장 명확 | SNN 제외해도 MEKF/proposed 유지 가능 | PROVISIONAL |
| D21 | primary metric/Phase 1 Gate | component MSE / geodesic+consistency | attitude geodesic RMSE/peak/P95, bias RMSE, recovery, divergence, valid NIS/NEES | quaternion sign invariant·filter consistency 포함 | metric scripts와 acceptance 변경 | LOCK |
| D22 | 현재 보류 설계 | NN layer/SNN neuron/FPGA 등 | neural architecture·surrogate gradient·FPGA·energy claim 보류 | S0 explicit scope | 지금 고정하면 premature optimization | REJECTED |
| D23 | data split | window / trajectory | trajectory-level + IID/OOD manifest | timing/orbit leakage 차단 | 모든 기존 split 재생성 필요 | LOCK |
| D24 | sun sensor interface | direct ideal vector / raw CSS only / CSS→WLS | Jacobian unit test는 ideal vector; MAIN은 CSS constellation→WLS sun vector | Basilisk 공식 CSS와 WLS를 연결하면서 MEKF interface 유지 | sensor adapter와 R characterization 영향 | PROVISIONAL |
| D25 | truth orbit/bus default | mission-specific / representative | 6U-class parameterized cuboid; 550 km, 97.6° circular near-polar provisional | mission 미정 상태에서 reproducible benchmark | truth config만 교체; trained data는 재생성 | PROVISIONAL |
| D26 | 외란 도입 | all-at-once / staged | Tier 0 none/prescribed; Tier 1 GG+generic pulse; Tier 2 aero/SRP/dipole/full RW vibration | 원인 식별성 확보 | truth model과 dataset 난도 영향 | LOCK |


## 상태 해석

- `LOCK`: Phase 1 코드와 데이터 schema가 의존하므로 승인 후 변경은 migration으로 취급한다.
- `PROVISIONAL`: representative 설정으로 진행할 수 있으나 hardware/mission 정보가 오면 parameter profile을 교체한다.
- `TBD-BY-EXPERIMENT`: oracle/intervention 결과로만 확정한다.
- `TBD-BY-HARDWARE`: 실측·제품 선택 없이는 확정하지 않는다.
- `REJECTED`: Phase 0A 범위에서 설계 대상으로 삼지 않는다.

## Lock 변경 절차

1. 변경 이유와 새 근거 표지
2. 영향을 받는 equation/config/schema 목록
3. 기존 trajectory 재생성 필요 여부
4. baseline 공정성 영향
5. regression test 재실행 결과
