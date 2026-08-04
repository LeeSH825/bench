# Phase 0A 근거·출처 등록부

> 작성일: 2026-07-30
> 표지 규칙: **[확인]** 실험·실측, **[문헌]** 논문·공식 자료, **[분석]** 수식·구조 해석, **[가설]** 검증 대상, **[결정]** 설계 선택, **[보류]** 근거 부족·후속 범위

| 항목 | 내용 |
|---|---|
| 목적 | 모든 산출물에서 사용하는 업로드 문서와 외부 공식 자료의 범위·한계를 고정한다. |
| 입력 근거 | S0–S4 및 E1–E4 |
| 결정 상태 | LOCK — 출처 계층과 증거 표지 방식 고정 |
| 남은 TBD | 정확한 `AI_ADCS_KalmanNet_Research_Handoff(1).md` 원본의 직접 대조 |
| 다음 Gate | 원본이 추가되면 Evidence Register의 문장 단위 차이만 재검토; 설계 계약은 새 근거가 반박할 때만 변경 |

### 근거 ID

| ID | 자료 | 근거 수준과 사용 범위 |
|---|---|---|
| S0 | `붙여넣은 텍스트 (1).txt` | [결정] 이번 Phase 0A의 사용자 요구, 고정 전제, 출력 계약 |
| S1 | `AI_ADCS_KalmanNet_Research_Handoff` 계열 | [확인]/[가설] 기존 실험·아이디어의 인수인계. 활성 파일시스템에는 정확한 `(1)` 파일이 없어 File Library의 일치 버전과 S2–S4의 교차 확인된 내용만 사용 |
| S2 | `01_AI_ADCS_KalmanNet_Research_Evaluation_Human_Readable(1).md` | [분석] 연구 방향, 위험, 권장 구조 평가 |
| S3 | `02_AI_ADCS_KalmanNet_Research_Action_Guidelines(1).md` | [결정] 용어·비교·모델링 규칙 |
| S4 | `03_AI_ADCS_KalmanNet_Detailed_Phase_Step_Roadmap(1).md` | [결정]/[가설] Phase–Gate 및 실험 순서 |
| E1 | Basilisk official documentation/release, v2.11.1 계열 | [문헌] 현재 지원 모듈·메시지·제약 확인 |
| E2 | F. L. Markley, *Attitude Error Representations for Kalman Filtering*; J. Solà, *Quaternion Kinematics for the Error-State Kalman Filter* | [문헌] MEKF/ESKF의 곱셈 오차, 주입, reset 및 quaternion 수학 |
| E3 | Revach et al., *KalmanNet*; Choi et al., *Split-KalmanNet*; Ni et al., *Adaptive KalmanNet* | [문헌] neural gain, split factor, context modulation의 원 논문 수준 근거 |
| E4 | Movella/Xsens official MTi-2 leaflet 및 MTi documentation | [문헌] 실제 후보 센서의 제조사 사양 출처. 수치는 실측을 대체하지 않음 |


## 출처 사용 규칙

1. [확인]은 실제 데이터·코드 실행·실험 로그가 있는 경우에만 쓴다.
2. 업로드 문서가 “권장”, “가설”, “향후”라고 쓴 내용은 완료된 결과로 승격하지 않는다.
3. 외부 문헌은 구조·API·수학 관례를 지지할 뿐, 이 연구의 성능이나 실제 센서의 품질을 증명하지 않는다.
4. 제조사 typical specification은 실측 characterization보다 낮은 우선순위다.
5. Basilisk 기본 센서 모듈에 없는 온도 의존 오차, 자기 간섭, false star solution은 사용자 정의 주입 계층으로 명시한다.
6. 논문이 Split branch를 covariance라고 부르더라도, 본 연구에서는 식별성·SPD·frame이 검증되기 전까지 `latent covariance-like factor`로 기록한다.

## 근거 충돌 처리

- S0의 고정 전제와 S2–S4의 권고가 일치하면 Phase 0A 설계 결정으로 잠근다.
- Handoff 계열의 수치가 보존되지 않았거나 다른 버전 사이에 차이가 있으면 수치를 재구성하지 않는다.
- 외부 공식 문서가 현재 Basilisk API를 갱신한 경우, API 사실은 최신 공식 문서를 우선하되 연구 의도는 S0–S4를 유지한다.


### 외부 공식/원문 링크

- Basilisk releases: https://github.com/AVSLab/basilisk/releases
- Basilisk spacecraft: https://avslab.github.io/basilisk/Documentation/simulation/dynamics/spacecraft/spacecraft.html
- Basilisk IMU: https://avslab.github.io/basilisk/Documentation/simulation/sensors/imuSensor/imuSensor.html
- Basilisk magnetometer: https://avslab.github.io/basilisk/Documentation/simulation/sensors/magnetometer/magnetometer.html
- Basilisk coarse sun sensor: https://avslab.github.io/basilisk/Documentation/simulation/sensors/coarseSunSensor/coarseSunSensor.html
- Basilisk CSS WLS estimator: https://avslab.github.io/basilisk/Documentation/fswAlgorithms/attDetermination/CSSEst/cssWlsEst.html
- Basilisk star tracker: https://avslab.github.io/basilisk/Documentation/simulation/sensors/starTracker/starTracker.html
- Basilisk eclipse: https://avslab.github.io/basilisk/Documentation/simulation/environment/eclipse/eclipse.html
- Basilisk WMM: https://avslab.github.io/basilisk/Documentation/simulation/environment/magneticFieldWMM/magneticFieldWMM.html
- Markley (NASA NTRS): https://ntrs.nasa.gov/citations/20020060647
- Solà: https://arxiv.org/abs/1711.02508
- KalmanNet: https://arxiv.org/abs/2107.10043
- Split-KalmanNet: https://arxiv.org/abs/2210.09636
- Adaptive KalmanNet: https://arxiv.org/abs/2309.07016
- Xsens MTi-2 official leaflet: https://www.movella.com/hubfs/Downloads/Leaflets/MTi-2.pdf
