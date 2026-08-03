# Code CLI Agent Prompt 4 — Package C Problem-Existence and Oracle Tests

이 프롬프트는 Package B와 C1이 Chat에서 승인된 뒤 실행한다.

---

당신은 Phase 1A Package C 구현 담당자다. 아직 ANN, KalmanNet, Split-KalmanNet, SNN을 학습하지 않는다.

## 먼저 읽을 문서

- `docs/research/phase0a/decision_lock/P0_07_CONTEXT_CONTRACT.md`
- `docs/research/phase0a/decision_lock/P0A_IMMEDIATE_TEST_SPEC.md`
- `experiments/phase1a/reports/P1A_UNIT_ST_REPORT.md`
- `experiments/phase1a/reports/P1A_C1_BASELINE_REPORT.md`
- `docs/research/phase1a/P1A_GATE_REPORT.md`

## 구현할 estimator

- `F-MIS`
- `F-TUNED`
- `ORACLE-QR`
- outlier가 있는 경우 `ROBUST`
- 진단용 `WRONG-SIDE`

## 구현할 실험

- C2 gyro process-uncertainty step
- C3 ST 또는 magnetometer measurement-reliability step
- C5 비슷한 innovation RMS를 갖는 process/measurement A/B pair
- C4 slow drift + fast event simultaneous

각 실험은 no-noise analytic 또는 controlled single-seed debug 후 Monte Carlo로 확장한다. 모든 estimator는 같은 truth/sensor realization을 사용한다.

## 금지

- learned context estimator
- future event time 사용
- ORACLE label을 deployable feature 파일에 포함
- F-TUNED가 inference 중 event time을 아는 것
- result가 나쁘다는 이유로 profile을 조용히 변경
- primary metric을 component-wise quaternion MSE로 대체

## 필수 report

- `experiments/phase1a/reports/P1A_PROBLEM_EXISTENCE_REPORT.md`
- `experiments/phase1a/reports/P1A_IDENTIFIABILITY_REPORT.md`
- `experiments/phase1a/reports/P1A_ORACLE_USEFULNESS_REPORT.md`
- updated `docs/research/phase1a/P1A_GATE_REPORT.md`

## 최종 판단 표

| 가설 | 지지 evidence | 반대 evidence | 현재 결론 | 다음 단계 영향 |

반드시 다음을 판단한다.

1. time-varying uncertainty adaptation이 실제로 필요한가
2. oracle context가 fixed/tuned baseline보다 유용한가
3. process-side와 measurement-side context를 분리할 필요가 있는가
4. innovation-only input의 구조적 한계가 관찰되는가
5. robust gate가 R inflation과 별도로 필요한가
6. ANN context 단계로 진행할 근거가 있는가

## 완료 출력

- diff stat와 changed files
- experiment config/seed/manifest 목록
- paired result summary
- Gate pass/fail
- ANN/KalmanNet/SNN 단계 진행 또는 중단 권고

---
