# AI-ADCS Phase 0A Decision-Lock Package

이 패키지는 AI-ADCS / KalmanNet 계열 위성 자세 추정 연구의 Phase 0A 산출물이다.

## 권장 읽기 순서

1. `P0A_PHASE_0A_SYNTHESIS.md`
2. `P0_00_EVIDENCE_REGISTER.md`
3. `P0_01_DECISION_LEDGER.md`
4. `P0_02_TRUTH_SENSOR_ESTIMATOR_BOUNDARY.md`
5. `P0_03_TRUTH_MODEL_SPEC.md`
6. `P0_04_SENSOR_ROLE_AND_MODEL_SPEC.md`
7. `P0_04_SENSOR_ERROR_CATALOG.md`
8. `P0_05_MEKF_MATH_CONTRACT.md`
9. `P0_05_MEKF_CONVENTION_TEST_VECTORS.md`
10. `P0_06_NEURAL_INSERTION_OPTIONS.md`
11. `P0_07_CONTEXT_CONTRACT.md`
12. `P0A_IMMEDIATE_TEST_SPEC.md`
13. `P0A_REFERENCE_REGISTER.md`
14. `P0A_MANIFEST_AND_QA.md`

## 핵심 잠금

- 주 기여: ADCS용 적응형 자세 추정기
- filter: 6D Kinematic MEKF
- convention: scalar-first Hamilton `q_NB`, right-multiplicative error
- scenarios: UNIT-ST, MAIN-FUSION, STRESS-MAG
- Split-KalmanNet: direct-gain baseline
- structured Q/R/gate MEKF: proposed candidate
- context: oracle → ANN → selected fast function SNN
- split: trajectory-level

## 사용 주의

- 문서의 대표 궤도·rate 값은 연구 benchmark 결정이지 특정 제품/임무 사실이 아니다.
- sensor noise magnitude가 `TBD`인 곳은 실제 characterization, Allan deviation, 공식 datasheet 순으로 채운다.
- 코드 구현은 이 패키지에 포함하지 않는다.

## QA

- `P0A_MANIFEST_AND_QA.md`: 사람용 QA/manifest
- `P0A_QA_RESULTS.json`: machine-readable QA 결과
