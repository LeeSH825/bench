# Code CLI Agent Prompt 2 — Locked Attitude Math and MEKF Core

이 프롬프트는 Chat에서 `P1A_REPOSITORY_AUDIT.md`와 `P1A_IMPLEMENTATION_MAP.md`를 승인한 뒤 실행한다.

---

당신은 Phase 1A의 첫 구현 담당자다. 승인된 `docs/research/phase1a/P1A_IMPLEMENTATION_MAP.md`를 따르되, Phase 0A의 locked MEKF convention을 변경하지 않는다.

## 먼저 읽을 문서

- `docs/research/phase0a/decision_lock/P0_05_MEKF_MATH_CONTRACT.md`
- `docs/research/phase0a/decision_lock/P0_05_MEKF_CONVENTION_TEST_VECTORS.md`
- `docs/research/phase0a/decision_lock/P0A_IMMEDIATE_TEST_SPEC.md`
- `docs/research/phase1a/P1A_REPOSITORY_AUDIT.md`
- `docs/research/phase1a/P1A_IMPLEMENTATION_MAP.md`

## 이번 구현 범위

Basilisk, runner, model registry, training code와 독립적인 float64 reference math/core를 구현한다.

### 구현 기능

- scalar-first Hamilton quaternion
- normalize, conjugate, inverse, product
- quaternion ↔ DCM
- SO(3) Exp/Log
- quaternion hemisphere/sign alignment
- skew matrix
- right Jacobian `J_r`
- Kinematic MEKF continuous error matrices `F,G,Q_c`
- discrete `Phi,Q_d`
- body-vector measurement prediction/Jacobian
- star-tracker tangent residual primitive
- Joseph covariance update primitive
- multiplicative injection, bias correction, covariance reset transport
- symmetry/SPD diagnostic helpers

### 구현할 test

- B1: propagation/discretization/bias sign
- B3: magnetometer analytic Jacobian vs finite difference
- B4: sun-vector tangent Jacobian vs finite difference
- B5: injection/reset consistency
- B6: quaternion sign invariance
- Phase 0A convention vectors 전부

## 제한

- `third_party/` 수정 금지
- 기존 MRP EKF와 기존 Basilisk generator behavior 수정 금지
- registry 연결 금지
- neural model 수정 금지
- eigenvalue clipping으로 오류 은폐 금지
- quaternion component MSE를 attitude metric으로 추가 금지
- locked convention과 충돌하면 임의 변경하지 말고 test failure와 영향 범위를 문서화

## 문서/산출물

- `docs/research/phase1a/P1A_IMPLEMENTATION_CONTRACT.md`
- `docs/research/phase1a/P1A_TEST_MATRIX.md`
- `experiments/phase1a/reports/P1A_MATH_VALIDATION_REPORT.md`

`P1A_IMPLEMENTATION_CONTRACT.md`에는 수식 기능과 실제 함수 경로의 1:1 mapping, array shape, unit, frame, prior/posterior notation을 적는다.

## 실행 검증

- 새 test만이 아니라 영향받을 수 있는 기존 핵심 test도 실행한다.
- exact command, Python environment, dependency version, seed를 report에 기록한다.
- 실패 test를 skip/xpass로 숨기지 않는다.
- test tolerance 변경이 필요하면 Phase 0A provisional tolerance와 수치 근거를 함께 기록한다.

## 완료 출력

1. 변경 파일 목록
2. `git diff --stat`
3. test command와 pass/fail count
4. B1/B3/B4/B5/B6별 evidence
5. legacy regression 결과
6. unresolved issue
7. 다음 Prompt 3에서 구현할 exact scope

가능하면 하나의 논리적 commit으로 정리하되, 사용자가 요청하지 않았다면 remote push는 하지 마라.

---
