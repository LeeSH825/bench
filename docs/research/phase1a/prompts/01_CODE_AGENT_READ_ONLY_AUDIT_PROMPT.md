# Code CLI Agent Prompt 1 — Phase 1A Read-Only Repository Audit

아래 프롬프트를 `<REPO_ROOT>`에서 Code CLI agent에 전달한다.

---

당신은 AI-ADCS/KalmanNet benchmark repository의 Phase 1A 구현 전 read-only auditor다.

## 반드시 먼저 읽을 문서

1. `docs/research/audits/2026-07-27/AUDIT_CURRENT_STATE.md`
2. `docs/research/phase0a/decision_lock/P0A_PHASE_0A_SYNTHESIS.md`
3. `docs/research/phase0a/decision_lock/P0_01_DECISION_LEDGER.md`
4. `docs/research/phase0a/decision_lock/P0_02_TRUTH_SENSOR_ESTIMATOR_BOUNDARY.md`
5. `docs/research/phase0a/decision_lock/P0_03_TRUTH_MODEL_SPEC.md`
6. `docs/research/phase0a/decision_lock/P0_04_SENSOR_ROLE_AND_MODEL_SPEC.md`
7. `docs/research/phase0a/decision_lock/P0_05_MEKF_MATH_CONTRACT.md`
8. `docs/research/phase0a/decision_lock/P0_05_MEKF_CONVENTION_TEST_VECTORS.md`
9. `docs/research/phase0a/decision_lock/P0A_IMMEDIATE_TEST_SPEC.md`
10. `docs/research/source/03_AI_ADCS_KalmanNet_Detailed_Phase_Step_Roadmap.md`

## 이번 실행의 목적

실제 코드를 수정하기 전에 현재 repository에서 Phase 1A를 어디에 구현할지 정확히 결정한다. 현재 benchmark의 generator, schema, cache, runner, model registry, metric, Basilisk bootstrap, test convention을 조사하고 최소 변경 경로를 제안한다.

## 고정 계약

- 1차 필터: 6D Kinematic MEKF
- nominal state: `[q_NB, b_g]`
- local error: `[delta_theta, delta_b_g]`
- quaternion: scalar-first Hamilton `[w,x,y,z]`
- attitude: active B-to-N `q_NB`
- error: right-multiplicative
- first scenario: gyro + star tracker `UNIT-ST`
- current MRP+angular-rate additive EKF/KalmanNet path는 legacy reference다.
- legacy MRP task와 MEKF task를 같은 모델이라고 취급하지 않는다.
- `third_party/`는 수정하지 않는다.
- neural/SNN/FPGA 코드를 구현하지 않는다.
- Phase 0A 문서를 수정하지 않는다.

## 금지 사항

- source code 수정
- config 수정
- auto-format으로 repository 전체 변경
- dependency upgrade
- legacy public API 변경
- 기존 test를 pass시키기 위한 기대값 완화
- MRP 상태를 이름만 바꿔 MEKF라고 부르는 것

## 조사 항목

1. repository root와 실제 entry point
2. `bench/models`, `bench/tasks`, `bench/tasks/generator`, `bench/runners`, `bench/metrics`, `bench/configs`, `tests` 구조
3. `basilisk_adcs.py`, `basilisk_imu_adcs.py`의 입력, 출력, state/frame convention, deterministic seed, cache behavior
4. existing MRP EKF adapter의 상태, propagation, measurement, metric contract
5. generated-task schema가 asynchronous sensor packet을 직접 표현할 수 있는지
6. 동일 truth/sensor realization을 여러 estimator가 replay하는 현재 경로
7. trajectory-level split 지원 여부와 window-level leakage 가능성
8. Basilisk 설치/버전/실행 진입점과 reproducibility 문제
9. geodesic attitude metric, bias metric, NIS, NEES, SPD diagnostics의 현재 구현 여부
10. 새 MEKF core가 기존 runner에 종속되지 않고 test 가능한 위치
11. 기존 Basilisk generator에서 안전하게 재사용 가능한 helper와 재사용하면 안 되는 MRP-specific logic
12. legacy regression을 보존하면서 Phase 1A를 추가할 최소 diff

## 반드시 작성할 파일

코드를 수정하지 말고 다음 Markdown만 새로 작성한다.

- `docs/research/phase1a/P1A_REPOSITORY_AUDIT.md`
- `docs/research/phase1a/P1A_IMPLEMENTATION_MAP.md`
- `docs/research/phase1a/P1A_RISK_REGISTER.md`

각 문서에는 근거가 되는 repository path와 line/function name을 기록한다.

## P1A_IMPLEMENTATION_MAP 필수 표

| 책임 | 권장 새 파일 | 재사용 파일/함수 | 수정 금지 legacy | 필요한 test | integration 시점 |

## 마지막 출력

1. `git status --short --branch`
2. 조사 command 목록
3. 생성한 문서 목록
4. 추천 구조
5. blocking issue
6. Phase 1A 구현 Prompt 2에서 수정할 정확한 파일 shortlist

실제 구현은 시작하지 마라.

---
