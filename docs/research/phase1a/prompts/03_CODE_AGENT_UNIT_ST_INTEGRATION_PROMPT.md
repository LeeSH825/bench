# Code CLI Agent Prompt 3 — UNIT-ST, Sensor Replay, Basilisk Adapter

이 프롬프트는 Math/Core test가 Chat에서 승인된 뒤 실행한다.

---

당신은 Phase 1A의 UNIT-ST integration 담당자다. 승인된 math/core를 사용하여 gyro + low-rate star tracker 시나리오와 deterministic replay를 구현한다.

## 먼저 읽을 문서

- `docs/research/phase0a/decision_lock/P0_02_TRUTH_SENSOR_ESTIMATOR_BOUNDARY.md`
- `docs/research/phase0a/decision_lock/P0_03_TRUTH_MODEL_SPEC.md`
- `docs/research/phase0a/decision_lock/P0_04_SENSOR_ROLE_AND_MODEL_SPEC.md`
- `docs/research/phase0a/decision_lock/P0A_IMMEDIATE_TEST_SPEC.md`
- `docs/research/phase1a/P1A_IMPLEMENTATION_CONTRACT.md`
- `experiments/phase1a/reports/P1A_MATH_VALIDATION_REPORT.md`

## 구현 범위

1. timestamped sensor packet schema
2. `measurement_time`, `arrival_time`, `sensor_id`, `value`, `validity`, `quality`, covariance profile ID, sequence ID
3. truth/evaluation, sensor/deployable, oracle label namespace 분리
4. deterministic seed와 config hash를 가진 trajectory manifest
5. analytic/synthetic UNIT-ST truth generator
6. gyro sensor output
7. star-tracker noisy quaternion/tangent measurement, multirate, zero latency와 delayed packet
8. Kinematic MEKF event loop
9. B2, B7, B8
10. C1 matched fixed-noise baseline과 all-one oracle equivalence
11. geodesic attitude metric, gyro-bias metric, NIS/NEES, quaternion norm, covariance SPD logging
12. Phase 1A dedicated runner/config
13. 필요한 경우 Basilisk Tier-0 adapter를 새 generator 이름으로 추가

## 중요한 통합 원칙

- 모든 estimator는 같은 pre-generated sensor stream을 replay해야 한다.
- true attitude/rate/bias, true Q/R, event label은 deployable estimator input에서 차단한다.
- star tracker는 ground truth가 아니다.
- 기존 MRP task output contract를 변경하지 않는다.
- 기존 `basilisk_adcs.py`/`basilisk_imu_adcs.py`의 public behavior를 바꾸지 않는다.
- 공통 Basilisk setup helper를 추출해야 한다면 먼저 legacy regression test를 추가하고 extraction 전후 동일성을 확인한다.
- 기존 general training runner에 무리하게 연결하지 말고, Phase 1A dedicated runner를 우선한다.

## 필수 UNIT-ST subcase

1. identity, no bias, no noise
2. constant angular rate, no bias, no noise
3. stationary known constant bias
4. Gaussian gyro/ST noise
5. asynchronous 100 Hz gyro / 1 Hz ST
6. zero-latency vs delayed-packet replay
7. moderate random initial attitude
8. large initial error regression

## 산출물

- configs under `bench/configs/phase1a/`
- tests under `tests/integration/phase1a/`
- `experiments/phase1a/reports/P1A_UNIT_ST_REPORT.md`
- `experiments/phase1a/reports/P1A_C1_BASELINE_REPORT.md`
- `docs/research/phase1a/P1A_GATE_REPORT.md`

## Gate 조건

- B1–B8 모두 통과
- no-noise case가 reference tolerance 내 일치
- constant bias가 합리적으로 수렴
- long-horizon quaternion norm과 covariance SPD 유지
- all-one ORACLE-QR과 F-TUNED가 수치적으로 동일
- sensor stream hash가 estimator별로 동일
- legacy benchmark 핵심 smoke test가 회귀하지 않음

## 완료 출력

1. changed files/diff stat
2. generated packet schema와 example manifest
3. test command 및 전체 결과
4. B2/B7/B8/C1 evidence
5. legacy regression 결과
6. Basilisk adapter를 사용했는지, synthetic only인지
7. Phase 1A Gate pass/fail과 근거
8. 다음 Package C implementation scope

---
