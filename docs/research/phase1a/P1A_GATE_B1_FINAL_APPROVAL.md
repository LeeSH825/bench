# Phase 1A Gate B1 Final Approval

- 결정일: 2026-07-31
- 결정: **GO**
- 적용 범위: Typed gyro/ST event schema, synthetic UNIT-ST, deterministic serialization/hash/split, direct-core replay
- 다음 Gate: Gate B2 — Basilisk frame/convention proof and gyro + star-tracker UNIT-ST generator

## 1. 최종 판단

`P1A_GATE_B1_VALIDATION_REPORT.md`의 실행 증거를 기준으로 다음 항목을 승인한다.

- versioned typed event metadata와 sensor-specific payload table
- `int64/int16/float64/float64/int64/bool/int64` event metadata dtype
- gyro `[G,3]`, star-tracker quaternion `[S,4]`, star-tracker covariance `[S,3,3]` float64 payload
- truth와 estimator event의 물리적·파일 구조적 분리
- 모든 nominal event의 zero-latency 계약
- 동일 timestamp의 gyro propagation 후 star-tracker update 순서
- canonical JSON 및 `allow_pickle=False` NPZ serialization
- truth, sensor payload, event order, manifest, dataset semantic hash
- truth/gyro/ST/sign/split seed namespace 분리
- stable int64 trajectory identity와 whole-trajectory split
- direct replay API의 truth/oracle/label 비의존성
- serialization round-trip replay equivalence
- raw star-tracker `q/-q` representation에 대한 posterior physical invariance
- quaternion unit norm과 posterior covariance SPD 유지
- Gate A 및 지정 legacy regression 유지

## 2. 검증 결과

- Gate A 재검증: `55 passed`
- 지정 legacy regression: `18 passed, 5 subtests passed`
- Gate B1 신규 시험: `39 passed`
- property sweep: 10 datasets × 4 trajectories
- same-seed regeneration: PASS
- serialization round trip: PASS
- sign-paired replay: PASS
- whole-trajectory split disjointness: PASS
- replay finite/unit-quaternion/Cholesky safety: PASS
- 기존 dirty path 1,260개의 status/content mismatch: 0
- Gate B1 allowlist 밖 agent 변경: 0

## 3. Gate B2에서 동결할 파일

다음은 Gate B1 source of truth로 동결한다. Gate B2는 읽고 import할 수 있지만 수정하지 않는다.

```text
bench/tasks/generator/mekf_events.py
bench/tasks/generator/unit_st_synthetic.py

tests/test_mekf_events.py
tests/test_unit_st_synthetic.py
tests/test_mekf_replay.py

docs/research/phase1a/P1A_EVENT_SCHEMA_CONTRACT.md
docs/research/phase1a/P1A_SYNTHETIC_UNIT_ST_CONTRACT.md
docs/research/phase1a/P1A_GATE_B1_TEST_MATRIX.md
experiments/phase1a/reports/P1A_GATE_B1_VALIDATION_REPORT.md
```

Gate A source도 계속 동결한다.

```text
bench/estimators/__init__.py
bench/estimators/mekf.py
tests/test_mekf_conventions.py
tests/test_mekf_core.py
```

## 4. Gate B2 불변 조건

1. Basilisk truth adapter는 locked active scalar-first `q_NB`를 출력해야 한다.
2. Basilisk recorder field 이름만 보고 frame 방향을 가정하지 않는다.
3. identity, 각 축 ±90°, MRP shadow set, constant-rate propagation의 executable proof로 변환을 잠근다.
4. `omega_BN_B`가 Gate A propagation의 body-frame angular rate와 같은 부호·단위를 갖는지 동역학 시험으로 증명한다.
5. Basilisk는 Tier-0 rigid-body truth source로 사용한다.
6. gyro와 star-tracker는 truth에서 생성되는 project-owned parameterized sensor-output layer로 구현한다.
7. built-in Basilisk star-tracker가 사용되지 않았다면 그렇게 명시하며, 사용한 것처럼 표현하지 않는다.
8. Gate B1 typed schema, serializer, semantic hash, split, direct replay를 그대로 재사용한다.
9. 최초 Basilisk UNIT-ST도 `arrival_time_s == measurement_time_s`를 유지한다.
10. nonzero latency, outage, false solution, magnetometer, sun sensor, orbit environment, canonical metric, runner integration은 구현하지 않는다.
11. simulator identity에는 최소 Python, NumPy, SciPy, Basilisk runtime version과 generator/convention/schema identity를 기록한다.
12. Gate B2는 Gate A/B1/legacy regression을 모두 유지해야 한다.

## 5. Gate B2 완료 조건

Gate B2는 최소 다음을 모두 만족해야 한다.

- Basilisk MRP recorder → locked `q_NB` basis-vector proof
- MRP shadow-set physical invariance
- `omega_BN_B` sign/frame/unit proof
- spherical-inertia zero-torque constant-rate trajectory와 Gate A analytic propagation의 일치
- deterministic Basilisk truth and sensor hashes
- independent truth/sensor/sign/split seed behavior
- B1 artifact serialization round trip
- direct replay finite/unit-quaternion/SPD safety
- raw ST `q/-q` stream physical posterior invariance
- Gate A, Gate B1, legacy regression PASS
- dirty-tree integrity PASS

Gate B2가 통과한 뒤에만 Gate C canonical geodesic/bias/NIS/NEES/SPD metric 구현으로 진행한다.
