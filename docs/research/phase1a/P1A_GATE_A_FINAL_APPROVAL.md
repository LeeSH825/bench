# Phase 1A Gate A Final Approval

- 결정일: 2026-07-31
- 결정: **GO**
- 적용 범위: 6D Kinematic MEKF pure math/core
- 다음 Gate: Gate B1 — Typed Event Schema, Synthetic UNIT-ST, Deterministic Direct-Core Replay

## 1. 최종 판단

Gate A 본체와 Amendment A1의 증거를 검토한 결과, 다음 항목을 최종 승인한다.

- scalar-first Hamilton quaternion
- active body-to-navigation `q_NB`
- right-multiplicative local attitude error
- nominal state `[q_NB, b_g]`
- local error `[delta_theta, delta_b_g]`
- gyro propagation과 bias 부호
- continuous error model `F/G/Q_c`
- Van Loan `Phi/Q_d`
- body-vector 및 sun tangent Jacobian
- star-tracker tangent residual
- Joseph covariance update
- multiplicative injection과 exact right-reset transport
- covariance symmetry/SPD fail-loud policy
- ordinary, near-pi, exact-pi quaternion antipodal invariance
- `MEKFState`의 defensive copy와 read-only array immutability

## 2. 검증 결과

- Gate A Amendment 후 신규 시험: `55 passed`
- 지정 legacy regression: `18 passed, 5 subtests passed`
- exact-pi x/y/z/arbitrary-axis antipodal update: 동일 residual/correction/posterior/covariance
- ordinary antipodal property sweep: 1,000쌍 통과
- near-pi outside-tie property sweep: 256쌍 통과
- state arrays `q_NB`, `b_g`, `P`: 직접 mutation 거부
- 실패 및 성공 propagation/update 모두 prior state 불변
- 기존 dirty status/content fingerprint 변화: 0
- Gate A allowlist 밖 agent 변경: 0

## 3. Concurrent external ledger 판단

`artifacts/benchmark_write_control/...` 아래에 외부 실행 파일 4개가 추가된 사실은 별도 ledger에 기록되었다. 이 파일들은 Gate A allowlist, MEKF source, test, contract를 변경하지 않았으며 agent가 읽거나 수정하지 않았다. 따라서 Gate A 결과를 무효화하지 않는다.

Gate B1 실행 중에는 동일한 target 또는 shared source를 다른 프로세스가 수정하지 않아야 한다. 별도 artifact root의 후속 파일 생성은 agent 변경과 분리하여 ledger에만 기록할 수 있다.

## 4. Gate A 이후 불변 조건

Gate B 이후 작업은 다음을 지켜야 한다.

1. `bench/estimators/mekf.py`는 검증된 수학 source of truth로 동결한다.
2. event/generator/replay 코드가 quaternion, propagation, update, reset 수학을 중복 구현하지 않는다.
3. Gate B1은 Basilisk, runner, registry, canonical metrics를 구현하지 않는다.
4. Gate B1 estimator replay는 truth를 입력으로 받지 않는다.
5. 최초 UNIT-ST는 zero latency로 제한하고 `arrival_time == measurement_time`을 강제한다.
6. nonzero latency/OOSM, magnetometer, sun sensor, learned context는 후속 Gate로 미룬다.

## 5. 다음 Gate

다음 작업은 다음 세 기능만 구현한다.

- versioned typed gyro/star-tracker event schema
- analytic synthetic UNIT-ST generator
- serialization/hash/split을 거친 deterministic direct-core replay

Basilisk frame adapter는 Gate B2, canonical NIS/NEES metric은 Gate C, benchmark runner 통합은 Gate D에서 수행한다.
