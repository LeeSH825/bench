# Phase 1A Gate B2 Final Approval

- 결정일: 2026-08-01
- 결정: **GO**
- 적용 범위: Basilisk frame/convention proof, Tier-0 rigid-body truth, parameterized gyro/star-tracker sensor layer, Gate B1 typed serialization/replay
- 다음 Gate: Gate C — Canonical MEKF Geodesic/Bias/NIS/NEES/SPD Metrics

## 1. 최종 판단

Gate B2 실행 증거를 검토한 결과 다음을 승인한다.

- `q_NB = normalize(MRP2EP(sigma_BN))`
- `R_NB = quat_to_dcm(q_NB) = MRP2C(sigma_BN).T`
- `MRP2C(sigma_BN) = C_BN`
- active scalar-first Hamilton body-to-navigation `q_NB`
- MRP shadow-set physical invariance
- `omega_BN_B`의 body-frame, rad/s, Gate A right-propagation 부호
- spherical-inertia, zero-torque constant-rate Basilisk truth
- project-owned gyro measurement model
- project-owned quaternion star-tracker measurement model
- Gate B1 event schema, serializer, hash, split, direct replay 재사용
- `generator_id=basilisk-unit-st-v1`
- strict expected-generator-ID load
- deterministic hash와 seed namespace 분리
- zero-latency 및 gyro-before-ST ordering
- truth/sensor/estimator 정보 경계
- Gate B1 문서의 passive `q_NB` 표현 정정

## 2. 검증 결과

- 신규 Gate B2: `67 passed`
- Gate A: `55 passed`
- Gate B1 Amendment A1: `55 passed`
- 지정 legacy regression: `18 passed, 5 subtests passed`
- static basis-vector 최대 오차: `4.440892098500626e-16`
- shadow-set DCM 최대 오차: `4.85722573273506e-16`
- fine-grid dynamic attitude log error: `4.872566201647101e-16 rad`
- 최대 local rate-increment error: `3.219646771412954e-14 rad/s`
- 5-seed regeneration/serialization/q/-q/split property sweep: 전부 PASS
- 기존 보호 경로 content/status mismatch: 0
- staged diff: 0

## 3. Gate C 이후 동결할 source

다음 Gate A/B1/B2 source와 tests는 Gate C에서 읽고 import할 수 있으나 수정하지 않는다.

```text
bench/estimators/mekf.py
bench/tasks/generator/mekf_events.py
bench/tasks/generator/unit_st_synthetic.py
bench/tasks/generator/basilisk_unit_st.py

tests/test_mekf_conventions.py
tests/test_mekf_core.py
tests/test_mekf_events.py
tests/test_unit_st_synthetic.py
tests/test_mekf_replay.py
tests/test_basilisk_unit_st_generator.py
```

Gate C는 estimator, generator, serializer, replay를 수정하지 않고 새로운 canonical metric module만 추가한다.

## 4. Gate C 불변 조건

1. 자세 오차는 quaternion component subtraction이 아니라 right-local log map으로 계산한다.
2. `q`와 `-q`는 동일한 물리 자세와 동일한 metric을 생성해야 한다.
3. attitude metric의 주 단위는 rad이며 보고용 deg를 함께 제공할 수 있다.
4. bias metric 단위는 rad/s다.
5. ST NIS는 `r^T S^-1 r`이며 ST update가 존재하는 시각에만 계산한다.
6. state NEES는 `[Log(q_hat^-1⊗q_true), b_true-b_hat]`와 posterior `P`를 사용한다.
7. NIS/NEES solve는 Cholesky 기반이며 inverse, pseudo-inverse, jitter, clipping을 사용하지 않는다.
8. 비대칭, nonfinite, non-SPD 입력은 fail-loud한다.
9. metric은 estimator 상태, truth, residual, covariance를 수정하지 않는다.
10. metric module은 Basilisk, runner, registry, visualization, torch를 import하지 않는다.
11. Gate C는 runner/registry/artifact integration을 시작하지 않는다.
12. 공식 metric과 visualization fallback을 혼합하지 않는다.

## 5. 다음 Gate

다음 작업은 다음만 구현한다.

- quaternion geodesic/right-local attitude error
- gyro-bias error와 RMSE
- star-tracker NIS
- 6D right-local state NEES
- covariance/innovation SPD diagnostics
- chi-square consistency summary 및 closed-form tests

Gate D의 adapter, registry, runner, YAML, visualization 연결은 Gate C 승인 이후로 미룬다.
