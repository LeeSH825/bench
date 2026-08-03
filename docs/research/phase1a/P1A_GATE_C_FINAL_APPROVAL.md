# Phase 1A Gate C Final Approval

- 결정일: 2026-08-02
- 결정: **GO**
- 적용 범위: Canonical MEKF geodesic/bias/NIS/NEES/SPD metrics
- 다음 Gate: Gate D1 — MEKF Adapter and Artifact Bridge

## 1. 최종 판단

Gate C 실행 증거를 검토한 결과 다음을 승인한다.

- Gate A exact-pi-aware log map을 재사용하는 right-local attitude error
- active scalar-first Hamilton `q_NB`의 geodesic attitude error
- gyro-bias per-axis/vector error 및 RMSE
- 실제 star-tracker update residual과 matching `S`를 이용한 NIS
- posterior right-local 6D error와 matching `P`를 이용한 NEES
- strict-SPD Cholesky solve와 P/S diagnostics
- chi-square consistency summary
- trajectory/timestamp/posterior pairing fail-loud 검증
- q/-q metric invariance
- input nonmutation 및 read-only result
- evaluation-only 정보 경계

## 2. 검증 결과

- Gate C 신규 시험: `43 passed`
- deterministic property sweep: `10 passed`
- Basilisk B2 replay metric smoke: `3 passed`
- Gate A: `55 passed`
- Gate B1 Amendment A1: `55 passed`
- Gate B2: `67 passed`
- legacy regression: `18 passed, 5 subtests passed`
- maximum right-local recovery error: `3.5388358909926865e-16`
- maximum q/-q NEES difference: `0`
- maximum NIS reference difference: `1.3010426069826053e-18`
- maximum NEES reference difference: `1.1102230246251565e-16`
- allowlist 밖 기존 path 변화: `0`
- frozen path hash 변화: `0`
- staged patch: `0 bytes`

## 3. Gate D에서 동결할 source

```text
bench/estimators/mekf.py
bench/tasks/generator/mekf_events.py
bench/tasks/generator/unit_st_synthetic.py
bench/tasks/generator/basilisk_unit_st.py
bench/metrics/mekf.py

tests/test_mekf_conventions.py
tests/test_mekf_core.py
tests/test_mekf_events.py
tests/test_unit_st_synthetic.py
tests/test_mekf_replay.py
tests/test_basilisk_unit_st_generator.py
tests/test_mekf_metrics.py
```

## 4. Gate D 분할

Gate D는 공유 runner 파일을 한 번에 수정하지 않고 두 단계로 나눈다.

### Gate D1

- unregistered MEKF adapter/bridge
- direct-core replay와 adapter replay 등가
- lossless q/b/P/r/S artifact bundle
- immutable dataset identity 전달
- truth-free estimator input
- Gate C metric용 exact pairing evidence

### Gate D2

- registry 및 task dispatch append-only 등록
- `run_suite.py` 국소 확장
- sidecar/cache 연결
- Phase 1A smoke YAML
- direct/adapter/runner replay 등가
- same-realization hash assertion
- canonical metric 및 artifact 출력
- legacy regression

Gate D1이 승인되기 전에는 registry, dispatch, runner, YAML을 수정하지 않는다.
