# Phase 1A Integration Checkpoint — Step 1 Final Approval

- 결정일: 2026-08-02
- 결정: **GO**
- 이전 명칭: Gate D1
- 권장 명칭: **P1A-CP4 Integration Step 1**
- 다음 작업: **P1A-CP4 Integration Step 2 — Registry, Dispatch, Runner, Cache, Smoke Suite**

## 1. 최종 판단

다음 항목을 승인한다.

- 기존 dense `ModelAdapter.predict(y_seq)` 경로가 typed event를 손실 없이 표현하지 못한다는 판단
- registry에 아직 등록하지 않은 explicit `MEKFEventReplayBridge`
- frozen Gate B1 `replay_trajectory`의 정확히 한 번 호출
- direct replay와 bridge replay의 exact q/b/P/final-state/r/S 등가성
- synthetic와 Basilisk UNIT-ST 공통 지원
- dataset semantic identity의 exact 보존
- truth/oracle/label/future/metric 비의존 estimator-facing API
- dense float32/zero-fill 변환 금지
- lossless posterior q/b/P artifact
- compact ST residual/S artifact
- q/-q invariance
- Gate C metric post-estimation pairing
- immutable/read-only artifact
- math/replay loop 비중복

## 2. 검증 결과

- D1 신규 시험: `24 passed`
- synthetic seed sweep: `5/5`
- Basilisk seed sweep: `3/3`
- Gate A: `55 passed`
- Gate B1: `55 passed`
- Gate B2: `67 passed`
- Gate C: `43 passed`
- legacy: `18 passed, 5 subtests passed`
- 기존 dirty fingerprint mismatch: `0`
- frozen file mismatch: `0`
- staged diff: `0`

## 3. Integration Step 2에서 사용할 ID

```text
model_id   = mekf_event_replay_v1
task_family = mekf_unit_st_v1
```

## 4. Integration Step 2 exact existing-file shortlist

```text
bench/tasks/bench_generated.py
bench/models/registry.py
bench/runners/run_suite.py
```

다음 파일은 기본 범위에서 수정하지 않는다.

```text
bench/tasks/data_format.py
bench/tasks/generator/contract.py
```

typed sidecar를 위 두 파일 수정 없이 전달할 수 없다는 executable evidence가 나오면 구현을 중단하고 schema-migration scope extension을 요청한다.

## 5. Integration Step 2 핵심 조건

1. registry/task family/model ID는 append-only다.
2. 기존 dense `x/y` 경로를 MEKF typed-event source of truth로 사용하지 않는다.
3. runner는 `_load_split_npz`, `_SeqDataset`, `_predict_batches` 전에 typed-event branch로 분기한다.
4. cache hit는 파일 존재가 아니라 strict manifest/hash 검증을 통과해야 한다.
5. bridge input은 sensor events와 explicit initial state/Qc뿐이다.
6. truth join은 estimation 완료 후 metric evaluation 단계에서만 수행한다.
7. direct/bridge/runner output은 exact-equivalent해야 한다.
8. runner artifact는 q/b/P와 compact ST r/S를 손실 없이 보존한다.
9. fresh generation과 verified cache hit 모두 synthetic/Basilisk에서 시험한다.
10. 이 Step 완료 후 Phase 1A Integration Checkpoint를 종료하고 Phase 1B classical benchmark completion으로 이동한다.
