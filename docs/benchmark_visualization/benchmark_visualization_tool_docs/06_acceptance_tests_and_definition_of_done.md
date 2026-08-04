# Benchmark Execution Visualization Tool — Acceptance Tests and Definition of Done

## 1. 테스트 원칙

- UI screenshot만으로 완료를 판단하지 않는다.
- domain/service/API를 먼저 test하고 UI는 그 위에 얹는다.
- continuous run과 resumed run의 parity는 수치로 검증한다.
- failure, interrupt, crash, stale heartbeat를 의도적으로 주입한다.
- GPU가 있는 환경과 CPU-only fallback을 모두 검증한다.
- legacy artifact와 new run을 별도 fixture로 유지한다.

## 2. Baseline gate

### B-01 Full test collection

- `pytest` 설치
- 모든 test file collection
- baseline pass/fail 목록 저장
- 기존 실패와 신규 실패 구분

### B-02 Repository provenance

- commit, branch, dirty diff, submodule revision 기록
- test report에 환경 fingerprint 포함

### B-03 Golden fixtures

최소 fixture:

- successful KalmanNet run
- successful Split-KalmanNet run
- model-based baseline run
- failed run
- incomplete/legacy run
- same model/different init runs
- representative checkpoint

## 3. Identity/config tests

### C-01 Stable identity

같은 canonical input은 같은 variant_id를 만들고, process restart 후에도 동일하다.

### C-02 Unique run allocation

같은 config를 동시에 두 번 launch해도 서로 다른 run_id/path를 받는다.

### C-03 Variant collision regression

같은 model_id, 다른 init_id/implementation_id가 다음에서 충돌하지 않는다.

- DB
- selector
- chart trace
- legend
- color
- checkpoint catalog
- URL/deep link
- comparison matrix

### C-04 Config round-trip

모든 supported preset에 대해:

```text
YAML → typed config → canonical YAML → typed config
```

resolved semantics가 동일하다.

### C-05 Unknown/invalid fields

- unknown key가 silent drop되지 않음
- range/enum/cross-field 오류가 field-level message로 반환됨

### C-06 Structural hash

구조적 field 변경 시 hash가 바뀌고 operational-only 변경은 structural hash를 바꾸지 않는다.

### C-07 Safe path

absolute path, `..`, symlink escape, unapproved repo/import path가 거부된다.

## 4. Registry/event tests

### R-01 State transition validation

허용되지 않은 transition을 거부한다.

### R-02 Optimistic concurrency

stale `state_version` action은 conflict를 반환한다.

### R-03 SQLite concurrency

heartbeat, action, checkpoint catalog, run query가 동시에 수행되어도 corruption이 없다.

### R-04 JSONL crash tail

마지막 line이 partial인 journal을 읽을 때 이전 valid event까지 복구하고 warning을 남긴다.

### R-05 Event monotonicity

run별 event_id가 중복/역행하지 않는다.

### R-06 DB/event reconciliation

DB last_event_id와 JSONL cursor가 어긋난 경우 recovery procedure가 동작한다.

### R-07 UI restart recovery

Dash/FastAPI 재시작 후 active/terminal run의 state, metric, log cursor를 복구한다.

## 5. Process lifecycle tests

### P-01 UI independence

worker 실행 중 Dash/FastAPI를 종료해도 run이 계속된다.

### P-02 PID/PGID correctness

worker와 DataLoader child가 같은 관리 대상 process group으로 추적된다.

### P-03 Stdout/stderr capture

일반 `print`, Python logging, exception traceback이 run별 log에 저장된다.

### P-04 Heartbeat

정상 worker는 heartbeat를 갱신하고 종료 후 멈춘다.

### P-05 Orphan detection

worker를 SIGKILL하면 stale heartbeat와 PID absence로 ORPHANED 후보가 된다.

### P-06 PID reuse defense

PID만 같고 process start time/worker token이 다른 process를 기존 worker로 오인하지 않는다.

### P-07 GPU lease

동일 GPU에 두 trainable run이 동시에 lease를 얻지 못한다.

### P-08 Manager restart

manager 재시작 후 살아 있는 worker를 재발견하거나 명확히 ORPHANED로 분류한다.

## 6. Telemetry tests

### T-01 NVIDIA environment

- whole GPU utilization/memory
- process GPU memory
- temperature/power availability
- collector timestamp

### T-02 CPU-only fallback

GPU가 없거나 collector가 실패해도 UI/API가 깨지지 않고 null/availability를 반환한다.

### T-03 Process tree aggregation

worker child process의 CPU/RSS가 합산된다.

### T-04 Sampling overhead

telemetry on/off tiny benchmark 비교에서 허용 overhead를 기록한다. 고정 숫자를 무조건 적용하지 않고 프로젝트 기준을 문서화한다.

### T-05 Stale/gap rendering

수집 중단 구간을 0으로 연결하지 않고 gap/stale로 표시한다.

## 7. Checkpoint tests

### K-01 Atomic checkpoint

checkpoint write 중 process를 강제 종료해도 final catalog에 partial checkpoint가 등록되지 않는다.

### K-02 Checksum

payload 변경/손상 시 validation 실패한다.

### K-03 Compatibility diff

model, implementation, structural config, dataset mismatch를 field-level로 설명한다.

### K-04 Warm-start semantics

warm start는 model weight만 복원하고 optimizer/cursor/RNG를 reset한다.

### K-05 Resume semantics

resume는 required states를 복원하며 child lineage를 생성한다.

### K-06 Retention safety

best/final/lineage source checkpoint가 retention으로 삭제되지 않는다.

## 8. Exact-resume certification

모델/implementation/version별로 인증한다.

### E-01 Continuous vs resumed

```text
A: N optimizer updates continuous
B: K updates → safe stop → interrupt checkpoint → resume → N updates
```

비교:

- model parameters
- optimizer state
- scheduler/scaler state
- global step/epoch/cursor
- validation sequence
- final metric
- RNG continuation

### E-02 Boundary declaration

지원 경계가 `optimizer_update`임을 명시하고 batch midpoint에서 resume를 거부한다.

### E-03 Determinism matrix

- CPU/fp32
- CUDA/fp32
- AMP가 지원되는 경우 해당 mode
- DataLoader worker count

지원/비지원 조합과 tolerance를 기록한다.

### E-04 Split-specific

현재 adapter의 실제 training phase/optimizer slot을 기준으로 인증한다. 논문식 alternating implementation이 별도라면 implementation_id와 인증을 분리한다.

### E-05 Multi-phase negative tests

AKNet/MAML/ME-Split이 미인증이면 UI/API가 exact resume를 거부한다.

## 9. API tests

### A-01 Validation API

valid/invalid RunSpec와 stable error schema

### A-02 Run create idempotency

같은 idempotency key의 재요청은 중복 run을 만들지 않는다.

### A-03 Action authorization/state

terminal run에 stop 요청, non-resumable checkpoint에 resume 요청을 거부한다.

### A-04 Event pagination

`after_event_id`, limit, ordering, gap 처리

### A-05 WebSocket reconnect

disconnect 후 cursor 기반 gap fill. WebSocket 미사용 MVP에서도 polling API가 동일한 semantics를 제공한다.

### A-06 Health endpoints

registry, worker manager, telemetry collector별 degraded 상태를 구분한다.

## 10. Frontend tests

### U-01 Runs table

state/filter/sort/pagination와 stable run_id row key

### U-02 Live updates

terminal 이전에 metric/resource/log가 갱신된다.

### U-03 Session restart

browser refresh/server restart 후 authoritative state가 복구된다.

### U-04 Button gating

capability/state에 따라 Stop/Resume/Warm start 버튼이 정확히 활성화된다.

### U-05 Safety dialog

force terminate와 warm start/resume 문구가 혼동되지 않는다.

### U-06 Config form/YAML synchronization

양방향 편집과 validation, diff preview

### U-07 Deep links

run/checkpoint/compare URL을 새 session에서 열어도 동일 대상을 표시한다.

### U-08 Legacy Inspector link

new run과 imported legacy run이 적절한 Inspector target으로 연결된다.

### U-09 Accessibility

상태를 색상만으로 표시하지 않고 keyboard focus와 text label을 제공한다.

### U-10 Large log

큰 log에서 browser가 freeze되지 않고 bounded tail/pagination이 작동한다.

## 11. Failure injection scenarios

- config validation failure
- dataset missing/corrupt
- CUDA OOM
- NaN loss
- adapter exception
- checkpoint disk full
- SQLite busy/temporary unavailable
- event write failure
- telemetry collector failure
- SIGINT
- SIGTERM
- SIGKILL
- manager crash
- UI crash
- stale heartbeat
- corrupt checkpoint
- legacy incomplete meta

각 시나리오는 final state, failure artifact, exit code, UI message, recovery action을 검증한다.

## 12. Performance tests

- 1,000 / 10,000 / 100,000 run registry list query
- large events.jsonl tail latency
- 24시간 telemetry sample size projection
- selected run detail update latency
- concurrent read while worker writes
- large trajectory artifact load
- browser memory under long log/metric session

## 13. Phase별 Definition of Done

### Phase 0 DoD

- identity/config schema versioned
- immutable run allocation
- existing tests regression 없음
- config/identity acceptance tests 통과
- docs와 migration note 작성

### Phase 1 DoD

- worker process/UI independence
- registry/event/heartbeat/telemetry
- ordinary failure 및 abrupt death 기록
- restart recovery 통과

### Phase 2 DoD

- checkpoint v1 atomic
- warm-start/resume 분리
- 최소 KNet/Split exact-resume certification 또는 명확한 미인증 상태
- compatibility/retention test

### Phase 3 DoD

- read-only Dash dashboard
- run list/detail/live log/metric/resource
- restart/deep-link/browser test
- legacy Inspector 연결

### Phase 4 DoD

- schema-driven config form
- GUI/CLI parity
- path/import validation
- launch collision/idempotency test

### Phase 5 DoD

- safe stop/force terminate/resume/warm start
- 정확한 상태 전이
- lineage 및 checkpoint selector
- negative capability gating

## 14. Release gate

MVP release에는 다음이 모두 필요하다.

- baseline test report
- schema/API version
- DB migration/backup procedure
- security/local-only note
- supported model/capability matrix
- exact-resume certification matrix
- known limitations
- operator recovery guide
- reproducible tiny demo
