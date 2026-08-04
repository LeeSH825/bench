# Benchmark Execution Visualization Tool — Do Not Do, Risks, and Migration

## 1. 반드시 하지 말아야 할 것

### DND-001 — Frontend callback에서 training 실행

금지:

- Dash callback
- Streamlit button handler
- Panel callback
- frontend background task
- UI-owned thread

이유:

- browser/session disconnect가 lifecycle에 영향을 준다.
- server restart 시 상태가 사라진다.
- long callback이 request worker를 점유한다.
- signal/process group/GPU lease 관리가 불명확해진다.

### DND-002 — Session state를 run registry로 사용

금지:

- `st.session_state`
- Dash `dcc.Store`
- browser local storage
- in-memory global dict

이들은 view/session cache일 수는 있지만 authoritative state가 될 수 없다.

### DND-003 — `model.pt` load를 resume라고 표시

optimizer, scheduler, RNG, sampler, cursor, phase가 없으면 warm start이다.

### DND-004 — deterministic path에 재실행 결과 덮어쓰기

같은 config라도 새 run_id와 directory를 사용한다. 기존 run은 immutable하다.

### DND-005 — `variant_label` 또는 bare `model_id`를 identity key로 사용

표시 문자열과 canonical identity를 분리한다.

### DND-006 — stdout parsing을 metric source of truth로 사용

stdout은 사람이 읽는 로그다. metric은 structured event로 발생시킨다.

### DND-007 — Stop과 Kill을 같은 버튼으로 구현

- Stop safely: checkpoint-safe boundary
- Force terminate: checkpoint 보장 없음

둘의 결과 상태와 UX를 분리한다.

### DND-008 — 모든 모델의 exact resume를 한 번에 구현

먼저 KNet/Split을 인증하고 multi-phase model은 이후에 추가한다.

### DND-009 — frontend framework object를 core schema에 넣기

Pydantic/domain object는 Dash/Streamlit/Panel을 import하지 않는다.

### DND-010 — UI가 DB state를 임의로 직접 수정

UI는 action request를 작성하고 manager/worker가 전이를 처리한다.

### DND-011 — checkpoint registry를 payload보다 먼저 commit

payload가 완전하고 hash가 검증된 뒤 catalog에 등록한다.

### DND-012 — untrusted path/import/config를 remote에 노출

local-only security rule이 완성되기 전 `0.0.0.0` bind를 금지한다.

### DND-013 — paper fidelity를 실행 가능 여부와 동일시

`supports_train=true`와 `paper_fidelity_status=verified`는 별도다.

### DND-014 — third-party code를 조용히 수정

필요한 경우 patch reason, file, revision, test를 exception record에 남긴다.

### DND-015 — JSONL만 또는 SQLite만 사용

- DB만 사용하면 portable audit/recovery가 약해진다.
- JSONL만 사용하면 query/state transition/concurrency가 약해진다.

둘의 역할을 분리한다.

## 2. 주요 위험 목록

| ID | 위험 | 가능성 | 영향 | 대응 |
|---|---|---|---|---|
| R-01 | 기존 run overwrite | 높음 | 치명적 | immutable run allocator, collision test |
| R-02 | Stop 시 checkpoint 손상 | 중간 | 치명적 | atomic write, fsync, checksum |
| R-03 | UI와 worker state 불일치 | 높음 | 높음 | DB state_version, heartbeat, event cursor |
| R-04 | orphan process/GPU 점유 | 중간 | 높음 | process group, start-time verification, orphan detector |
| R-05 | exact resume 오인 | 높음 | 연구 신뢰도 치명적 | certification flag, parity test, warm-start 분리 |
| R-06 | event 폭주/DB 병목 | 중간 | 중간 | JSONL append, sampling, batching, DB에는 index만 |
| R-07 | large log browser freeze | 높음 | 중간 | virtualized tail, bounded fetch, archive |
| R-08 | GPU attribution 오류 | 중간 | 중간 | whole-device/process 분리, quality flag |
| R-09 | config form/YAML divergence | 높음 | 높음 | typed object single source, round-trip test |
| R-10 | path traversal/dynamic import | 중간 | 치명적 | allowlist, safe root, argv execution |
| R-11 | SQLite contention | 중간 | 중간 | WAL, short transaction, busy timeout |
| R-12 | schema migration failure | 중간 | 높음 | backup, migration test, version gate |
| R-13 | legacy artifact 불완전 | 높음 | 중간 | read-only importer, confidence/unknown status |
| R-14 | adapter instrumentation drift | 높음 | 높음 | shared observer contract, adapter tests |
| R-15 | model phase checkpoint 누락 | 중간 | 높음 | capability slots, phase-aware schema |
| R-16 | UI framework lock-in | 중간 | 중간 | FastAPI/domain service boundary |
| R-17 | WebSocket reconnect gap | 중간 | 중간 | event cursor gap fill, polling fallback |
| R-18 | third-party dirty revision 미기록 | 현재 존재 | 높음 | dirty diff/provenance snapshot |
| R-19 | full test baseline 부재 | 현재 존재 | 높음 | pytest 설치, baseline report |
| R-20 | 현재 Split adapter와 논문 절차 혼동 | 높음 | 연구적 영향 높음 | implementation_id/fidelity audit |

## 3. exact resume 관련 위험

### 3.1 단순히 저장하면 안 되는 state

- optimizer state
- scheduler state
- AMP scaler
- RNG
- sampler/DataLoader generator
- early stopping
- best model tracking
- current phase/subphase
- MAML task/inner/outer cursor
- AKNet base/hypernetwork stage
- ME enhancer/split stage

### 3.2 deterministic parity 한계

CUDA kernel, nondeterministic operation, data-loader worker, precision에 따라 bitwise parity가 불가능할 수 있다. 따라서 certification은 다음을 명시한다.

- deterministic prerequisites
- supported device/precision
- exact cursor boundary
- parameter/metric tolerance
- unsupported configuration

`Exact`라는 UI label은 이 선언 범위 내의 equivalence를 뜻한다.

## 4. 기존 Streamlit Inspector 이행 전략

### Stage A — 병행 유지

- 기존 Inspector 변경 최소화
- new run의 visualization artifact를 기존 contract로 생성
- Dash에서 legacy Inspector deep link
- run_id ↔ legacy path mapping

### Stage B — shared visualization service

- artifact discovery/loading을 pure service로 추출
- Plotly figure builder를 UI-independent function으로 추출
- identity와 comparison selection을 run_id/variant_id 기반으로 변경

### Stage C — 선택적 포팅

- 가장 자주 쓰는 metric/trajectory/diagnostic부터 Dash page로 포팅
- visual regression과 numerical parity 확보
- legacy Inspector를 read-only fallback으로 유지

### Stage D — retirement 판단

다음이 모두 통과할 때만 Streamlit retirement를 검토한다.

- 주요 panel parity
- deep-link parity
- legacy run coverage
- performance/load test
- researcher acceptance

## 5. DB/event migration

### 5.1 새 run

새 run은 처음부터 registry/event contract를 사용한다.

### 5.2 legacy run

- 원 directory를 수정하지 않는다.
- read-only import record 생성
- missing field는 `unknown`으로 둔다.
- completion/failed 판정 confidence를 기록한다.
- 기존 checkpoint를 exact-resume certified로 승격하지 않는다.

### 5.3 migration rollback

- DB migration 전 backup
- migration version record
- 실패 시 이전 binary와 DB snapshot으로 rollback
- artifact filesystem은 migration 중 수정하지 않음

## 6. Frontend 선택 관련 위험

### Dash callback complexity

대응:

- page/component/service 분리
- callback당 단일 책임
- pattern-matching callback 남용 금지
- API/domain unit test를 UI test보다 우선

### WebSocket premature optimization

대응:

- bounded polling MVP
- event cursor/reconnect contract 완성 후 push 도입
- push 실패 시 polling fallback

### React로 너무 이른 전환

대응:

- multi-user/custom UX 요구가 실제로 생기기 전까지 Dash 유지
- FastAPI API를 안정화해 future frontend option만 보존

## 7. Third-party isolation exception record

수정이 불가피할 때 다음 형식을 남긴다.

```yaml
exception_id: TP-001
model_id: split_knet
upstream_repo: third_party/Split-KalmanNet
upstream_revision: 0d626566...
files_changed:
  - path: ...
    reason: checkpoint-safe callback hook unavailable
minimality: one callback injection; no numerical logic change
validation:
  - upstream smoke test
  - benchmark parity test
  - exact-resume test
owner: ...
status: active
```

## 8. 연구 결과 보호 규칙

- UI 변경으로 metric computation을 바꾸지 않는다.
- raw linear MSE와 dB conversion을 event에서 분리한다.
- config diff와 code provenance를 result 옆에서 확인 가능하게 한다.
- model comparison에서 implementation/fidelity mismatch warning을 표시한다.
- resumed run과 continuous run을 lineage 없이 같은 run으로 합치지 않는다.

## 9. 운영 안전 규칙

- disk free space threshold 아래에서는 launch 차단 또는 warning
- stale heartbeat 후 자동 kill 금지; 먼저 PID identity 확인
- force kill은 process group 전체 대상
- UI server 종료가 worker kill로 이어지지 않음
- GPU lease release는 worker terminal + process absence 확인 후 수행
- checkpoint retention이 lineage source를 삭제하지 않음
