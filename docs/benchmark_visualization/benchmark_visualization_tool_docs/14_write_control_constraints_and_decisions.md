# Benchmark Execution Visualization — Write-Control Constraints and Decisions

작성 기준일: 2026-07-31  
상태: normative addendum  
적용 범위: canonical resumable training path, WorkerManager child resume, write API, Dash Stop/Resume controls

## 1. 문서 목적

이 문서는 write-control tranche에서 의미가 흔들리기 쉬운 결정들을 고정한다. 구현 편의를 이유로 암묵적으로 바꾸지 않는다. 변경이 필요하면 기존 ADR을 삭제하거나 소급 수정하지 말고, 새 ADR에서 `supersedes` 관계를 기록한다.

## 2. 용어

### Certified control-plane run

다음 tuple이 registry의 exact-resume certification row와 정확히 일치하는 새로운 control-plane supervised run이다.

```text
model_id
implementation_id
checkpoint_schema_version
resume_boundary
precision
device_class
num_workers
training_mode
```

### Legacy training path

기존 adapter `train()`을 사용하는 경로이다. 기존 CLI와 과거 run의 semantics를 보존한다.

### Canonical resumable path

control-plane certified run이 첫 update부터 사용하는 `resumable_train()` 경로이다.

## 3. Architecture Decision Records

### ADR-WC-001 — Certified control runs use resumable training from update 0

인증 tuple이 정확히 일치하는 새 control-plane run은 자동으로 다음을 사용한다.

```text
training_path_id = control_resumable_v1
```

사용자에게 legacy/resumable toggle을 제공하지 않는다. 이 결정은 Stop/Resume를 나중에 켜기 위해 중간에 loop를 바꾸는 위험을 제거한다.

### ADR-WC-002 — Legacy `train()` remains unchanged

기존 `train()`을 삭제하거나 default batch order, optimizer update, validation cadence, early-stop semantics를 바꾸지 않는다.

기존 CLI와 과거 run은 다음으로 분류한다.

```text
training_path_id = legacy_train_v1
```

training-path field가 없는 old RunSpec도 `legacy_train_v1`이다. 소급해서 resumable이라고 표시하지 않는다.

### ADR-WC-003 — No silent fallback

`control_resumable_v1`로 resolve된 run이 resumable path를 실행할 수 없으면 validation 또는 worker failure로 종료한다. `train()`으로 silently fallback하지 않는다.

### ADR-WC-004 — Training path is structural provenance

`training_path_id`는 다음에 저장한다.

- resolved RunSpec
- structural config hash
- run registry/detail
- lifecycle start event
- checkpoint manifest/payload snapshot
- comparison provenance

resume child는 parent와 동일한 `training_path_id`를 유지한다.

### ADR-WC-005 — Certification is tuple-specific, not model-name-specific

`kalmannet_tsp` 또는 `split_knet`라는 이름만으로 controls를 활성화하지 않는다. CPU/fp32/0-workers 등의 인증 envelope가 하나라도 다르면 default는 `false`이다.

coarse `supports_exact_resume=True`를 모델 전체에 부여하지 않는다.

### ADR-WC-006 — Direct legacy/resumable parity is a migration characterization

KNet/Split의 `train()`과 update-0 `resumable_train()`을 직접 비교한다.

- bitwise 동일하면 결과를 문서화하고 regression test로 고정한다.
- 동일하지 않으면 legacy semantics를 억지로 변경하지 않는다.
- 차이가 있으면 training path를 structural identity에서 분리하고 과거 run과 자동 동등 비교를 금지한다.
- exact resume certification은 resumable path 내부의 연속-vs-resume 동등성이지, legacy path와의 동등성 주장이 아니다.

### ADR-WC-007 — Resume is an immutable child execution

resume는 parent를 다시 `RUNNING`으로 만들지 않는다.

child는 다음을 가진다.

```text
new run_id
new run directory
parent_run_id
resumed_from_run_id
resumed_from_checkpoint_id
same variant_id
same training_path_id
```

parent의 state, events, checkpoints, artifacts, exit code는 변경하지 않는다.

### ADR-WC-008 — Child lifecycle does not use parent `RESUMING`

resume child는 일반 lifecycle을 사용한다.

```text
CREATED → VALIDATING → QUEUED → STARTING → RUNNING → terminal
```

resume 여부는 lineage와 start event로 표현한다. parent를 `RESUMING`으로 변경하지 않는다.

### ADR-WC-009 — Write API creates durable actions, not training work

HTTP handler는 다음만 수행한다.

- request validation
- eligibility/compatibility validation
- durable action creation/retrieval
- coordinator/WorkerManager에 위임
- action resource 반환

handler 안에서 model setup, checkpoint unpickle, training loop를 장시간 동기 실행하지 않는다. checkpoint validation이 필요한 경우 bounded service call로 수행하며, child training은 반드시 별도 worker process이다.

### ADR-WC-010 — Stop remains registry-backed and worker-owned

Stop API는 signal을 보내지 않는다. 기존과 동일하게 registry action을 기록하고 worker가 safe boundary에서 처리한다.

```text
RUNNING → STOP_REQUESTED → CHECKPOINTING → INTERRUPTED
```

`INTERRUPTED`는 valid interrupt checkpoint가 존재한 뒤에만 가능하다.

### ADR-WC-011 — Stop before first update

이번 tranche에서는 현재 backend semantics를 유지한다.

- update 0에서 Stop 요청을 받아도 valid interrupt checkpoint를 생성한다.
- terminal state는 `INTERRUPTED`이다.
- `CANCELLED`-before-first-update 정책은 후속 결정으로 미룬다.

UI에서 이를 “아직 시작하지 않았으므로 취소됨”이라고 표현하지 않는다.

### ADR-WC-012 — Resume UI targets validated interrupt checkpoints

MVP Dash의 `Resume training`은 다음에만 활성화한다.

- parent state `INTERRUPTED`
- checkpoint kind `interrupt`
- validation status `VALID`
- compatibility status `COMPATIBLE`
- certification tuple exact match
- `training_path_id = control_resumable_v1`
- remaining update budget 존재

backend 내부 서비스는 다른 checkpoint kind를 검증할 수 있으나, periodic/best/final resume UI는 이번 tranche에서 노출하지 않는다.

### ADR-WC-013 — Warm start remains separate and unimplemented on HTTP/UI

`model.pt` 또는 weight-only payload는 resume endpoint에서 거부한다. warm-start launch API와 UI는 이번 tranche에 포함하지 않는다.

Split의 legacy `model.pt`가 `hn1_init`/`hn2_init`을 포함하지 않아 bitwise reproducible하지 않다는 caveat를 UI/문서에서 숨기지 않는다.

### ADR-WC-014 — Action idempotency is mandatory

모든 write request는 idempotency key를 요구한다.

```text
same key + same logical payload
    → same action and same child/checkpoint

same key + different logical payload
    → 409 conflict
```

Dash callback retry와 double-click은 같은 key를 재사용해야 한다.

### ADR-WC-015 — Optimistic concurrency is mandatory

Stop request는 expected run `state_version`을 포함한다. Resume request는 expected parent `state_version`과 checkpoint identity를 포함한다.

stale request는 최신 state와 reason을 포함한 `409 Conflict`로 거부한다.

### ADR-WC-016 — Resume action completion semantics

Resume action과 child run lifecycle을 분리한다.

- action `REQUESTED`: durable row created
- action `ACKNOWLEDGED`: validation/allocation coordinator가 소유
- action `COMPLETED`: exactly one child worker가 성공적으로 launch되어 child identity가 확정
- action `FAILED`: validation/allocation/launch 실패

child가 이후 training에서 실패해도 이미 완료된 launch action을 다시 실패로 바꾸지 않는다. child outcome은 child run state로 판단한다.

Stop action의 completion은 기존 semantics를 유지한다. valid interrupt checkpoint와 `INTERRUPTED`가 완성되어야 action `COMPLETED`이다.

### ADR-WC-017 — Resume action must survive API restart

Resume request는 API process memory에만 존재하면 안 된다.

crash window마다 다음이 가능해야 한다.

- action row만 존재 → restart reconciler가 처리
- child allocated, worker 미실행 → 같은 child를 launch하거나 명시적으로 fail
- worker launched, API 사망 → worker 계속 실행
- retry → duplicate child 없음

### ADR-WC-018 — Write mode is explicit and local-only

기본값은 read-only이다.

```text
BENCH_CONTROL_ENABLE_WRITES=0  # default
```

write routes를 사용하려면 명시적으로 활성화한다.

```text
BENCH_CONTROL_ENABLE_WRITES=1
```

이번 tranche에서 write mode는 loopback bind에만 허용한다. `BENCH_CONTROL_ALLOW_PUBLIC_BIND=1`이 있더라도 unauthenticated non-loopback write server는 startup을 거부한다.

### ADR-WC-019 — Dash uses the API only

Dash는 SQLite, checkpoint filesystem, worker PID를 직접 수정하지 않는다. 모든 write는 API client를 통해 요청한다.

browser callback이 training function 또는 `WorkerManager`를 import/실행하지 않는다.

### ADR-WC-020 — UI and backend gating must agree

UI disabled/hidden 상태는 편의 기능일 뿐 보안 경계가 아니다. backend가 동일한 certification, state, checkpoint, version 검증을 다시 수행한다.

unsupported reason은 machine-readable code와 human-readable message로 제공한다.

### ADR-WC-021 — No force terminate in this tranche

`Force terminate`, `SIGTERM`, `SIGKILL` action route/button을 추가하지 않는다. 외부 kill은 기존 `ORPHANED` semantics를 유지한다.

### ADR-WC-022 — No third-party source changes

KNet/Split third-party tracked source를 수정하지 않는다. 필요한 execution/checkpoint hook은 adapter/control layer에 둔다.

### ADR-WC-023 — Read-only mode remains first-class

write mode가 꺼져 있을 때:

- 기존 GET API 모두 동작
- Dash는 read-only marker 표시
- action button은 숨기거나 disabled
- POST는 side effect 없이 명확히 거부
- 기존 Playwright read-only test 유지

### ADR-WC-024 — Polling remains the transport

action progress는 기존 bounded polling과 event/action GET endpoint로 표시한다. WebSocket/SSE는 이번 tranche에 포함하지 않는다.

## 4. API contract decisions

권장 endpoint:

```text
POST /api/v1/runs/{run_id}/actions/stop
POST /api/v1/checkpoints/{checkpoint_id}/actions/resume
GET  /api/v1/actions/{action_id}
```

기존 `GET /runs/{run_id}/actions`는 유지한다.

권장 Stop request:

```json
{
  "idempotency_key": "uuid-or-client-key",
  "expected_state_version": 12,
  "reason": "operator_requested"
}
```

권장 Resume request:

```json
{
  "idempotency_key": "uuid-or-client-key",
  "expected_parent_state_version": 19
}
```

이번 tranche에서는 user-configurable learning-rate/max-update override를 body에 받지 않는다.

권장 accepted response:

```json
{
  "action_id": "...",
  "action_type": "STOP_GRACEFUL | RESUME_EXACT",
  "state": "REQUESTED | ACKNOWLEDGED | COMPLETED | FAILED",
  "run_id": "...",
  "checkpoint_id": "...",
  "child_run_id": null,
  "status_url": "/api/v1/actions/..."
}
```

HTTP semantics:

- `202 Accepted`: 새 action 또는 처리 중인 동일 action
- `200 OK`: 이미 완료된 동일 idempotent action 조회
- `403`: write mode disabled
- `404`: run/checkpoint 없음
- `409`: stale state, invalid current state, key collision
- `422`: certification/compatibility/schema eligibility 실패
- `500/503`: coordinator/manager infrastructure failure; action evidence 보존

## 5. 금지사항

다음은 하지 않는다.

- certified run에서 resumable path 실패 후 legacy fallback
- old RunSpec을 자동 resumable로 승격
- model 이름만 보고 Stop/Resume 활성화
- API handler 안에서 training loop 실행
- Dash가 SQLite에 직접 action row 삽입
- resume 결과를 parent directory에 쓰기
- same request retry로 child 여러 개 생성
- `model.pt`를 exact resume payload로 허용
- GPU run에 CPU certification을 일반화
- action 완료와 child training 완료를 혼동
- Stop과 Force Kill을 같은 버튼으로 제공
- UI에서만 compatibility를 검사
- third-party source를 편의상 수정
- `git add -A`, `git clean`, `reset --hard`로 unrelated work 훼손

## 6. 연구 해석 제약

- exact resume certification은 구현의 continuation equivalence를 뜻하며 paper fidelity를 뜻하지 않는다.
- 현재 Split adapter의 `supervised_single_optimizer_split_deviation` 상태는 유지한다.
- `train()`과 `resumable_train()`이 다르면 두 경로의 결과를 같은 implementation condition처럼 합치지 않는다.
- control-plane fresh run이 resumable path를 사용했다는 사실을 결과 provenance에서 표시한다.
