# Benchmark Execution Visualization — Write-Control Acceptance and Test Plan

작성 기준일: 2026-07-31  
선행 gate: `READY_FOR_WRITE_CONTROL_TRANCHE`  
대상: canonical resumable training path, WorkerManager child resume, Stop/Resume API, Dash controls

## 1. 판정 원칙

- 문서 주장보다 실제 code/run/test를 우선한다.
- mock-only test로 process lifecycle을 승인하지 않는다.
- KNet/Split certification test는 pinned third-party implementation을 실제로 load해야 한다.
- 모든 destructive/fault test는 temporary control root와 disposable worktree에서 수행한다.
- UI gating test와 backend enforcement test를 모두 요구한다.
- exact resume의 기본 비교는 certified CPU/fp32/0-workers fixture에서 bitwise이다.
- 지원하지 않는 조합은 성공시키는 것이 아니라 구체적으로 거부하는 것이 통과이다.

## 2. Baseline gate

### B-WC-001 — Correct checkout

- checkpoint implementation commit `c0eaaf7`가 ancestor이다.
- checkpoint docs correction commit까지 기록한다.
- `bench/control/checkpoints/**`와 관련 tests가 존재한다.
- authoritative dirty tree가 아니라 isolated worktree를 사용한다.

### B-WC-002 — Baseline regression

변경 전 최소:

```bash
python -m pytest --collect-only -q
python -m pytest -q
python -m pytest -q tests/test_control_checkpoint_schema_atomicity.py
python -m pytest -q tests/test_control_exact_resume_certification.py
python -m pytest -q tests/test_control_graceful_stop.py
```

기준 결과의 감소, test 삭제, unconditional skip, xfail 추가를 허용하지 않는다.

## 3. Canonical training-path tests

### TP-001 — Certified KNet fresh run path

CPU/fp32/0-workers `kalmannet_tsp` control run이 다음을 만족한다.

- `training_path_id = control_resumable_v1`
- `resumable_train()` 호출
- legacy `train()` 미호출
- update 0부터 stop/checkpoint state 준비
- RunSpec/registry/event/checkpoint에 path 기록

### TP-002 — Certified Split fresh run path

TP-001과 동일한 조건을 `split_knet`에 적용한다.

### TP-003 — Tuple, not model-name gating

다음 중 하나를 바꾸면 `control_resumable_v1`이 되지 않아야 한다.

- device → CUDA
- precision → AMP/bf16
- `num_workers → 1`
- implementation id 변경
- training mode 변경
- checkpoint schema version 변경

### TP-004 — Old spec compatibility

training-path field가 없는 old RunSpec은 `legacy_train_v1`로 읽힌다. exact resume/Stop control을 허용하지 않는다.

### TP-005 — No silent fallback

certified spec에서 resumable entry point를 의도적으로 실패시키면 legacy `train()`이 실행되지 않고 명확한 failure가 남는다.

### TP-006 — Structural identity

`training_path_id`만 다르면 structural config hash가 달라진다. display label 변경은 hash를 바꾸지 않는다.

### TP-007 — Direct KNet path characterization

같은 fixture로 `train()`과 uninterrupted `resumable_train()`을 직접 비교한다.

비교:

- initial/final state dict
- Adam state
- train-loss sequence
- validation history
- update count
- best step/metric/state
- final prediction/metric

결과는 bitwise 또는 명시적 non-equivalence report여야 한다. 관측되지 않은 상태로 남기지 않는다.

### TP-008 — Direct Split path characterization

TP-007을 Split에 적용하고 `hn1_init`/`hn2_init`이 비교에 포함되는지 확인한다.

## 4. WorkerManager resume-child tests

### WR-001 — KNet full child lifecycle

```text
fresh certified parent uses resumable_train
→ stop at K
→ valid interrupt checkpoint
→ resume action
→ new child run
→ WorkerManager fresh process
→ restore
→ finish N
→ COMPLETED
```

continuous N-update reference와 다음이 bitwise 동일해야 한다.

- model state
- optimizer state
- full loss sequence
- validation history
- update/batch-plan cursor
- best state
- final prediction/metric

### WR-002 — Split full child lifecycle

WR-001을 Split에 적용한다. child adapter는 parent와 다른 construction seed를 사용한다.

### WR-003 — Parent immutability

resume 전후 parent의 다음 digest/snapshot이 동일하다.

- run state/version/exit code
- events file
- checkpoint package list and hashes
- artifacts
- resolved RunSpec

### WR-004 — Lineage

child에 다음이 정확히 기록된다.

- new `run_id`
- `parent_run_id`
- `resumed_from_run_id`
- `resumed_from_checkpoint_id`
- inherited `variant_id`
- inherited `training_path_id`

### WR-005 — API independence after launch

child worker launch 후 API와 Dash를 종료해도 child가 계속 진행하고, 재시작 후 동일 run/action/event cursor가 복구된다.

### WR-006 — Launch failure

WorkerManager launch failure를 주입한다.

- action `FAILED`
- child가 이미 할당됐다면 child는 명확한 terminal failure
- worker row가 허위 `RUNNING`이 아님
- retry semantics 명확

## 5. Durable action/idempotency tests

### AC-001 — Stop idempotency

동일 Stop request 5회:

- action row 1개
- interrupt checkpoint 1개
- terminal transition 1회

### AC-002 — Resume idempotency

동일 Resume request 5회:

- action row 1개
- child run 1개
- worker launch 1개

### AC-003 — Key collision

같은 idempotency key에 다른 run/checkpoint/payload를 사용하면 `409`이다.

### AC-004 — Stale state version

stale expected version은 side effect 없이 `409`이며 latest state/version을 응답한다.

### AC-005 — API crash after action row

resume action row 생성 직후 API를 종료한다. restart reconciler가 같은 action을 계속 처리하고 child는 최대 1개이다.

### AC-006 — API crash after child allocation

child directory/row 생성 후 worker launch 전 crash를 주입한다. restart 후 같은 child를 launch하거나 명시적으로 fail하며 duplicate child는 없다.

### AC-007 — API crash after worker launch

worker는 계속 실행한다. restart 후 action-child mapping과 liveness가 복구된다.

### AC-008 — Stop completion semantics

Stop action은 `INTERRUPTED`와 valid checkpoint가 모두 존재해야 `COMPLETED`이다.

### AC-009 — Resume completion semantics

Resume action은 child worker launch 성공으로 `COMPLETED`가 된다. child의 후속 training failure는 child run에만 기록된다.

## 6. Write API tests

### API-W-001 — Read-only default

`BENCH_CONTROL_ENABLE_WRITES`가 없거나 `0`이면:

- GET endpoints 정상
- POST stop/resume side effect 없음
- 명시적 `403`
- capability response가 write-disabled 표시

### API-W-002 — Loopback-only write mode

write mode + non-loopback bind는 startup에서 거부된다. public-bind override만으로 우회되지 않는다.

### API-W-003 — Stop accepted

valid request → `202`, typed action response, status URL.

### API-W-004 — Resume accepted

valid interrupt checkpoint → `202`, action id; child allocation/launch는 async service가 수행한다.

### API-W-005 — Unsupported envelope

CUDA, AMP, nonzero workers, Adaptive, MAML, ME-Split은 `422` 또는 documented `409`로 거부되고 side effect가 없다.

### API-W-006 — Invalid checkpoint

corrupt, missing, untrusted-root, schema mismatch, incompatible structural hash/dataset fingerprint는 child 생성 없이 거부된다.

### API-W-007 — Invalid state

- STOP on terminal run
- RESUME from RUNNING parent
- RESUME from non-INTERRUPTED parent through MVP endpoint

모두 side effect 없이 거부된다.

### API-W-008 — No synchronous training

API request latency가 child training duration에 묶이지 않는다. handler call stack에서 adapter training이 실행되지 않음을 test/inspection으로 확인한다.

### API-W-009 — Method inventory

이번 tranche에서 추가된 write route는 stop/resume뿐이다. launch/warm-start/terminate/config write route가 없어야 한다.

## 7. Dash UI tests

### UI-W-001 — Read-only mode

write-disabled mode에서 기존 read-only marker와 0 enabled action control을 유지한다.

### UI-W-002 — Eligible Stop button

certified resumable `RUNNING` run detail에서만 `Stop safely`가 활성화된다.

### UI-W-003 — Stop confirmation

버튼 클릭 전에 다음 의미를 확인한다.

> 현재 optimizer update를 완료한 뒤 verified interrupt checkpoint를 저장하고 종료합니다. 즉시 종료되지 않을 수 있습니다.

### UI-W-004 — Stop action progress

REQUESTED/ACKNOWLEDGED/CHECKPOINTING/COMPLETED 또는 FAILED 상태가 polling으로 갱신된다. duplicate click은 같은 key를 사용하거나 disabled된다.

### UI-W-005 — Eligible Resume button

latest valid compatible interrupt checkpoint가 있는 `INTERRUPTED` parent에서만 활성화된다.

### UI-W-006 — Unsupported reason

버튼을 단순히 숨기지 않고 다음과 같은 이유를 표시한다.

- uncertified device/precision/workers
- legacy training path
- checkpoint invalid/incompatible
- parent not terminal/interrupted
- no remaining updates
- write mode disabled

### UI-W-007 — Child navigation

resume action이 child를 만들면 child run detail deep link가 나타난다.

### UI-W-008 — Actual browser KNet workflow

Playwright가 실제 API와 Dash를 띄우고 Stop → Interrupt → Resume → child navigation을 수행한다.

### UI-W-009 — Actual browser Split workflow

UI-W-008을 Split에 적용한다.

### UI-W-010 — No premature controls

Force terminate, Warm start, Config launch, GPU queue button이 없어야 한다.

## 8. Failure and recovery tests

### FR-W-001 — Checkpoint write failure through API Stop

Stop button/API로 요청한 뒤 checkpoint write failure를 주입한다.

- parent `FAILED`
- exit 50
- action `FAILED`
- valid checkpoint row 없음
- Resume button 없음

### FR-W-002 — Child ordinary failure

resume action launch는 성공하지만 child training이 예외를 발생한다.

- resume action launch result는 보존
- child `FAILED`, exit 40
- parent unchanged

### FR-W-003 — Child SIGKILL

child는 `ORPHANED`; parent와 resume action은 변경하지 않는다.

### FR-W-004 — Registry busy/retry

bounded SQLite busy 상황에서 duplicate action/child가 발생하지 않는다.

### FR-W-005 — Event/action partial visibility

API/UI가 action row는 보지만 아직 child가 없는 transient state를 정상 표시한다.

## 9. Security and boundary tests

### SEC-W-001 — No `shell=True`

write-control path 전체에서 subprocess argv list와 process-group semantics 유지.

### SEC-W-002 — No browser direct filesystem mutation

Dash process가 registry/checkpoint path를 write-open하지 않음을 import/monkeypatch test로 확인한다.

### SEC-W-003 — Trusted-local checkpoint only

write API를 통해 외부 path/URI를 지정할 수 없다.

### SEC-W-004 — No broad CORS/public writes

wildcard CORS와 non-loopback unauthenticated write가 없어야 한다.

### SEC-W-005 — Bounded payloads

reason/idempotency/action response payload size가 제한된다.

## 10. Regression gates

- full `pytest --collect-only -q`
- full `pytest -q`
- 28 init-provenance regression
- observer/telemetry parity
- checkpoint atomicity/fault tests
- exact-resume certification/mutation probes
- graceful-stop backend tests
- previous normal/restart/failure/orphan E2E
- existing Streamlit Inspector
- read-only Dash mode
- third-party tracked diff 0
- repository `runs/`/`reports/`를 새 test가 쓰지 않음

## 11. Performance characterization

최소 측정:

- write API request p50/p95
- action-to-worker-ack latency
- stop-request-to-interrupt latency distribution on tiny fixture
- resume-request-to-child-RUNNING latency
- dashboard polling overhead

장시간/대형 모델 일반화 주장을 하지 않는다.

## 12. Definition of Done

다음을 모두 만족해야 `READY_FOR_CONFIG_LAUNCH_TRANCHE`이다.

1. certified fresh KNet/Split control run이 update 0부터 resumable path 사용
2. old/uncertified run은 controls 미지원
3. child WorkerManager resume가 KNet/Split에서 bitwise certification 통과
4. Stop/Resume API가 idempotent하고 restart-safe
5. actual Dash browser workflow 통과
6. write-disabled read-only mode 유지
7. parent immutability와 lineage 통과
8. corrupt/incompatible/uncertified requests side-effect 없이 거부
9. full regression green
10. no third-party tracked source modification
11. docs/API/operator guide 최신화
12. implementation commit을 clean worktree에서 재현 가능
