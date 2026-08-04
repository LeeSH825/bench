# Benchmark Execution Visualization — Write-Control Tranche Plan

작성 기준일: 2026-07-31  
선행 구현 기준: `benchmark-viz/checkpoint-stop-resume` branch, checkpoint implementation commit `c0eaaf7` 및 후속 문서 수정 commit  
근거 보고서: `docs/benchmark_visualization/checkpoint_stop_resume_tranche_report.md`  
선행 gate: `READY_FOR_WRITE_CONTROL_TRANCHE`

## 1. 이 문서의 역할

이 문서는 checkpoint v1, exact resume, graceful-stop backend가 완료된 이후의 **write-control tranche 전용 계획**이다. 기존 장기 계획인 `02_target_architecture_and_mvp_plan.md`와 checkpoint backend 계획인 `09`~`11`을 대체하지 않고, 현재 구현 상태에 맞춰 다음 단계만 구체화한다.

문서 우선순위는 다음과 같다.

1. 실제 최신 코드와 최신 검증 보고서
2. 본 문서와 `14_write_control_constraints_and_decisions.md`
3. `15_write_control_acceptance_and_test_plan.md`
4. `09`~`11` checkpoint/stop/resume backend 문서
5. `02`~`06` 범용 아키텍처 문서

이번 tranche의 핵심은 화면에 버튼을 먼저 붙이는 것이 아니다. 다음 순서를 지킨다.

```text
canonical control-plane training path
→ resumed child WorkerManager execution
→ durable action orchestration
→ write API
→ Dash action controls
→ real worker/browser E2E
```

## 2. 확정된 연구자 결정

### Decision WC-A — 인증 대상 control-plane run은 처음부터 `resumable_train()`을 사용한다

다음 조건을 모두 만족하는 **새로운 control-plane supervised run**은 첫 optimizer update부터 checkpointable training path를 사용한다.

- `model_id ∈ {kalmannet_tsp, split_knet}`
- 실행의 exact-resume certification tuple이 registry의 인증 행과 정확히 일치
- CPU
- fp32
- `num_workers = 0`
- 완료된 optimizer-update boundary
- 현재 인증된 single-optimizer training mode
- `init_id = trained` 또는 exact-resume child run

이 경우 training path는 사용자 선택 옵션이 아니다. resolver가 다음 concrete contract를 기록한다.

```text
training_path_id = control_resumable_v1
```

반대로 다음 경우에는 exact resume를 광고하지 않는다.

- GPU/CUDA/MPS
- fp16/bf16/AMP
- `num_workers != 0`
- distributed 또는 gradient accumulation
- Adaptive/MAML/ME-Split
- legacy run
- 기존 `model.pt` weight-only load
- control plane 밖에서 실행되는 기존 CLI

기존 `train()`은 삭제하거나 의미를 바꾸지 않는다. 기존 CLI, 과거 run, 비교 재현을 위한 **legacy training path**로 유지한다.

```text
training_path_id = legacy_train_v1
```

old `ResolvedRunSpec`에 training-path field가 없으면 소급해서 resumable이라고 간주하지 않고 `legacy_train_v1`로 해석한다.

## 3. 현재 기준선

선행 tranche에서 다음이 구현·검증되었다.

- checkpoint v1 typed manifest/payload
- atomic publication, digest, schema/inventory validation
- crash reconciliation
- `periodic` / `best` / `interrupt` / `final` checkpoint kinds
- KNet exact resume certification
- Split-KNet exact resume certification
- Split의 `state_dict()` 밖 `hn1_init` / `hn2_init` 보존
- persistent graceful-stop request
- `RUNNING → STOP_REQUESTED → CHECKPOINTING → INTERRUPTED`
- checkpoint write failure → `FAILED`, exit 50
- resume child lineage planning과 parent immutability
- CLI 기반 checkpoint 조회/검증, stop 요청, resume planning
- read-only FastAPI checkpoint/lineage/action 조회
- API GET-only, Dash action button 0개
- clean worktree full pytest 497 collected / 496 passed / 1 skipped

남은 핵심 gap은 다음이다.

1. `plan_resume()`가 child lineage를 계산하지만 `WorkerManager`를 통해 실제 child worker를 시작하지 않는다.
2. control-plane fresh run이 자동으로 `resumable_train()`을 선택하지 않는다.
3. public write API가 없다.
4. Dash에서 Stop/Resume action을 요청할 수 없다.
5. `train()`과 `resumable_train()`의 직접 parity가 아직 특성화되지 않았다.

## 4. Tranche 목표

### 목표 A — Canonical control-plane training path

인증된 새 control-plane run은 update 0부터 `resumable_train()`을 사용한다. 선택 결과는 RunSpec, structural hash, registry, event, checkpoint manifest에 기록한다.

### 목표 B — Full worker-level resume execution

validated checkpoint에서 immutable child run을 만들고, `WorkerManager`가 fresh child process를 실행하며, child worker가 checkpoint를 복원해 남은 update를 수행한다.

### 목표 C — Durable write actions

Stop과 Resume를 durable `run_actions` 기반으로 처리한다. API/UI retry, API restart, process crash가 중복 checkpoint나 중복 child run을 만들지 않아야 한다.

### 목표 D — Minimal write API

다음 두 action만 HTTP write surface로 추가한다.

```text
POST /api/v1/runs/{run_id}/actions/stop
POST /api/v1/checkpoints/{checkpoint_id}/actions/resume
```

Fresh launch, warm start, force terminate, config edit는 이번 tranche에 포함하지 않는다.

### 목표 E — Dash action controls

Run Detail에서만 다음을 제공한다.

- `Stop safely`
- `Resume training`
- action progress/status
- child-run deep link
- unsupported reason

### 목표 F — Real worker/browser certification

KNet과 Split에 대해 CLI/API/Dash 요청부터 child worker completion까지 end-to-end로 검증한다.

## 5. 이번 tranche의 포함 범위

포함:

- `training_path_id`와 control execution contract
- certified fresh control run의 `resumable_train()` 강제 선택
- old spec의 legacy interpretation
- direct `train()` vs `resumable_train()` characterization
- resumed child run allocation
- `WorkerManager` child launch
- resume action crash recovery/reconciliation
- stop/resume action idempotency
- optimistic concurrency/state-version 검증
- local-only opt-in write API
- Dash Run Detail action controls
- run/checkpoint-specific eligibility reason
- Playwright real-browser action E2E
- operator 문서와 API contract
- full regression 및 legacy Inspector non-regression

## 6. 명시적 제외 범위

다음은 구현하지 않는다.

- GUI config editor 또는 fresh-run launch API
- warm-start API/UI
- force terminate API/UI
- GPU queue/lease enforcement
- shared GPU execution
- GPU/AMP/multi-worker exact-resume certification
- Adaptive/MAML/ME-Split exact resume
- arbitrary batch-midpoint resume
- WebSocket/SSE migration
- remote worker
- authentication/multi-user
- public unauthenticated write deployment
- third-party source 수정
- 기존 `train()`의 default semantics 변경
- 과거 run을 resume-certified로 소급 표시

## 7. Canonical training-path selection

resolver는 worker가 실행되기 전에 concrete path를 결정한다.

```text
non-trainable or init != trained
    → training_path_id = not_applicable

trainable + exact certification tuple matches
    → training_path_id = control_resumable_v1

trainable + certification tuple does not match
    → training_path_id = legacy_train_v1
    → exact_resume_eligible = false
    → graceful_stop_control_eligible = false
```

worker는 persisted `training_path_id`를 실행할 뿐, model 이름으로 다시 추측하지 않는다.

인증 대상 run에서 `resumable_train()` 준비가 실패하면 `train()`으로 silently fallback하지 않는다. validation 또는 worker protocol failure로 정직하게 실패한다.

## 8. 단계별 계획

### Gate 0 — Provenance와 안전한 작업 공간

- authoritative root `/home/dss-pc-05/bench` 확인
- checkpoint implementation commit `c0eaaf7` ancestry 확인
- 최신 checkpoint branch HEAD와 docs correction commit 확인
- dirty authoritative tree snapshot
- isolated clean worktree와 새 branch 생성
- submodule revision/dirty state 기록
- baseline full pytest와 checkpoint targeted test 재실행

완료 조건:

- 올바른 checkpoint branch를 기준으로 작업한다.
- unrelated ADCS/Vizard/SNN/Phase 작업을 수정·삭제·commit하지 않는다.

### Gate 1 — Training-path contract와 migration

- RunSpec에 versioned `training_path_id` 추가
- structural hash 포함 여부 고정
- old spec → `legacy_train_v1`
- registry/API/event/checkpoint에서 path 조회 가능
- certification tuple로 path 결정
- legacy CLI path unchanged

완료 조건:

- 같은 model name만으로 resumable path가 선택되지 않는다.
- 인증 tuple이 정확히 맞는 fresh control run만 `control_resumable_v1`이다.

### Gate 2 — Direct path characterization

KNet/Split 각각에 대해 같은 data, seed, initial model state, update budget, batch order에서 다음을 비교한다.

```text
legacy train()
vs
resumable_train() from update 0, no interruption
```

비교:

- initial/final model state
- optimizer state
- full train-loss sequence
- validation history
- best state
- update count
- prediction/final metric

bitwise 동일하면 `legacy_equivalence = certified`로 기록한다.

동일하지 않으면 legacy path를 억지로 바꾸지 않는다. 원인을 기록하고 `training_path_id`를 structural identity에 반영하여 과거 run과 자동 동등 비교를 막는다. 이 경우 write-control 진행 가능 여부를 보고서에서 명시적으로 판정한다.

### Gate 3 — Fresh control run의 resumable execution

- KNet/Split certified fresh run이 `resumable_train()`을 호출하는지 실제 worker test
- update 0 periodic/internal state 준비
- stop polling과 checkpoint hooks가 처음부터 활성
- completed run의 final checkpoint와 progress state 정합성

완료 조건:

- legacy `train()` 호출 없이 certified run이 정상 completion한다.
- numerical observer/telemetry parity가 유지된다.

### Gate 4 — Child resume WorkerManager execution

```text
validated interrupt checkpoint
→ plan_resume()
→ immutable child allocation
→ child RunSpec + lineage persist
→ WorkerManager.launch(child)
→ fresh child process
→ checkpoint restore
→ resumable_train()
→ COMPLETED
```

완료 조건:

- parent directory/state/events/checkpoints immutable
- child에 새 `run_id`
- parent `variant_id` 상속
- lineage fields 모두 존재
- continuous run과 child result가 certification 기준 내 동일

### Gate 5 — Resume action recovery

- action request persistence
- same-key idempotency
- child allocation/launch crash windows
- API restart 후 pending action reconciliation
- launch failure classification
- one action → at most one child run/worker

완료 조건:

- 반복 요청과 API crash가 duplicate child를 만들지 않는다.

### Gate 6 — Write API

- write mode opt-in
- loopback-only enforcement
- typed request/response
- state-version concurrency
- `202 Accepted`
- run/checkpoint eligibility reasons
- no synchronous training in handler
- no direct frontend registry writes

완료 조건:

- disabled mode에서는 read-only behavior 유지
- enabled mode에서는 stop/resume action만 노출
- unsupported requests가 구체적 4xx로 거부

### Gate 7 — Dash Run Detail controls

- Stop safely confirmation
- Resume training confirmation
- action pending/progress/error
- disabled reason
- child deep link
- read-only mode marker
- duplicate-click 방지

완료 조건:

- UI gating과 backend gating이 일치한다.
- UI를 우회해도 backend가 unsupported request를 거부한다.

### Gate 8 — End-to-end certification

KNet과 Split 각각:

```text
fresh control run uses resumable_train()
→ API/Dash Stop safely
→ validated interrupt checkpoint
→ API/Dash Resume training
→ child WorkerManager process
→ COMPLETED
→ continuous control과 bitwise comparison
```

추가:

- API restart during STOP_REQUESTED
- API restart during child RUNNING
- same stop request 5회
- same resume request 5회
- corrupt/incompatible/uncertified rejection
- actual Playwright DOM interaction

### Gate 9 — Regression, docs, release baseline

- full pytest
- 28 variant regression
- existing lifecycle E2E
- observer/telemetry parity
- checkpoint atomicity/fault tests
- third-party tracked diff 0
- generated output only under tmp/control roots
- implementation/report commits separated

## 9. 첫 end-to-end milestone

> CPU/fp32/0-worker Split-KalmanNet control run이 처음부터 `resumable_train()`으로 실행된다. Dash Run Detail에서 Stop safely를 눌러 interrupt checkpoint를 만들고 `INTERRUPTED`가 된다. Resume training을 눌러 immutable child run이 생성되고 `WorkerManager`의 fresh process에서 남은 update를 수행한다. child가 `COMPLETED`가 되며 continuous reference와 bitwise 동일하다. API와 Dash를 중간에 재시작해도 lifecycle과 action 상태가 복구된다.

KNet에도 동일한 milestone을 적용한다.

## 10. 최종 판정

다음 중 하나를 사용한다.

- `READY_FOR_CONFIG_LAUNCH_TRANCHE`
- `READY_AFTER_SPECIFIC_FIXES`
- `NOT_READY`
- `WRONG_CHECKOUT_OR_UNSAFE_WORKTREE`

`READY_FOR_CONFIG_LAUNCH_TRANCHE`는 다음이 모두 참일 때만 사용한다.

- certified fresh control runs use `control_resumable_v1`
- child WorkerManager resume E2E passes for KNet/Split
- Stop/Resume API and Dash controls pass real-browser E2E
- idempotency and crash recovery pass
- unsupported envelopes are refused
- full pytest and legacy regressions pass
- no third-party tracked source change
