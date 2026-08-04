# CLI Coding Agent Prompt — Write-Control Backend Continuation

당신은 AI-ADCS Neural Kalman Filter Benchmark 저장소에서 **중단된 write-control tranche를 이어서 구현하는 시니어 ML platform engineer**이다.

이번 작업은 처음부터 다시 구현하는 작업이 아니다. 이전 작업은 canonical training path까지만 완료했고, 이후 계층은 의도적으로 또는 실행 한계 때문에 남겨 두었다.

이번 continuation은 범위를 다시 작게 나눈다.

```text
[이미 완료 — 보존]
A. canonical control-plane training path

[이번 continuation에서 완료]
B0. training_path_id 영속 전파 + checkpoint manifest contract
B1. immutable resumed child의 WorkerManager 실행
C. durable Resume action + crash/restart reconciliation
D. CLI 기반 full backend E2E

[다음 별도 tranche로 미룸]
E. HTTP POST write API
F. Dash Stop/Resume controls + Playwright action workflow
```

즉, 이번 작업의 최종 목표는 **API/UI 없이도** 다음 경로가 실제 production worker에서 끝까지 동작하도록 만드는 것이다.

```text
fresh control_resumable_v1 parent
→ graceful stop
→ validated interrupt checkpoint
→ durable RESUME_EXACT action
→ immutable child run
→ WorkerManager fresh child process
→ checkpoint restore
→ remaining updates
→ child COMPLETED
```

HTTP write route와 Dash button을 얇게 먼저 붙이지 마라. backend가 독립 검증된 뒤 다음 tranche에서 연결한다.

────────────────────────────────────
0. 현재 부분 구현 상태 — 먼저 사실 확인
────────────────────────────────────

사용자가 전달한 이전 작업 결과는 다음과 같다. 이를 주장으로만 취급하고 Git과 코드에서 다시 확인하라.

```text
authoritative repository:
  /home/dss-pc-05/bench
  authoritative dirty tree는 손대지 않았음

checkpoint baseline branch:
  benchmark-viz/checkpoint-stop-resume

continuation branch:
  benchmark-viz/write-control

expected checkpoint-branch baseline:
  9454766  (contains c0eaaf7)

expected partial implementation commit:
  48b0edb

expected partial docs commit:
  dfb932d

expected isolated worktree:
  /tmp/bench-wc-tranche
```

부분 구현에서 완료되었다고 보고된 내용:

- `bench/control/training_path.py` 추가
- config `ExecutionSection` 및 structural hash에 training path 반영
- resolver에서 path를 한 번만 결정
- runner/executor가 persisted execution contract로 dispatch
- `control_resumable_v1`, `legacy_train_v1`, `not_applicable`
- old spec에 field가 없으면 영구적으로 legacy
- certified path 실패 시 legacy fallback 거부
- KNet/Split은 동일 batch sequence에서 `train()`과 `resumable_train()`이 bitwise 동일
- 두 loop의 차이는 DataLoader/RandomSampler의 shuffle RNG 소비 방식
- explicit `BatchPlan`을 유지하고 Torch 내부 RNG 소비를 흉내 내지 않음
- full pytest 515 collected / 514 passed / 1 skipped
- 기존 init-provenance regression 28 passed
- third-party tracked diff 0

미완료라고 보고된 내용:

- `training_path_id`의 registry/checkpoint/API/UI 전파
- actual resumed child의 WorkerManager launch
- durable Resume action과 restart recovery
- POST write API
- Dash Stop/Resume controls
- write-control Playwright E2E

먼저 실제 상태를 확인하라.

```bash
pwd
git rev-parse --show-toplevel
git worktree list
git branch --all --verbose --no-abbrev
git log --oneline --decorate --graph -30 benchmark-viz/write-control
git show --stat --oneline 48b0edb
git show --stat --oneline dfb932d
git merge-base --is-ancestor c0eaaf7 benchmark-viz/write-control
git status --short
git submodule status --recursive
```

판정:

- branch와 commits가 존재하고 code가 확인되면 그 tip에서 이어간다.
- `/tmp/bench-wc-tranche`가 존재하고 안전하면 그대로 사용한다.
- worktree가 사라졌다면 `benchmark-viz/write-control`의 최신 committed tip에서 새 isolated worktree를 만든다.
- checkpoint branch나 `c0eaaf7`에서 다시 시작하여 48b0edb의 작업을 재구현하지 마라.
- expected hash가 실제 Git과 다르면 실제 branch tip과 ancestry를 보고서에 기록하고, 의미가 동일한 commit을 식별한다.
- authoritative dirty tree에서 직접 수정하지 마라.

다음이면 즉시 중단한다.

```text
WRONG_CHECKOUT_OR_LOST_PARTIAL_IMPLEMENTATION
```

────────────────────────────────────
1. 반드시 읽을 문서
────────────────────────────────────

선행 checkpoint backend의 source of truth:

```text
docs/benchmark_visualization/checkpoint_stop_resume_tranche_report.md
docs/benchmark_visualization/checkpoint_stop_resume_tranche_summary.json
docs/benchmark_visualization/checkpoint_resume_state_audit.md
docs/benchmark_visualization/checkpoint_v1_schema.md
docs/benchmark_visualization/exact_resume_certification_matrix.md
docs/benchmark_visualization/graceful_stop_operator_guide.md
```

write-control normative documents:

```text
docs/benchmark_visualization/benchmark_visualization_tool_docs/
  13_write_control_tranche_plan.md
  14_write_control_constraints_and_decisions.md
  15_write_control_acceptance_and_test_plan.md
  16_write_control_implementation_prompt.md
```

부분 구현 결과:

```text
docs/benchmark_visualization/control_plane_training_path_contract.md
docs/benchmark_visualization/write_control_tranche_report.md
docs/benchmark_visualization/write_control_tranche_summary.json
```

이 continuation prompt는 **16번 전체 작업 중 backend dependency boundary까지를 의도적으로 분리한 addendum**이다.

우선순위:

```text
latest executable code
→ this continuation prompt
→ 14 constraints/decisions
→ 15 acceptance semantics
→ 13/16 original full-tranche documents
→ older general planning documents
```

14번 ADR의 의미를 바꾸지 마라. 구현상 ADR 변경이 불가피하면 기존 항목을 소급 수정하지 말고 새 ADR에서 `supersedes` 관계를 기록한다.

────────────────────────────────────
2. 작업 보호와 commit 기준
────────────────────────────────────

- authoritative repository의 약 999개 unrelated dirty entry를 그대로 보존한다.
- isolated worktree에서만 작업한다.
- `git add -A`, `git commit -am`, `git clean`, `git reset --hard`, broad checkout/revert 금지.
- staging은 명시적 path만 사용한다.
- `runs/`, `reports/`, dataset cache, 기존 checkpoint, verification artifact를 production commit에 넣지 않는다.
- 새 test output은 모두 `tmp_path` 또는 timestamped temporary control root에 둔다.
- third-party tracked source를 수정하지 않는다.
- remote push를 하지 않는다.

작업 시작 전 evidence snapshot:

```text
artifacts/benchmark_write_control_backend/<UTC_TIMESTAMP>/preflight/
```

최소 포함:

- branch/HEAD/worktree
- working-tree status
- partial commit diff
- submodule revisions/diff
- test baseline
- registry/checkpoint schema versions
- existing action/checkpoint tables

artifact는 commit하지 않는다.

────────────────────────────────────
3. 변경 전 baseline gate
────────────────────────────────────

부분 branch tip에서 다음을 실행한다.

```bash
python -m pytest --collect-only -q
python -m pytest -q
python -m pytest -q tests/test_control_training_path_selection.py
python -m pytest -q tests/test_control_checkpoint_schema_atomicity.py
python -m pytest -q tests/test_control_exact_resume_certification.py
python -m pytest -q tests/test_control_graceful_stop.py
python -m pytest -q tests/test_viz_init_provenance_comparison.py
```

알려진 partial baseline:

```text
515 collected
514 passed
1 skipped
0 failed
28 init-provenance passed
```

실제 test filename이 다르면 repository에서 해당 test를 찾는다. test를 삭제, ignore, unconditional skip, xfail하여 맞추지 마라.

baseline이 깨져 있으면 먼저 다음 중 무엇인지 분류한다.

- wrong commit/worktree
- missing submodule
- stale environment
- partial implementation regression
- unrelated test discovery

원인이 해결되지 않으면 새로운 기능을 추가하지 말고 `PARTIAL_BASELINE_BROKEN`으로 보고한다.

────────────────────────────────────
4. 완료된 Step A를 고정하라
────────────────────────────────────

다음은 완료된 계약이며 다시 설계하거나 되돌리지 마라.

```text
control_resumable_v1
legacy_train_v1
not_applicable
```

필수 유지 조건:

- certified fresh KNet/Split control run은 update 0부터 `resumable_train()`.
- legacy CLI/default `train()` semantics는 그대로 유지.
- old RunSpec은 `legacy_train_v1`.
- user toggle 없음.
- no silent fallback.
- `training_path_id`는 structural provenance.
- explicit BatchPlan을 유지.
- Torch DataLoader 내부 RNG 소비를 재현하려고 BatchPlan을 왜곡하지 않음.
- direct characterization test는 동일 intended batch sequence에서 bitwise equality를 계속 검증.

48b0edb의 path-selection logic을 다른 위치에 중복 구현하지 마라. resolver가 결정하고 worker는 persisted result를 실행하는 원칙을 유지한다.

────────────────────────────────────
5. Gate B0 — `training_path_id` 영속 전파
────────────────────────────────────

현재 config/spec에만 있는 `training_path_id`를 lifecycle 전체에 전파한다.

최소 대상:

```text
ResolvedRunSpec.execution.training_path_id
structural_config_hash
SQLite run row / RunRecord
run detail/read model
lifecycle start event
checkpoint manifest
checkpoint payload의 resolved spec snapshot
checkpoint compatibility/eligibility
exact-resume certification lookup
read-only API run/checkpoint response
Dash provenance display (action button 없음)
```

### 5.1 Registry

현재 registry version을 실제 코드에서 확인하고 **forward-only additive migration**을 추가한다.

필수:

- existing row 보존
- automatic backup
- rollback-safe transaction
- old row를 절대로 `control_resumable_v1`로 승격하지 않음
- evidence가 없는 old trainable run은 `legacy_train_v1`
- provably non-trainable run은 `not_applicable`을 사용할 수 있음
- migration 이후 API/CLI round-trip
- registry row와 resolved spec 불일치 검출

필요하다면 다음을 저장한다.

```text
training_path_id
training_path_reason_code
training_path_contract_version
```

실제 column naming은 repository convention에 맞춘다. display text만 저장하지 마라.

### 5.2 Checkpoint manifest — v1을 소급 변경하지 마라

선행 `Checkpoint v1`은 이미 공개·검증된 contract이며 manifest에 `training_path_id`가 없다.

따라서 **v1의 의미를 조용히 다시 정의하지 마라.** 기본 권장안은 다음이다.

```text
Checkpoint schema v1
- 기존 read/validate/adapter-service restore 유지
- 파일을 수정하거나 자동 upgrade하지 않음
- training path provenance를 self-contained하게 증명하지 못함
- public write-control child launch에는 부적격

Checkpoint schema v2
- training_path_id 필수
- resolved RunSpec snapshot과 일치 검증
- certification tuple에 training_path_id 포함
- 새 control_resumable_v1 run/checkpoint가 사용
```

다른 versioning 방식을 선택하려면 다음을 모두 증명하고 ADR로 기록해야 한다.

- 기존 v1 reader와 artifact 의미가 바뀌지 않음
- old package를 resumable로 과대 주장하지 않음
- manifest만 읽고 eligibility를 결정할 수 있음
- schema/version test가 명확함

가장 안전한 구현은 schema v2이다.

v2 manifest 최소 추가 필드:

```text
training_path_id
training_path_contract_version
```

그리고 다음과 정확히 일치해야 한다.

```text
manifest.training_path_id
payload.resolved_run_spec.execution.training_path_id
registry.runs.training_path_id
child/parent expected training path
certification row training_path_id
```

v1 package에 대해 다음을 구분한다.

- `VALID` artifact인가?
- adapter/service 수준 restore가 가능한가?
- write-control child launch eligibility가 있는가?

앞의 두 항목이 true여도 마지막은 false일 수 있다. 이유 code를 제공한다.

권장 reason code:

```text
CHECKPOINT_TRAINING_PATH_UNPROVEN
CHECKPOINT_SCHEMA_NOT_WRITE_CONTROL_CERTIFIED
```

### 5.3 Certification

exact-resume certification key를 다음 tuple로 확장한다.

```text
model_id
implementation_id
checkpoint_schema_version
resume_boundary
precision
device_class
num_workers
training_mode
training_path_id
```

model 이름만 보고 true를 반환하지 않는다.

새 schema/path 조합을 KNet/Split에서 다시 certification하고 registry seed row를 추가한다. 기존 v1 certification evidence를 삭제하지 않는다.

### 5.4 Read-only exposure

아직 POST route와 action button은 추가하지 않는다.

그러나 기존 GET response와 Run Detail provenance에는 최소 다음이 보여야 한다.

```json
{
  "training_path_id": "control_resumable_v1",
  "training_path_reason_code": "CERTIFIED_CONTROL_PATH",
  "exact_resume_eligibility": {
    "eligible": true,
    "reason_codes": [],
    "certification_id": "..."
  }
}
```

unsupported/legacy에는 정확한 reason을 반환한다.

Dash는 이 tranche에서 provenance와 non-action eligibility 설명만 표시할 수 있다. Stop/Resume button은 계속 0개여야 한다.

────────────────────────────────────
6. Gate B1 — 실제 Resume child를 `WorkerManager`로 실행
────────────────────────────────────

현재 `plan_resume()`의 validation/lineage planning을 실제 child execution으로 연결한다.

필수 flow:

```text
checkpoint_id
→ manifest-only trust/schema/digest/inventory validation
→ registry parent/checkpoint lookup
→ exact certification + training_path match
→ parent terminal + expected state_version validation
→ durable RESUME_EXACT action
→ immutable child allocation
→ child ResolvedRunSpec persist
→ child lineage persist
→ action-child link
→ WorkerManager.launch(child)
→ fresh child process
→ trusted checkpoint payload restore
→ resumable_train()
→ remaining updates
→ child terminal state
```

### 6.1 Trust boundary

- API/UI가 없는 이번 tranche에서도 coordinator는 manifest와 registry로 eligibility를 먼저 판단한다.
- `torch.load`/pickle payload는 approved control root, digest, manifest validation 이후 **child worker 또는 bounded trusted backend**에서만 읽는다.
- browser/CLI request가 arbitrary path를 전달하게 하지 않는다.

### 6.2 Child invariants

child는 다음을 가진다.

```text
new run_id
new immutable run directory
parent_run_id
resumed_from_run_id
resumed_from_checkpoint_id
same variant_id
same training_path_id = control_resumable_v1
same structural config hash
same dataset fingerprint
same implementation identity
target global update = parent original total target
```

operational field만 새 child에 맞게 바꾼다.

parent의 다음 항목은 byte/row 수준에서 불변이어야 한다.

- state/state_version/exit code
- events
- artifacts
- checkpoint package
- checkpoint rows
- run directory file list

parent를 `RESUMING` 또는 `RUNNING`으로 바꾸지 않는다.

child는 일반 lifecycle을 사용한다.

```text
CREATED → VALIDATING → QUEUED → STARTING → RUNNING → terminal
```

resume provenance는 lineage/start event로 표현한다.

### 6.3 Restore ordering

child worker는 첫 새로운 optimizer update 이전에 다음을 완료해야 한다.

- checkpoint validation recheck
- model/optimizer/RNG/BatchPlan cursor/best state/validation state restore
- Split conditional extra state restore
- `training_path_id` match
- target update budget sanity check

restore 실패 시 legacy train으로 fallback하지 않는다.

권장 terminal:

- pre-training compatibility/restore failure → validation/protocol failure contract에 맞는 nonzero exit
- training exception → existing ordinary failure exit
- false `RUNNING`/`COMPLETED` 금지

### 6.4 WorkerManager launch failure

child allocation 이후 launch가 실패할 수 있다.

필수:

- action `FAILED`
- child가 있으면 명확한 failed/cancelled-before-start terminal evidence
- worker row를 false live 상태로 남기지 않음
- retry가 같은 child를 재사용하거나 명시적으로 fail
- duplicate child 없음

### 6.5 CLI

기존 CLI `resume --checkpoint-id`가 plan-only라면 그 의미를 유지한다.

실제 launch는 명시적으로 분리한다.

권장:

```bash
python -m bench.control.cli resume \
  --checkpoint-id <id> \
  --launch \
  --idempotency-key <key> \
  --expected-parent-state-version <n>
```

repository convention에 더 맞는 새 subcommand를 사용해도 된다. plan-only command를 조용히 실행형으로 바꾸지 않는다.

────────────────────────────────────
7. Gate C — Durable Resume action과 recovery
────────────────────────────────────

기존 registry v2의 `run_actions`와 Stop action service를 재사용한다. 별도 action DB/JSON store를 만들지 않는다.

필수 action type:

```text
RESUME_EXACT
```

필수 states:

```text
REQUESTED
ACKNOWLEDGED
COMPLETED
FAILED
```

의미:

- `REQUESTED`: durable action row 존재
- `ACKNOWLEDGED`: coordinator ownership 확보
- `COMPLETED`: exactly one child identity 확정 + worker launch 성공
- `FAILED`: validation/allocation/launch 실패

child의 후속 training outcome은 child run state이다. 이미 `COMPLETED`인 launch action을 child failure 때문에 다시 `FAILED`로 바꾸지 않는다.

### 7.1 Idempotency

모든 resume launch는 idempotency key를 요구한다.

```text
same key + same logical payload
→ same action
→ same child
→ same worker launch result

same key + different checkpoint/parent/version/payload
→ conflict
```

동일 요청 5회에서 다음이 하나여야 한다.

- action row
- child run
- child directory
- worker instance/process

### 7.2 Optimistic concurrency

Resume에는 최소 다음을 검사한다.

```text
expected_parent_state_version
checkpoint_id
parent terminal state
checkpoint still valid/compatible
```

stale request는 side effect 없이 conflict이다.

### 7.3 Coordinator ownership/recovery

API process에 의존하지 않는 backend coordinator/service를 구현한다.

crash window:

1. action row 직후
2. validation 직후
3. child row/directory allocation 직후
4. action-child link 직후
5. worker launch 직후, action completion 전
6. process restart
7. same request retry

각 window에서 restart reconciliation이 다음 중 하나를 수행해야 한다.

- 같은 child를 안전하게 계속 처리
- worker identity를 확인하고 action 완료
- 명시적 FAILED 처리

절대 duplicate child/worker를 만들지 않는다.

WorkerManager의 PID/start-time/token 방어를 그대로 사용한다. PID만 보고 worker를 재사용하거나 kill하지 않는다.

### 7.4 Stop action regression

Resume action을 추가하면서 기존 Stop action semantics를 바꾸지 않는다.

```text
RUNNING → STOP_REQUESTED → CHECKPOINTING → INTERRUPTED
```

valid interrupt checkpoint 이후에만 Stop action `COMPLETED`이다.

────────────────────────────────────
8. 이번 continuation에서 추가하지 말아야 할 것
────────────────────────────────────

이번 작업에서는 다음을 구현하지 마라.

- `POST /api/v1/...` write routes
- Dash `Stop safely` button
- Dash `Resume training` button
- write-mode environment switch의 public UI 동작
- Playwright Stop/Resume action workflow
- GUI config editor/launch
- force terminate
- warm-start API/UI
- GPU queue/scheduler
- GPU/AMP/multi-worker exact resume
- Adaptive/MAML/ME-Split resume
- authentication/multi-user
- WebSocket/SSE
- third-party source patch

기존 FastAPI는 계속 GET/HEAD-only여야 한다. 기존 Dash action button 수는 계속 0개여야 한다.

이 제외는 기능 취소가 아니다. backend가 `READY_FOR_WRITE_API_UI_TRANCHE`가 된 뒤 다음 prompt에서 연결한다.

────────────────────────────────────
9. Mandatory tests
────────────────────────────────────

mock-only test로 승인하지 마라. SQLite, filesystem, checkpoint package, real subprocess, WorkerManager, real KNet/Split adapters를 사용한다.

### 9.1 Persistence/schema

- registry migration preserves existing rows
- old rows never become `control_resumable_v1`
- RunSpec ↔ registry ↔ API GET round-trip
- registry/spec mismatch detected
- lifecycle start event contains training path
- v1 checkpoint unchanged/readable
- v1 write-control launch ineligible with explicit reason
- v2 manifest/payload/registry training path equality
- unknown/future checkpoint schema refused
- certification lookup includes training path
- structural hash sensitivity retained

### 9.2 Fresh-run path regression

KNet/Split 각각:

- fresh certified control run uses `control_resumable_v1`
- `resumable_train()` called
- legacy `train()` not called
- old/uncertified spec remains legacy
- no silent fallback
- existing direct loop characterization remains bitwise for same batch sequence

### 9.3 KNet full worker stop/resume

```text
continuous reference: N updates

parent: K updates
→ persistent graceful Stop
→ interrupt checkpoint schema v2
→ INTERRUPTED
→ RESUME_EXACT action
→ child WorkerManager fresh process
→ restore
→ N updates total
→ COMPLETED
```

비교:

- final model tensor bytes
- Adam state
- full per-update loss sequence
- validation history
- global update and BatchPlan cursor
- best state/step/metric
- final prediction/metric if available
- parent immutability
- child lineage

### 9.4 Split full worker stop/resume

KNet과 동일하며 추가로:

- child adapter construction seed는 parent와 다름
- `hn1_init`/`hn2_init` extra state restored
- extra state 제거 mutation은 restore refusal 또는 divergence detection
- third-party tracked source unchanged

### 9.5 Idempotency/restart

- same Resume request 5회 → action 1, child 1, worker 1
- same key different payload → conflict
- stale parent version → no child
- crash after action row → recovery
- crash after child allocation → same child or explicit failure
- crash after launch → worker identity recovered, no second launch
- coordinator restart while child RUNNING → child continues
- API process가 꺼져 있어도 backend/worker lifecycle은 완료

### 9.6 Failure injection

- corrupt checkpoint
- manifest/payload training path mismatch
- schema v1 public-launch attempt
- certification mismatch
- dataset/structural/implementation mismatch
- child allocation failure
- WorkerManager launch exception
- child restore exception
- child ordinary training exception
- child SIGKILL → ORPHANED; parent/action unchanged
- bounded SQLite busy/retry

### 9.7 Regression

반드시 제외 옵션 없이 실행한다.

```bash
python -m pytest --collect-only -q
python -m pytest -q
python -m pytest -q tests/test_viz_init_provenance_comparison.py
```

그리고 별도 기록:

- training path targeted tests
- checkpoint v1/v2 tests
- KNet/Split worker child E2E
- graceful-stop tests
- observer/telemetry parity
- normal/restart/failure/orphan lifecycle
- read-only API GET-only test
- Dash action buttons 0 test
- Streamlit Inspector import/load
- third-party tracked diff

기존 test를 삭제, ignore, xfail, unconditional skip하여 green으로 만들지 않는다.

────────────────────────────────────
10. 문서 산출물
────────────────────────────────────

부분 구현 문서는 지우지 말고 continuation provenance를 보존하면서 갱신한다.

### 새로 작성

```text
docs/benchmark_visualization/
  resume_child_worker_contract.md
  durable_resume_action_contract.md
  write_control_backend_continuation_report.md
  write_control_backend_continuation_summary.json
```

### 실제 동작에 맞게 갱신

```text
docs/benchmark_visualization/control_plane_training_path_contract.md
docs/benchmark_visualization/checkpoint_v1_schema.md
docs/benchmark_visualization/exact_resume_certification_matrix.md
docs/benchmark_visualization/graceful_stop_operator_guide.md
docs/benchmark_visualization/known_limitations.md
docs/benchmark_visualization/implementation_status_phase0_phase1_phase3.md
```

Checkpoint schema v2를 채택하면 기존 `checkpoint_v1_schema.md`를 v2 문서로 덮어쓰지 마라. 다음을 새로 작성한다.

```text
docs/benchmark_visualization/checkpoint_v2_schema.md
```

그리고 v1 문서에는 successor link만 필요한 범위에서 추가한다.

### 기존 partial tranche 문서

```text
docs/benchmark_visualization/write_control_tranche_report.md
docs/benchmark_visualization/write_control_tranche_summary.json
```

이 두 파일은 다음 중 한 방식으로 처리한다.

1. partial state를 명시한 채 “backend continuation pending” pointer를 추가하거나,
2. final full write-control tranche 전까지 historical partial report로 그대로 둔다.

부분 결과를 삭제하거나, 당시 API/UI가 구현된 것처럼 소급 수정하지 마라.

### 아직 작성하지 않음

실제 write API/UI가 없으므로 다음 파일은 이번 continuation에서 작성하지 마라.

```text
write_control_api_contract.md
write_control_operator_guide.md
```

존재하지 않는 동작을 문서화하지 않는다.

### continuation report 목차

1. Executive verdict
2. Source branch/commits/worktree
3. Partial tranche inheritance
4. Baseline tests
5. Training-path persistence
6. Checkpoint versioning decision
7. Certification update
8. Resume child architecture
9. Durable action/recovery
10. CLI execution workflow
11. KNet full worker result
12. Split full worker result
13. Fault injection
14. Parent immutability/lineage
15. Read-only API/UI non-regression
16. Full pytest/regressions
17. Third-party isolation
18. Remaining API/UI work
19. Final gate
20. Evidence index

────────────────────────────────────
11. Commit discipline
────────────────────────────────────

48b0edb와 dfb932d를 유지하고 그 뒤에 continuation commits를 추가한다.

권장 분리:

```text
feat: persist training path and add checkpoint schema v2
feat: launch immutable resume children through WorkerManager
feat: add durable resume-action reconciliation and CLI launch

test: add worker-level resume, idempotency and crash-recovery gates

docs: add child/action contracts and backend continuation report
```

실제 dependency에 따라 조정 가능하나 explicit path staging만 사용한다.

commit 전:

```bash
git diff --cached --check
git status --short
git diff --submodule=log -- third_party
```

verification artifact, temp control root, DB/WAL/SHM, run, log, screenshot, cache를 commit하지 마라.

push하지 마라.

────────────────────────────────────
12. Final acceptance gate
────────────────────────────────────

다음을 모두 만족해야 한다.

- partial Step A behavior preserved
- `training_path_id` persisted across spec/registry/event/checkpoint/API GET/provenance
- old rows/spec/checkpoints are never falsely upgraded
- checkpoint versioning is explicit and backward-safe
- new control checkpoint is self-describing for training path
- certification tuple includes training path
- KNet child launched by real WorkerManager and bitwise matches continuous reference
- Split child launched by real WorkerManager and bitwise matches continuous reference
- Split extra state mutation remains detectable
- parent immutable and lineage complete
- Resume action is durable, idempotent, and restart-safe
- same request cannot create duplicate child/worker
- CLI plan-only behavior preserved; launch is explicit
- Stop backend regression passes
- full pytest passes without exclusions
- 28 init-provenance tests pass
- GET-only API and zero Dash action buttons remain true
- third-party tracked diff is empty
- implementation is reproducible from local commits in a clean worktree

최종 verdict는 다음 중 하나이다.

```text
READY_FOR_WRITE_API_UI_TRANCHE
READY_AFTER_SPECIFIC_FIXES
NOT_READY
WRONG_CHECKOUT_OR_LOST_PARTIAL_IMPLEMENTATION
```

`READY_FOR_WRITE_API_UI_TRANCHE`는 POST route와 Dash button이 구현됐다는 뜻이 아니다. 그 계층을 안전하게 추가할 backend dependency가 완성됐다는 뜻이다.

────────────────────────────────────
13. 완료 시 터미널 요약
────────────────────────────────────

완료 시 다음만 구조적으로 출력한다.

1. authoritative repository와 isolated worktree
2. continuation baseline branch/commit
3. 기존 partial commits 확인 결과
4. 새 implementation/test/docs commit hashes
5. registry training-path propagation 결과
6. checkpoint versioning과 v1 compatibility 결과
7. KNet WorkerManager child-resume 결과
8. Split WorkerManager child-resume 결과
9. Resume action idempotency/restart 결과
10. CLI resume-launch 결과
11. full pytest 결과
12. 28 init-provenance 결과
13. API GET-only / Dash 0-action-button 결과
14. third-party diff 결과
15. 남은 API/UI scope
16. 최종 verdict
17. 생성한 report/JSON 경로

이제 위 문서를 모두 읽고, **48b0edb/dfb932d 이후에서 backend continuation만 완료하라.**
