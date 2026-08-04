# CLI Coding Agent Prompt — Write-Control Tranche

당신은 AI-ADCS Neural Kalman Filter Benchmark 저장소의 **write-control tranche**를 구현하는 시니어 ML platform engineer이다.

이번 작업은 checkpoint backend를 다시 만드는 작업이 아니다. 선행 tranche에서 다음이 이미 구현·검증되었다.

- Checkpoint v1 atomic package
- KNet/Split exact resume certification
- persistent graceful-stop backend
- interrupt checkpoint
- resume child lineage planning
- read-only FastAPI checkpoint/action/lineage 조회
- Dash read-only dashboard

선행 최종 판정은 다음이다.

```text
READY_FOR_WRITE_CONTROL_TRANCHE
```

이번 작업의 핵심 연구자 결정은 다음과 같다.

> **Exact-resume 인증 대상인 새로운 control-plane run은 처음부터 `resumable_train()`을 사용한다.**

UI button부터 붙이지 마라. 아래 순서를 지킨다.

```text
canonical training path
→ WorkerManager child resume
→ durable action recovery
→ write API
→ Dash controls
→ real worker/browser E2E
```

────────────────────────────────────
0. 반드시 먼저 읽을 문서
────────────────────────────────────

저장소에서 다음 문서를 읽고 실제 코드와 대조하라.

```text
docs/benchmark_visualization/checkpoint_stop_resume_tranche_report.md
docs/benchmark_visualization/checkpoint_stop_resume_tranche_summary.json
docs/benchmark_visualization/checkpoint_resume_state_audit.md
docs/benchmark_visualization/checkpoint_v1_schema.md
docs/benchmark_visualization/exact_resume_certification_matrix.md
docs/benchmark_visualization/graceful_stop_operator_guide.md

docs/benchmark_visualization/benchmark_visualization_tool_docs/
  13_write_control_tranche_plan.md
  14_write_control_constraints_and_decisions.md
  15_write_control_acceptance_and_test_plan.md
```

문서와 코드가 다르면 실제 코드를 우선하고 불일치를 보고서에 기록하라. 문서의 구현 완료 주장을 그대로 증거로 사용하지 마라.

────────────────────────────────────
1. Authoritative checkout과 작업 보호
────────────────────────────────────

알려진 상태:

```text
repository root: /home/dss-pc-05/bench
stabilization baseline: d92cd0c
checkpoint implementation: c0eaaf7
checkpoint branch: benchmark-viz/checkpoint-stop-resume
branch is local-only
```

실제 branch HEAD에는 `c0eaaf7` 이후 docs commit이 존재할 수 있다. 먼저 확인하라.

필수 preflight:

```bash
pwd
git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git log --oneline --decorate -20
git merge-base --is-ancestor c0eaaf7 HEAD
git worktree list
git status --short
git submodule status --recursive
```

규칙:

- authoritative working tree에는 약 999개의 unrelated dirty entry가 있을 수 있다.
- authoritative tree에서 직접 개발하지 마라.
- checkpoint branch의 최신 clean commit에서 새 isolated worktree와 branch를 만든다.
- 권장 branch: `benchmark-viz/write-control`
- 권장 worktree: `/tmp/bench-write-control-tranche`
- submodule을 recursive 초기화한다.
- `git add -A`, `git commit -am`, `git clean`, `git reset --hard`, broad checkout 금지.
- staging은 explicit path만 사용한다.
- `runs/`, `reports/`, verification artifacts, temp DB, screenshots, logs, cache를 commit하지 마라.
- third-party tracked source를 수정하지 마라.
- branch를 push하지 말고 local commit만 만들며 commit hash를 보고한다. 사용자가 명시적으로 push를 요청한 경우에만 예외이다.

작업 전 snapshot을 다음에 저장하라.

```text
artifacts/benchmark_write_control/<UTC_TIMESTAMP>/preflight/
```

artifact는 commit하지 않는다.

checkout이 잘못됐거나 `c0eaaf7` ancestry가 없으면 구현하지 말고 다음으로 종료하라.

```text
WRONG_CHECKOUT_OR_UNSAFE_WORKTREE
```

────────────────────────────────────
2. Baseline 재검증
────────────────────────────────────

변경 전 clean worktree에서 다음을 실행하라.

```bash
python -m pytest --collect-only -q
python -m pytest -q
python -m pytest -q tests/test_control_checkpoint_schema_atomicity.py
python -m pytest -q tests/test_control_exact_resume_certification.py
python -m pytest -q tests/test_control_graceful_stop.py
python -m pytest -q tests/test_viz_init_provenance_comparison.py
```

알려진 선행 결과는 497 collected / 496 passed / 1 skipped이다. 실제 결과가 다르면 원인을 분류한다.

- wrong checkout
- environment/submodule
- docs-only commit difference
- pre-existing regression
- test discovery difference

기존 test를 삭제, ignore, xfail, unconditional skip하여 green으로 만들지 마라.

────────────────────────────────────
3. 이번 tranche의 정확한 범위
────────────────────────────────────

구현 대상:

1. certified control-plane fresh run의 canonical `resumable_train()` path
2. versioned `training_path_id`
3. direct `train()` vs `resumable_train()` characterization
4. resume child allocation + `WorkerManager` launch
5. durable resume-action reconciliation
6. Stop/Resume write API
7. Dash Run Detail Stop/Resume controls
8. KNet/Split real worker/browser E2E
9. docs와 reproducible commit

명시적 제외:

- fresh run launch API/UI
- config form/raw YAML editor
- warm-start API/UI
- force terminate API/UI
- GPU queue/lease enforcement
- shared GPU
- GPU/AMP/multi-worker exact resume
- Adaptive/MAML/ME-Split exact resume
- WebSocket/SSE
- remote worker
- authentication/multi-user
- third-party source patch
- arbitrary batch-midpoint resume

제외 기능을 stub route, disabled-but-promised button, false-positive capability로 노출하지 마라.

────────────────────────────────────
4. Mandatory decision: canonical control-plane training path
────────────────────────────────────

다음 조건을 모두 만족하는 새로운 supervised control-plane run은 update 0부터 `resumable_train()`을 사용해야 한다.

```text
model_id ∈ {kalmannet_tsp, split_knet}
exact certification tuple match
checkpoint schema version 1
resume boundary = optimizer_update
precision = fp32
device_class = cpu
num_workers = 0
current certified training_mode
init_id = trained OR exact-resume child
```

구현할 concrete path id:

```text
control_resumable_v1
```

기존 `train()` path id:

```text
legacy_train_v1
```

non-trainable/evaluation-only:

```text
not_applicable
```

필수 정책:

- path 선택은 resolver/launch validation에서 한 번만 수행한다.
- worker는 persisted concrete path를 실행한다.
- model name만 보고 다시 선택하지 않는다.
- certified path 준비가 실패해도 `train()`으로 fallback하지 않는다.
- old RunSpec에 field가 없으면 `legacy_train_v1`이다.
- user에게 legacy/resumable toggle을 제공하지 않는다.
- `training_path_id`는 structural config hash에 포함한다.
- RunSpec, registry, API run detail, start event, checkpoint manifest에 기록한다.
- resume child는 parent path를 상속한다.
- legacy CLI의 default `train()` behavior는 바꾸지 않는다.

필요하다면 config schema version을 올리되, old document를 silent reinterpretation하지 마라. registry 변경이 필요하면 forward-only additive migration과 backup을 사용한다.

────────────────────────────────────
5. 기존 runner와의 연결 방식
────────────────────────────────────

현재 control-plane `SuiteExecutor`와 `run_suite.run_one()` 호출 경로를 실제 코드에서 추적하라.

목표는 training loop를 복제하는 것이 아니다. 기존 runner에 좁은 internal execution contract를 전달한다.

권장 개념:

```text
ControlExecutionSpec
  training_path_id
  resume_checkpoint_id | None
  target_global_update
  run_id
  action/stop polling context
```

실제 naming은 repository convention에 맞춰도 된다.

원칙:

- `run_one()` public behavior는 default에서 legacy-compatible여야 한다.
- control worker만 explicit execution contract를 전달한다.
- fresh certified run은 `resumable_train()`.
- resume child는 checkpoint restore 후 `resumable_train()`.
- uncertified trainable run은 기존 `train()`, no Stop/Resume capability.
- evaluation-only run은 기존 behavior.
- training entry selection을 stdout text parsing이나 global mutable flag에 의존하지 않는다.
- worker start event에 chosen path와 certification id를 기록한다.

────────────────────────────────────
6. Direct `train()` vs `resumable_train()` characterization
────────────────────────────────────

선행 보고서는 두 loop가 numerical hooks를 공유하지만 직접 parity test가 없다고 명시한다. 이번 tranche에서 반드시 확인하라.

KNet과 Split 각각:

```text
A: legacy train(), uninterrupted
B: resumable_train() from update 0, uninterrupted
```

동일 조건:

- same dataset tensors/fingerprint
- same initial model state
- same optimizer hyperparameters
- same seed/RNG
- same update budget
- same validation cadence
- same intended batch order
- CPU/fp32/deterministic/0 workers

비교:

- initial/final state dict tensor bytes
- Adam state
- per-update train-loss sequence
- validation history
- update count and best state
- prediction/final metric
- Split extra state

가능하면 bitwise equality를 요구한다.

차이가 나면 tolerance를 확대하거나 legacy `train()` default를 바꾸지 마라. 원인을 조사한다.

- batch-plan ordering
- DataLoader generator consumption
- validation timing
- best-state restoration
- hidden state reset
- RNG order

차이가 본질적으로 남으면:

1. exact resume within `control_resumable_v1` certification은 유지할 수 있다.
2. `legacy_train_v1`과 `control_resumable_v1`은 structural provenance에서 구분한다.
3. 과거 run과 자동 동등 비교를 금지한다.
4. `control_plane_training_path_contract.md`에 결과와 영향 기록.
5. 최종 verdict에서 이를 condition 또는 blocker로 정직하게 판정.

self-contained pytest를 추가한다. test는 real third-party modules가 load되었음을 assert한다.

────────────────────────────────────
7. Fresh control run을 resumable path로 실행
────────────────────────────────────

다음 실제 worker tests를 먼저 통과시켜라.

- certified KNet fresh run
- certified Split fresh run

각 run에서:

- resolved spec path = `control_resumable_v1`
- legacy `train()` 미호출
- `resumable_train()` 호출
- live metrics/telemetry 유지
- final checkpoint/metrics/artifacts 정상
- full completion exit 0
- worker/API restart semantics 유지

uncertified synthetic tuple을 만들어 path가 `legacy_train_v1`이고 action eligibility가 false임을 확인한다. 실제 GPU training을 certification test로 만들지 마라.

────────────────────────────────────
8. Resume child를 WorkerManager로 실제 실행
────────────────────────────────────

현재 `plan_resume()`는 validation/lineage planning만 한다. 이를 다음 full path로 연결하라.

```text
checkpoint id
→ load manifest without unsafe payload use
→ validate digest/schema/inventory/root
→ compatibility + certification
→ parent terminal/state-version validation
→ immutable child run allocation
→ child ResolvedRunSpec
→ lineage registry transaction
→ action-child link
→ WorkerManager.launch(child)
→ fresh child process
→ checkpoint payload restore
→ resumable_train()
→ remaining updates
→ terminal state
```

필수 invariant:

- parent immutable
- child new run id/directory
- child inherits variant id
- child inherits `control_resumable_v1`
- no parent `RESUMING` transition
- child normal state machine 사용
- child first status/event indicates resume lineage
- checkpoint restored before first new optimizer update
- child adapter constructed with different seed in certification test
- exactly one worker per child

resume action `COMPLETED`는 child worker launch 성공을 의미한다. child training outcome은 child run state로 관리한다.

WorkerManager launch failure:

- action `FAILED`
- allocated child가 있다면 false RUNNING으로 남기지 않음
- traceback/reason 기록
- retry/idempotency policy 일관

CLI 기존 `resume --checkpoint-id`가 plan-only라면 semantics를 조용히 바꾸지 마라. 실행 기능은 명시적 flag 또는 새 command로 추가한다.

권장:

```bash
python -m bench.control.cli resume --checkpoint-id <id> --launch
```

without `--launch`는 기존 plan output을 유지한다.

────────────────────────────────────
9. Durable action orchestration과 recovery
────────────────────────────────────

기존 registry v2 `run_actions` schema/service를 감사하고 중복 시스템을 만들지 마라.

필수 semantics:

```text
REQUESTED
ACKNOWLEDGED
COMPLETED
FAILED
```

Stop action은 기존 completion semantics를 유지한다.

Resume action:

- REQUESTED: durable row
- ACKNOWLEDGED: coordinator가 ownership 확보
- COMPLETED: child identity 확정 + worker launch 성공
- FAILED: validation/allocation/launch 실패

idempotency:

- same key/same payload → same action/child
- same key/different payload → conflict

optimistic concurrency:

- Stop: expected parent state version
- Resume: expected parent state version + checkpoint

recovery/fault injection:

1. action row 이후 crash
2. validation 이후 crash
3. child allocation 이후 crash
4. action-child link 이후 crash
5. worker launch 직후 crash
6. API restart
7. request retry

각 point에서 duplicate child/worker가 없어야 한다.

API가 없는 동안 이미 launch된 worker는 계속 실행해야 한다. API restart 후 pending action을 reconcile한다.

────────────────────────────────────
10. Minimal write API
────────────────────────────────────

추가할 endpoint:

```text
POST /api/v1/runs/{run_id}/actions/stop
POST /api/v1/checkpoints/{checkpoint_id}/actions/resume
GET  /api/v1/actions/{action_id}
```

기존 GET endpoints는 유지한다.

write mode default:

```text
BENCH_CONTROL_ENABLE_WRITES=0
```

활성화:

```text
BENCH_CONTROL_ENABLE_WRITES=1
```

보안 경계:

- write mode는 loopback bind에서만 허용
- `BENCH_CONTROL_ALLOW_PUBLIC_BIND=1`만으로 public writes 허용 금지
- no wildcard CORS
- JSON typed request
- bounded string/payload
- external checkpoint path/URI 입력 금지
- browser가 API filesystem path를 지정할 수 없음

request contract에는 최소 다음이 있어야 한다.

Stop:

```text
idempotency_key
expected_state_version
optional bounded reason
```

Resume:

```text
idempotency_key
expected_parent_state_version
```

이번 tranche에서 learning-rate, update-budget, config override를 body에 받지 않는다.

response:

- new/in-progress action → `202 Accepted`
- same completed action → `200 OK`
- disabled → `403`
- unknown → `404`
- state/idempotency conflict → `409`
- certification/compatibility failure → `422` 또는 명시적으로 문서화된 `409`

handler 안에서 training loop를 실행하지 마라. API는 durable action을 만들고 coordinator/WorkerManager에 위임한다.

capability API를 업데이트하되 coarse model-wide exact-resume flag를 true로 만들지 마라.

run/checkpoint detail에 machine-readable eligibility를 추가한다.

```json
{
  "eligible": false,
  "reason_codes": ["UNCERTIFIED_DEVICE"],
  "messages": ["Exact resume is certified only for CPU/fp32/0 workers."],
  "training_path_id": "legacy_train_v1",
  "certification_id": null
}
```

────────────────────────────────────
11. Dash Run Detail controls
────────────────────────────────────

Runs table에 대량 action button을 넣지 말고, 첫 구현은 Run Detail에 제한한다.

### Stop safely

활성 조건:

- write mode enabled
- state `RUNNING`
- live worker
- `training_path_id = control_resumable_v1`
- exact certification tuple match
- no active stop action

confirmation text에는 다음을 포함한다.

```text
현재 optimizer update를 완료한 뒤 verified interrupt checkpoint를 저장하고 종료합니다.
즉시 종료되지 않을 수 있습니다.
```

### Resume training

활성 조건:

- write mode enabled
- parent `INTERRUPTED`
- valid compatible `interrupt` checkpoint
- `control_resumable_v1`
- exact certification tuple match
- remaining updates
- no active resume action

### UI behavior

- action pending 상태 표시
- latest action polling
- duplicate click disabled
- same idempotency key reuse
- backend conflict/error 표시
- resume child id가 생기면 deep link
- unsupported reason 표시
- API unreachable/write disabled 시 read-only degradation

다음 button은 추가하지 마라.

- Force terminate
- Warm start
- New run/config launch
- GPU queue

Dash callback은 training code, registry writer, WorkerManager를 import하지 않는다. server-side `ApiClient`만 사용한다.

────────────────────────────────────
12. Mandatory tests
────────────────────────────────────

최소 신규 test 파일을 repository convention에 맞게 추가한다. 예:

```text
tests/test_control_training_path_selection.py
tests/test_control_worker_resume_child.py
tests/test_control_write_api.py
tests/test_control_dash_write_actions.py
```

필수 test:

### Training path

- certified KNet fresh → resumable
- certified Split fresh → resumable
- legacy CLI unchanged
- old spec → legacy
- uncertified tuple → legacy/no controls
- no silent fallback
- structural hash includes path
- direct KNet train/resumable characterization
- direct Split train/resumable characterization

### Worker resume

- KNet continuous vs full worker stop/resume child bitwise
- Split continuous vs full worker stop/resume child bitwise
- different child construction seed
- parent immutable
- lineage complete
- child API-restart survival

### Actions/API

- Stop 5 retries → 1 action/1 checkpoint
- Resume 5 retries → 1 action/1 child/1 worker
- same key different payload → 409
- stale version → 409/no side effect
- write-disabled → 403/no side effect
- public write startup refused
- corrupt/incompatible/uncertified → no child
- handler returns before training completes
- API crash windows recover

### Dash/Playwright

- read-only mode no enabled action buttons
- eligible KNet Stop/Resume workflow
- eligible Split Stop/Resume workflow
- action status updates
- child link navigation
- unsupported reason
- no force/warm-start/config buttons

### Regression

- full pytest
- 28 variant regression
- observer/telemetry parity
- existing checkpoint atomicity/mutation probes
- existing graceful-stop tests
- normal/restart/failure/orphan E2E
- Streamlit Inspector
- third-party tracked diff 0

mock test만으로 승인하지 마라. 최소 KNet/Split E2E는 real worker subprocess, real SQLite, real checkpoint package, real third-party adapters를 사용한다.

all new output은 `tmp_path` 또는 timestamped verification root에 둔다.

────────────────────────────────────
13. Fault injection
────────────────────────────────────

다음을 실제로 주입하라.

- resume action row 후 API crash
- child allocation 후 API crash
- worker launch 직후 API crash
- WorkerManager launch exception
- checkpoint corruption
- checkpoint write failure during Stop
- child ordinary exception
- child SIGKILL
- SQLite busy/retry
- duplicate browser callback/retry

각 fault의 expected registry/action/run state를 보고서에 표로 기록한다.

────────────────────────────────────
14. Documentation outputs
────────────────────────────────────

다음 파일을 작성하라.

```text
docs/benchmark_visualization/
  control_plane_training_path_contract.md
  write_control_api_contract.md
  write_control_operator_guide.md
  write_control_tranche_report.md
  write_control_tranche_summary.json
```

기존 문서도 실제 behavior에 맞게 필요한 부분만 갱신한다.

```text
docs/benchmark_visualization/operator_quickstart.md
docs/benchmark_visualization/known_limitations.md
docs/benchmark_visualization/implementation_status_phase0_phase1_phase3.md
docs/benchmark_visualization/graceful_stop_operator_guide.md
docs/benchmark_visualization/exact_resume_certification_matrix.md
```

과거 tranche report를 덮어쓰지 마라.

### `control_plane_training_path_contract.md`

반드시 포함:

- Decision WC-A
- selection algorithm
- `control_resumable_v1` / `legacy_train_v1` / `not_applicable`
- old spec migration
- structural hash/identity semantics
- direct train-vs-resumable result
- historical comparison caveat

### `write_control_api_contract.md`

반드시 포함:

- endpoint schemas
- idempotency
- state version
- response/error codes
- write enablement/local-only boundary
- action completion semantics
- recovery behavior

### `write_control_operator_guide.md`

반드시 포함:

- write mode enabling
- API/Dash startup
- Stop safely workflow
- Resume training workflow
- child lineage
- retry/error handling
- unsupported envelope reasons
- what Stop/Resume are not

### tranche report

목차:

1. Executive verdict
2. Authoritative checkout/provenance
3. Safety/worktree protection
4. Baseline results
5. Canonical training-path implementation
6. Direct path parity characterization
7. WorkerManager child-resume implementation
8. Action/idempotency/recovery
9. Write API
10. Dash controls
11. KNet full E2E
12. Split full E2E
13. Fault injection
14. Security boundary
15. Full regression
16. Third-party isolation
17. Remaining risks
18. Explicitly deferred features
19. Final gate
20. Evidence index

────────────────────────────────────
15. Commit discipline
────────────────────────────────────

권장 commit 분리:

```text
feat: canonical resumable control training path + worker child resume
feat: durable write action API
feat: Dash Stop/Resume controls
test: write-control E2E and fault injection
docs: contracts, operator guide, tranche report
```

실제 dependency에 따라 조정할 수 있으나 production/test/docs를 하나의 거대한 commit으로 섞지 마라.

commit 전:

```bash
git diff --cached --check
git status --short
git diff --submodule=log -- third_party
```

explicit path staging만 사용한다.

────────────────────────────────────
16. 최종 acceptance gate
────────────────────────────────────

다음을 모두 만족해야 한다.

- certified fresh KNet/Split control run uses `control_resumable_v1`
- old/uncertified run is not mislabelled
- KNet full worker stop/resume child passes bitwise
- Split full worker stop/resume child passes bitwise
- Stop/Resume API is idempotent and restart-safe
- Dash real-browser workflows pass
- write-disabled read-only mode remains safe
- corrupt/incompatible/uncertified requests create no child
- parent immutability and lineage pass
- full pytest passes without exclusion
- 28 variant regression passes
- previous checkpoint/graceful-stop tests pass
- third-party tracked source diff is empty
- implementation is reproducible from local commits in a clean worktree

최종 verdict:

```text
READY_FOR_CONFIG_LAUNCH_TRANCHE
READY_AFTER_SPECIFIC_FIXES
NOT_READY
WRONG_CHECKOUT_OR_UNSAFE_WORKTREE
```

────────────────────────────────────
17. 작업 완료 터미널 요약
────────────────────────────────────

터미널에는 다음만 간단히 출력하라.

1. authoritative source commit과 새 branch/worktree
2. implementation/test/docs commit hashes
3. certified fresh-run training path 결과
4. KNet direct path characterization
5. Split direct path characterization
6. KNet full worker stop/resume result
7. Split full worker stop/resume result
8. write API + Dash Playwright result
9. full pytest result
10. third-party diff result
11. 최종 verdict
12. 생성한 보고서와 JSON 경로
