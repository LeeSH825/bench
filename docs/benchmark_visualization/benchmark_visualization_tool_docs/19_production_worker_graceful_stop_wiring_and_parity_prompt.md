# Production Worker Graceful-Stop Wiring and Real-Worker Parity Prompt

당신은 AI-ADCS Neural Kalman Filter Benchmark 저장소의
Benchmark Execution Visualization **Production Worker Graceful-Stop Wiring**
micro-tranche를 수행하는 시니어 ML 플랫폼 엔지니어이다.

이번 작업은 Checkpoint, Resume coordinator, action schema, API 또는 Dash를
처음부터 다시 만드는 작업이 아니다.

현재 남은 blocker는 정확히 다음이다.

> `StopCoordinator`와 `settle_graceful_stop()` 서비스는 구현되어 있으나,
> 실제 `WorkerManager → worker_cli → SuiteExecutor → run_suite →
> _call_resumable_train()` production 경로에 연결되지 않았다.

따라서 실제 worker는 durable Stop action을 보지 못하고,
interrupt Checkpoint v2를 만들지 못한다.

이번 tranche의 목표는 이 wiring을 최소 변경으로 완성한 뒤,
실제 KNet/Split parent 및 resumed child OS process를 사용하여
continuous-vs-stop/resume bitwise parity를 인증하는 것이다.

---

## 0. Worktree와 repository 용어를 혼동하지 마라

이 작업에서는 다음 두 경로의 역할을 엄격히 분리한다.

### Common Git repository / user working tree

```text
/home/dss-pc-05/bench
```

- 사용자의 기존 ADCS, Vizard, SNN 등 대량의 dirty 작업이 존재한다.
- 이 working tree는 **보존 대상**이다.
- 여기에서 visualization production code를 수정하거나 테스트를 실행하지 마라.
- 이 경로를 최신 visualization implementation worktree라고 부르지 마라.

### Feature implementation worktree

```text
/tmp/bench-wc-tranche
```

- 현재 `benchmark-viz/write-control` branch의 구현·검증 기준이다.
- 모든 source inspection, implementation, test, report 작성은 여기에서 수행한다.
- 존재하지 않으면 Git에서 branch와 commit을 확인한 뒤 별도의 isolated worktree를 복구한다.
- user working tree의 dirty 파일을 복사하거나 이동하지 마라.

모든 보고서 첫 부분에 반드시 다음을 기록하라.

```yaml
git_common_repository: /home/dss-pc-05/bench/.git
user_working_tree: /home/dss-pc-05/bench
feature_worktree: /tmp/bench-wc-tranche
branch: benchmark-viz/write-control
head_commit: <exact sha>
control_root: <actual temporary control root>
verification_layer: production_worker
```

“repository root” 하나만 기록해서 실행 위치를 모호하게 만들지 마라.

---

## 1. 시작 전 provenance 확인

`/tmp/bench-wc-tranche`에서 다음을 실행하고 기록하라.

```bash
pwd
git rev-parse --show-toplevel
git rev-parse --git-common-dir
git branch --show-current
git rev-parse HEAD
git status --short
git worktree list
git submodule status --recursive
```

다음 구현이 실제 ancestry에 있는지 확인하라.

- checkpoint/stop/resume backend implementation
- `control_resumable_v1`
- registry migration 3
- Checkpoint v2
- durable `RESUME_EXACT` action
- child allocation/lineage coordinator
- F-1 fix: `prepare_run()`이 `training_path_id`를 registry run row에 기록
- latest known fix commit `7caeb04`, 또는 의미상 동일한 후속 commit

정확한 hash가 다르면 Git history에서 의미상 대응되는 commit을 식별하고,
그 사실을 보고서에 기록하라. 존재하지 않는 hash를 가정하지 마라.

현재 branch나 구현을 찾지 못하면 production code를 재작성하지 말고
`WRONG_CHECKOUT_OR_LOST_CONTINUATION_IMPLEMENTATION`으로 종료하라.

금지:

- `/home/dss-pc-05/bench`에서 source 변경
- `git add -A`
- `git commit -am`
- `git clean`
- `git reset --hard`
- unrelated file checkout/revert
- 기존 runs/reports/checkpoint/dataset 수정
- verification artifact commit
- third-party tracked source 수정
- 원격 push

---

## 2. 반드시 읽을 최신 근거 문서

```text
docs/benchmark_visualization/
  write_control_real_worker_parity_report.md
  write_control_real_worker_parity_summary.json
  write_control_backend_continuation_report.md
  write_control_backend_continuation_summary.json
  checkpoint_stop_resume_tranche_report.md
  checkpoint_stop_resume_tranche_summary.json
  checkpoint_resume_state_audit.md
  checkpoint_v2_schema.md
  exact_resume_certification_matrix.md
  graceful_stop_operator_guide.md
  control_plane_training_path_contract.md
  resume_child_worker_contract.md
  durable_resume_action_contract.md
```

```text
docs/benchmark_visualization/benchmark_visualization_tool_docs/
  13_write_control_tranche_plan.md
  14_write_control_constraints_and_decisions.md
  15_write_control_acceptance_and_test_plan.md
  16_write_control_implementation_prompt.md
  17_write_control_backend_continuation_prompt.md
  18_write_control_real_worker_parity_prompt.md
  19_production_worker_graceful_stop_wiring_and_parity_prompt.md
```

최신 실제 코드와 실행 결과가 source of truth이다.

과거 report의 다음 표현은 production capability 증거로 사용하지 마라.

```text
"The worker polls for its own outstanding action."
```

이 문장은 service-level behavior를 production worker behavior로 확대 기술한
과거 문구이며, F-2가 닫히기 전에는 실제 기능이 아니다.

---

## 3. 현재 확인된 상태

### 이미 구현되고 유지해야 하는 것

- stable identity와 immutable run directory
- typed `ResolvedRunSpec`
- SQLite registry와 JSONL events
- independent `WorkerManager`
- heartbeat, telemetry, failure/orphan lifecycle
- read-only FastAPI와 Dash
- Checkpoint v1/v2 package, digest, atomic publication
- KNet/Split adapter/service-level fresh-process exact resume
- Split `hn1_init` / `hn2_init` extra state
- `control_resumable_v1`
- Checkpoint v2 eligibility
- durable Resume action
- child allocation, lineage, idempotency, reconciliation
- F-1 fix: registry run row의 `training_path_id`

### 현재 open blocker

- production worker가 `StopCoordinator`를 생성하지 않음
- execution contract에 `stop_requested` callback을 넣지 않음
- `result.interrupted` 뒤 `settle_graceful_stop()`을 호출하지 않음
- 실제 worker가 interrupt Checkpoint v2를 생성하지 않음
- actual worker exit code 10 / `INTERRUPTED` 미인증
- KNet/Split WorkerManager parent→child bitwise parity 미실행

---

## 4. 이번 tranche의 production 구현 범위

### 4.1 Worker가 Stop action을 관측하도록 연결

`control_resumable_v1`인 실제 run에 대해서만,
worker 실행 경로에서 run-scoped `StopCoordinator`를 생성하라.

```text
worker_cli / worker setup
→ registry 및 run identity를 이용해 StopCoordinator 생성
→ execution contract에 최소 callback 주입
```

권장 contract:

```python
execution_contract["stop_requested"] = stop_coordinator.stop_requested
```

필수 조건:

- `legacy_train_v1`에는 연결하지 않음
- `not_applicable`에는 연결하지 않음
- callback은 요청 확인만 수행
- callback 내부에서 checkpoint를 쓰지 않음
- registry read 오류를 stop 요청으로 오인하지 않음
- polling failure를 조용히 삼키지 않음
- signal handler에서 `torch.save`나 SQLite write를 하지 않음

### 4.2 `_call_resumable_train()`에서 graceful-stop settlement 수행

실제 live adapter가 scope에 있는 production 경로에서:

```text
adapter.resumable_train(...)
→ result.interrupted == true
→ settle_graceful_stop(...)
```

를 연결하라.

`settle_graceful_stop()`에는 최소한 다음이 전달되어야 한다.

- live adapter
- run-scoped `CheckpointService`
- run directory
- registry
- observer/event writer
- current training result/state
- action/coordinator identity
- `training_path_id="control_resumable_v1"`
- Checkpoint schema v2에 필요한 provenance
- current certification tuple

새 checkpoint 구현을 만들지 말고 기존 service를 사용하라.

### 4.3 Terminal ordering

성공 시:

```text
RUNNING
→ STOP_REQUESTED
→ CHECKPOINTING
→ interrupt Checkpoint v2 durable publication
→ manifest/digest/inventory validation
→ action completion
→ INTERRUPTED
→ worker exit 10
```

Checkpoint 저장 실패 시:

```text
CHECKPOINTING
→ FAILED
→ worker exit 50
→ valid checkpoint row 없음
→ Resume launch-eligible 아님
```

### 4.4 Terminal overwrite 방지

`result.interrupted`인 경우:

- 일반 `COMPLETED` handler까지 진행하지 않음
- exit 10을 ordinary failure exit 40으로 바꾸지 않음
- reconciler가 `INTERRUPTED`를 `ORPHANED`로 바꾸지 않음
- terminal event가 중복되지 않음

---

## 5. 스키마와 설계 제약

이번 wiring은 기존 pieces를 연결하는 bounded fix이다.

원칙적으로 다음을 만들지 마라.

- registry migration 4
- Checkpoint schema v3
- 별도 action table
- 별도 checkpoint service
- 별도 worker manager
- 새로운 training loop

정말 schema 변경이 필요하면 구현 전에 보고서 초안에 이유와 대안을 기록하라.
기존 Checkpoint v1/v2 의미를 소급 변경하지 마라.

---

## 6. Production wiring 테스트

### 6.1 Wiring reachability

실제 production worker setup을 통해 증명하라.

- `control_resumable_v1` → callback 존재
- `legacy_train_v1` → callback 없음
- `not_applicable` → callback 없음
- `StopCoordinator`가 current run/action을 조회
- direct `RunRecord` construction으로 propagation gap을 숨기지 않음

### 6.2 Real interrupt checkpoint

실제 WorkerManager로 control run을 시작하고 Stop action을 기록한다.

확인:

- action acknowledge
- actual `result.interrupted`
- Checkpoint v2 package 존재
- manifest `training_path_id == control_resumable_v1`
- `launch_eligible == true`
- 상태 순서 정확
- exit code 10

### 6.3 Checkpoint failure

실제 worker checkpoint write를 의도적으로 실패시킨다.

확인:

- `FAILED`
- exit 50
- action `FAILED`
- valid checkpoint row 없음
- `INTERRUPTED` 없음
- Resume 불가

### 6.4 Real-process stop idempotency

같은 idempotency key로 Stop 요청 5회:

```text
action 1개
acknowledgement 1개
interrupt checkpoint 1개
terminal transition 1개
```

---

## 7. KNet actual WorkerManager parity E2E

Mock, `SyntheticExecutor`, direct adapter call, coordinator-only call,
mocked `Popen`으로 대체하지 마라.

### Reference A

```text
kalmannet_tsp
control_resumable_v1
CPU / fp32 / num_workers=0
N updates continuous
→ COMPLETED
```

### Parent + child B

```text
same canonical config
→ parent WorkerManager process
→ RUNNING
→ persistent Stop request
→ K_actual boundary
→ valid interrupt Checkpoint v2
→ INTERRUPTED / exit 10
→ durable Resume action
→ immutable child allocation
→ fresh child WorkerManager process
→ restore
→ N updates까지 완료
→ COMPLETED
```

Stop 시점을 hard-code하지 말고 actual checkpoint cursor를 사용하라.

### Process identity evidence

- parent/child PID, PGID, start time, token
- parent PID != child PID
- child PID != coordinator/test PID
- child command line에 child run ID
- parent/child stdout·stderr 분리
- parent run directory 불변
- child run directory 신규

### Bitwise comparison

- final model `state_dict`
- Adam optimizer state
- full training-loss sequence
- validation history
- global update
- BatchPlan id/cursor
- best state/step/metric
- prediction
- final metrics
- structural config hash
- dataset fingerprint
- variant ID
- training path ID
- implementation ID

Operational 값인 run ID, PID, timestamp, directory, telemetry는 제외할 수 있다.
Loss/event sequence에는 reset, duplicate, gap이 없어야 한다.

---

## 8. Split actual WorkerManager parity E2E

KNet과 같은 실제 process workflow를 수행한다.

추가 조건:

- child construction seed는 parent와 다름
- Checkpoint v2가 `hn1_init`, `hn2_init`를 extra state로 보유
- first child update 이전 복원
- extra-state 누락 mutation은 restore 거부 또는 parity 실패
- third-party source 변경 없음

현재 single-optimizer Split 구현을 인증하는 것이며,
paper fidelity 인증으로 표현하지 마라.

---

## 9. 실제 child failure fault cases

### 9.1 Restore/protocol failure before first update

확인:

- Resume action terminal state
- child terminal state
- worker row와 exit code
- failure event/traceback
- duplicate child 없음
- parent 불변

### 9.2 Ordinary failure after restore during child training

확인:

- restore 성공
- 최소 한 update 이후 exception
- child `FAILED`
- action launch completion과 child training completion 구분
- lineage/checkpoint 보존
- duplicate worker 없음

Never-started child를 `CANCELLED`로 처리하는 기존 정책은 유지할 수 있다.

---

## 10. Coordinator restart와 idempotency

### Stop requester/API 종료

- Stop row 기록
- requester/API 종료
- worker가 자체적으로 요청 감지
- interrupt checkpoint
- `INTERRUPTED`

### Resume recovery

다음 시점에서 coordinator를 중단하고 reconcile한다.

1. action row 생성 직후
2. child allocation 직후, launch 전

확인:

- child 1개
- worker 1개
- duplicate launch 없음
- 두 번째 reconcile no-op

동일 Resume 요청 5회:

```text
action 1개
child run 1개
worker process 1개
```

---

## 11. Mandatory regression gates

제외 옵션 없이:

```bash
python -m pytest --collect-only -q
python -m pytest -q
```

최근 baseline은 534 collected / 533 passed / 1 skipped였으나,
정확한 수치는 실제 branch tip 기준으로 기록하라.

별도 기록:

- production worker wiring
- KNet real-worker parity
- Split real-worker parity
- child failure fault tests
- real-process idempotency/restart
- checkpoint v1/v2 tests
- adapter/service exact resume
- observer/telemetry parity
- 28 init-provenance regression
- Streamlit Inspector import/load
- third-party tracked diff

테스트 삭제, unconditional skip, xfail, ignore 금지.
새 테스트는 `tmp_path` 또는 임시 `BENCH_CONTROL_ROOT`만 사용한다.

---

## 12. Read-only surface 유지

이번 tranche에서는 다음을 유지한다.

```text
FastAPI: GET / HEAD only
Dash action buttons: 0
```

구현하지 마라.

- POST Stop/Resume
- write-mode switch
- Dash Stop/Resume
- Playwright write workflow
- config GUI/launch
- force terminate
- warm start API/UI
- GPU scheduling
- GPU/AMP/multi-worker exact resume
- Adaptive/MAML/ME-Split resume
- authentication/remote worker
- WebSocket/SSE

---

## 13. 문서 산출물

새 문서:

```text
docs/benchmark_visualization/
  production_worker_graceful_stop_contract.md
  write_control_worker_wiring_report.md
  write_control_worker_wiring_summary.json
```

다음은 KNet/Split real-worker parity가 모두 통과한 경우에만 작성한다.

```text
docs/benchmark_visualization/
  worker_level_exact_resume_certification.md
```

실제 결과에 맞게 갱신:

```text
graceful_stop_operator_guide.md
resume_child_worker_contract.md
durable_resume_action_contract.md
exact_resume_certification_matrix.md
known_limitations.md
implementation_status_phase0_phase1_phase3.md
```

과거 tranche report는 덮어쓰지 마라.
최신 문서에 다음 correction을 기록하라.

```text
Historical clarification:
The checkpoint tranche certified graceful-stop behavior at the service layer.
Production WorkerManager integration was not certified until commit <new sha>.
```

Artifact:

```text
artifacts/benchmark_write_control_worker_wiring/<timestamp>/
```

commit하지 마라.

---

## 14. Commit 정책

권장:

```text
1. fix: production worker graceful-stop wiring
2. test: KNet/Split real-worker parity and fault/idempotency
3. docs: contracts, correction, certification/report/summary
```

명시적 path만 stage한다.
push하지 마라.

---

## 15. 최종 판정

### `READY_FOR_WRITE_API_UI_TRANCHE`

모두 통과한 경우에만:

- actual worker Stop polling
- valid interrupt Checkpoint v2
- state ordering/exit 10
- checkpoint failure/exit 50
- KNet bitwise worker parity
- Split bitwise worker parity
- Split mutation detection
- two child failure cases
- idempotency/restart
- full pytest
- 28 init-provenance
- FastAPI GET/HEAD-only
- Dash buttons 0
- third-party diff empty

그 외:

- `READY_AFTER_SPECIFIC_FIXES`
- `NOT_READY`
- `WRONG_CHECKOUT_OR_LOST_CONTINUATION_IMPLEMENTATION`

---

## 16. 완료 시 터미널 출력

1. Git common repository
2. user working tree와 untouched 여부
3. feature worktree
4. branch와 baseline/new HEAD
5. 새 commits
6. 변경 production files
7. production stop wiring 결과
8. interrupt checkpoint/state/exit
9. KNet real-worker parity
10. Split real-worker parity
11. child failure cases
12. idempotency/restart
13. full pytest
14. 28 init-provenance
15. API methods와 Dash button 수
16. third-party diff
17. 최종 verdict
18. report/summary/certification 경로

이제 이 prompt의 전체 내용을 `/tmp/bench-wc-tranche`에서 실행하라.
