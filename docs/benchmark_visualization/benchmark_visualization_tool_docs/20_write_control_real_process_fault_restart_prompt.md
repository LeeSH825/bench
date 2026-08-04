# Write-Control Real-Process Fault and Restart Hardening Prompt

당신은 AI-ADCS Neural Kalman Filter Benchmark 저장소의 Benchmark Execution Visualization **Real-Process Fault and Restart Hardening** micro-tranche를 수행하는 시니어 ML 플랫폼 엔지니어이다.

이번 작업은 새로운 Stop/Resume 기능을 만드는 작업이 아니다. 현재 정상 경로는 실제 production process에서 이미 인증되었다.

- production worker가 durable Stop action을 관측한다.
- `RUNNING → STOP_REQUESTED → CHECKPOINTING → INTERRUPTED`
- validated interrupt Checkpoint v2
- worker exit code 10
- KNet real WorkerManager parent→child bitwise parity
- Split real WorkerManager parent→child bitwise parity
- Stop/Resume idempotency의 정상 경로

이번 tranche의 목적은 아직 service-level로만 확인된 실패 및 재시작 경로를 **실제 WorkerManager OS process**에서 검증하여, Write API와 Dash controls를 공개할 수 있는 마지막 backend gate를 닫는 것이다.

최종 목표 판정은 `READY_FOR_WRITE_API_UI_TRANCHE`이다.

---

## 0. Worktree와 repository 역할

### Common Git repository / user working tree

```text
/home/dss-pc-05/bench
```

- 사용자의 기존 연구 작업을 보존하는 working tree이다.
- source 수정, test 실행, artifact 생성 대상으로 사용하지 마라.
- 현재 visualization feature의 최신 checkout으로 취급하지 마라.

### Feature implementation worktree

```text
/tmp/bench-wc-tranche
```

- `benchmark-viz/write-control` branch의 최신 구현 기준이다.
- 모든 inspection, implementation, test, report 작성은 여기에서 수행한다.
- worktree가 없으면 Git branch와 commit ancestry를 확인한 뒤 복구한다.
- `/home/dss-pc-05/bench`의 dirty 파일을 복사·이동·삭제하지 마라.

모든 새 보고서 상단에 다음을 기록하라.

```yaml
git_common_repository: /home/dss-pc-05/bench/.git
user_working_tree: /home/dss-pc-05/bench
feature_worktree: /tmp/bench-wc-tranche
branch: benchmark-viz/write-control
baseline_commit: <exact sha before this tranche>
head_commit: <exact sha after this tranche>
control_root: <temporary test root>
verification_layer: real_process_fault_restart
```

---

## 1. Preflight와 ancestry

`/tmp/bench-wc-tranche`에서 다음을 먼저 실행하라.

```bash
pwd
git rev-parse --show-toplevel
git rev-parse --git-common-dir
git branch --show-current
git rev-parse HEAD
git status --short
git log --oneline --decorate -20
git worktree list
git submodule status --recursive
```

다음 의미의 commit이 ancestry에 있는지 확인하라.

- `ba65524` 또는 의미상 동일한 production graceful-stop wiring commit
- `7296294` 또는 의미상 동일한 real-worker parity test commit
- 그 이후 docs commit
- F-1 `training_path_id` propagation fix
- Checkpoint v2 및 registry migration 3
- durable Resume action / child lineage
- `control_resumable_v1`

정확한 hash가 다르면 실제 Git history에서 대응되는 commit을 식별하고 기록하라. 존재하지 않는 hash를 가정하지 마라.

필수 baseline reproduction:

```bash
python -m pytest --collect-only -q
python -m pytest -q
```

최근 보고 기준은 약 `541 collected / 540 passed / 1 skipped / 0 failed`이나, 실제 branch tip의 결과를 기준선으로 사용하라.

branch나 구현을 찾지 못하면 `WRONG_CHECKOUT_OR_LOST_WORKER_WIRING_IMPLEMENTATION`으로 종료하라.

---

## 2. 반드시 읽을 최신 문서

```text
docs/benchmark_visualization/
  production_worker_graceful_stop_contract.md
  worker_level_exact_resume_certification.md
  write_control_worker_wiring_report.md
  write_control_worker_wiring_summary.json
  write_control_real_worker_parity_report.md
  write_control_real_worker_parity_summary.json
  write_control_backend_continuation_report.md
  write_control_backend_continuation_summary.json
  checkpoint_v2_schema.md
  exact_resume_certification_matrix.md
  resume_child_worker_contract.md
  durable_resume_action_contract.md
  graceful_stop_operator_guide.md
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
  20_write_control_real_process_fault_restart_prompt.md
```

최신 코드와 실제 실행 결과가 source of truth이다. 과거 report를 덮어쓰지 마라. 현재 정상 경로를 다시 미구현으로 판단하지 마라.

---

## 3. 이미 완료된 기능 — 재작성 금지

다음을 다시 설계하거나 별도 구현으로 대체하지 마라.

- `StopCoordinator`
- `settle_graceful_stop()`
- production worker stop wiring
- `control_resumable_v1`
- Checkpoint v1/v2
- Checkpoint atomic publication
- Checkpoint validation / eligibility
- registry migration 3
- durable Stop/Resume action tables
- child allocation 및 lineage
- WorkerManager
- action reconciliation
- KNet real-worker parity
- Split real-worker parity
- Split extra-state restore
- existing idempotency logic
- read-only FastAPI
- read-only Dash

정상 경로를 위한 production code 변경은 실제 fault test가 버그를 발견한 경우에만 최소 범위로 허용한다.

---

## 4. 이번 tranche의 필수 실제-process 시나리오

다음 네 가지는 mock, direct adapter call, `SyntheticExecutor`, in-process service call, mocked `Popen`으로 대체할 수 없다.

실제 SQLite, filesystem, CheckpointService, WorkerManager, `bench.control.process.worker_cli`, production executor/runner, 실제 KNet/Split adapter를 사용하라.

모든 output은 `tmp_path` 또는 새로운 임시 `BENCH_CONTROL_ROOT`에 둔다.

### Scenario A — Real worker checkpoint-write failure

실제 running parent worker가 Stop 요청을 감지한 후 interrupt checkpoint publication이 실패했을 때 거짓 `INTERRUPTED` 또는 거짓 launch-eligible checkpoint가 생기지 않는지 검증한다.

실행:

```text
control_resumable_v1 parent worker
→ RUNNING
→ persistent Stop action
→ STOP_REQUESTED
→ CHECKPOINTING
→ 실제 checkpoint publication fault 주입
```

기존 production fault-injection seam을 우선 사용한다. payload temp write, manifest write, atomic replace, fsync, registry transaction 전후 중 exit 50 semantics를 실제 worker path에서 충분히 검증할 최소 fault set을 선정하라.

필수 assertion:

```text
final run state == FAILED
worker exit code == 50
Stop action == FAILED
INTERRUPTED transition 없음
valid checkpoint row == 0
launch-eligible checkpoint == 0
partial package가 VALID로 표시되지 않음
normal COMPLETED handler가 상태를 덮어쓰지 않음
parent lineage 및 기존 artifacts 보존
```

Reconciler 이후에도 partial package가 valid로 채택되지 않아야 한다. 완전한 package가 registry transaction 직전 생성된 경우 기존 reconciliation contract에 따른 adoption과 exit 50 의미가 모순되지 않는지 설명하라.

### Scenario B — Child restore/protocol failure before first update

정상 real-worker path로 parent를 graceful stop하여 valid launch-eligible Checkpoint v2를 만든 뒤 durable Resume action으로 actual child를 launch한다.

첫 optimizer update 이전에 다음 중 적절한 production fault를 주입한다.

- checkpoint restore failure
- expected extra-state missing
- child protocol/config mismatch
- resolved spec read failure
- implementation identity mismatch
- deliberate restore-hook exception

필수 assertion:

```text
child optimizer updates == 0
child never reports COMPLETED
child run terminal state가 contract와 일치
worker exit code가 protocol/restore failure contract와 일치
failure event와 traceback 존재
Resume action state가 정확
parent run/checkpoint/directory 불변
duplicate child == 0
duplicate worker == 0
lineage row 보존
same idempotency retry가 새 child를 만들지 않음
```

Never-started workload를 `CANCELLED`로 처리한다면 Resume action `FAILED`와 child `CANCELLED`의 의미를 명확히 구분하라. 실제 worker가 시작됐으나 restore 중 실패한 경우에는 기존 state contract를 따르고 판단 근거를 문서화하라.

### Scenario C — Child ordinary failure after restore and at least one update

```text
valid parent interrupt checkpoint
→ durable Resume action
→ actual child WorkerManager process
→ restore PASS
→ at least 1 optimizer update PASS
→ intentional ordinary exception
```

필수 assertion:

```text
child global_update > resume cursor
child final state == FAILED
ordinary failure exit code == 40
failure event / traceback artifact 존재
Resume action의 launch 완료 의미와 child training 완료 의미가 구분됨
action이 child COMPLETED를 허위로 의미하지 않음
parent 불변
source interrupt checkpoint VALID 유지
lineage 유지
duplicate child/worker 없음
retry semantics 명확
```

### Scenario D — Coordinator restart while resumed child is RUNNING

```text
parent INTERRUPTED + valid Checkpoint v2
→ durable Resume action
→ child allocation
→ WorkerManager child launch
→ child state RUNNING 확인
→ coordinator/API process 종료
→ child가 계속 update 수행하는지 확인
→ coordinator/API 재시작
→ reconcile actions/workers
→ original child 재발견
→ child terminal state까지 관찰
```

child가 충분히 오래 RUNNING 상태를 유지하도록 tiny fixture의 update 수 또는 pacing을 조절하라. sleep만으로 판단하지 말고 실제 global update progression을 확인하라.

필수 assertion:

```text
API/coordinator 종료 중 child PID alive
global_update 증가
restart 전후 run_id 동일
worker_instance_id/token 동일
child PID/start-time 동일
action_id 동일
lineage 동일
새 child allocation 없음
새 worker launch 없음
reconcile 1회 후 정상 복구
reconcile 2회는 no-op
event cursor 연속
child가 최종 terminal state에 도달
```

동일 Resume request를 restart 전후 총 5회 재전송해도 action 1개, child 1개, worker 1개여야 한다.

---

## 5. 정상 경로 회귀

Fault test 추가 후 다음을 다시 확인한다.

- KNet continuous vs parent→stop→child real-worker bitwise PASS
- Split continuous vs parent→stop→child real-worker bitwise PASS
- Split certified implementation ID가 실제 resolver 값 `bench_split_adapter_v1`과 일치
- 잘못된 과거 implementation ID가 다시 seed되지 않음

전체 heavy E2E를 매 full suite에서 수행하기 어렵다면 기존 certification test는 유지하고 이번 tranche에서 별도 실제 실행 로그를 남기되, 테스트를 삭제하거나 skip하여 green으로 만들지 마라.

---

## 6. Restart 및 idempotency 상세 검증

실제 registry에서 다음 invariants를 확인한다.

Stop 5회 동일 요청:

```text
STOP action row 1
acknowledgement 1
checkpoint 1
terminal transition 1
```

Resume 5회 동일 요청:

```text
RESUME_EXACT action 1
child run 1
worker row 1
worker process 1
```

Conflict:

- 같은 idempotency key + 다른 checkpoint → conflict, side effect 없음
- stale parent `state_version` → conflict, side effect 없음
- terminal parent state 변경 후 stale retry → conflict
- child launch 실패 후 retry policy가 contract와 일치

---

## 7. Production fix 원칙

Scenario가 실패하면 다음 순서로 첫 divergence를 국소화한다.

1. registry action state
2. run state/version
3. worker identity/token
4. child allocation transaction
5. WorkerManager launch/reap
6. checkpoint package/registry consistency
7. event sequence
8. terminal overwrite
9. reconciliation

금지:

- 실패를 tolerance로 무시
- expected state를 production 결과에 맞게 낮춤
- unconditional retry loop
- duplicate child를 사후 삭제해 통과 처리
- terminal state를 test에서 직접 수정
- fault test용 별도 fake lifecycle
- API/UI 구현으로 backend 문제를 감춤

실제 production bug가 있을 때만 최소 production fix를 적용한다. 스키마 변경이 필요하면 migration 4를 즉시 만들지 말고 필요성과 backward compatibility를 먼저 문서화하라.

---

## 8. 이번 tranche에서 구현하지 않을 것

계속 다음을 구현하지 마라.

- POST Stop API
- POST Resume API
- write-mode environment switch
- Dash Stop safely button
- Dash Resume training button
- Playwright write workflow
- Config GUI/editor
- GUI benchmark launch
- Force terminate API/UI
- Warm-start API/UI
- GPU queue/scheduler
- GPU/AMP/multi-worker exact resume
- Adaptive/MAML/ME-Split exact resume
- authentication
- remote worker
- WebSocket/SSE

FastAPI는 계속 GET/HEAD only, Dash action button은 0이어야 한다.

---

## 9. Mandatory tests

제외 옵션 없이 실행하라.

```bash
python -m pytest --collect-only -q
python -m pytest -q
```

필수 별도 결과:

- Scenario A real checkpoint-write failure
- Scenario B child pre-update restore/protocol failure
- Scenario C child post-restore ordinary failure
- Scenario D coordinator restart while child RUNNING
- real-process Stop idempotency
- real-process Resume idempotency
- KNet real-worker parity regression
- Split real-worker parity regression
- Checkpoint v1/v2 atomicity/reconciliation
- adapter/service exact-resume
- observer/telemetry numerical parity
- 28 init-provenance regression
- Streamlit Inspector import/load
- API method audit
- Dash button audit
- third-party tracked diff

새 테스트는 tracked `runs/`, `reports/`, production registry를 사용하지 마라.

---

## 10. 산출물

새 파일:

```text
docs/benchmark_visualization/
  write_control_real_process_fault_contract.md
  write_control_fault_restart_report.md
  write_control_fault_restart_summary.json
```

실제 결과에 맞게 갱신:

```text
worker_level_exact_resume_certification.md
production_worker_graceful_stop_contract.md
resume_child_worker_contract.md
durable_resume_action_contract.md
known_limitations.md
implementation_status_phase0_phase1_phase3.md
```

과거 tranche report/summary는 수정하지 마라. 새 문서에서 supersede 또는 보완한다고 명시하라.

Raw evidence:

```text
artifacts/benchmark_write_control_fault_restart/<timestamp>/
```

commit하지 마라.

---

## 11. Commit 정책

권장:

```text
1. test: real-process checkpoint/child/restart fault harness
2. fix: minimal production fixes found by the harness (only if needed)
3. docs: fault contract, report, summary, status updates
```

명시적 path만 stage하라.

금지:

- `git add -A`
- `git commit -am`
- verification artifact commit
- temp DB/run/log commit
- `/home/dss-pc-05/bench` unrelated work commit
- third-party generated files commit
- remote push

---

## 12. 최종 판정

### `READY_FOR_WRITE_API_UI_TRANCHE`

다음을 실제 process로 모두 통과한 경우에만 사용한다.

```text
A. checkpoint-write failure → FAILED / exit 50 / no false checkpoint
B. child restore/protocol failure → correct terminal semantics
C. child ordinary failure after update → FAILED / exit 40
D. coordinator restart during RUNNING child → same child recovered
Stop idempotency → 1 action / 1 checkpoint
Resume idempotency → 1 action / 1 child / 1 worker
KNet worker parity regression PASS
Split worker parity regression PASS
full pytest PASS
28 init-provenance PASS
API GET/HEAD only
Dash buttons 0
third-party tracked diff empty
```

그 외:

- `READY_AFTER_SPECIFIC_FIXES`
- `NOT_READY`
- `WRONG_CHECKOUT_OR_LOST_WORKER_WIRING_IMPLEMENTATION`

---

## 13. 완료 시 터미널 출력

1. Git common repository
2. user working tree와 untouched 여부
3. feature worktree
4. branch / baseline / new HEAD
5. 새 commits
6. production files changed
7. Scenario A 결과
8. Scenario B 결과
9. Scenario C 결과
10. Scenario D 결과
11. Stop/Resume idempotency
12. KNet/Split parity regression
13. full pytest
14. 28 init-provenance
15. API methods / Dash button count
16. third-party tracked diff
17. final verdict
18. report/summary/artifact paths

이제 이 prompt 전체를 `/tmp/bench-wc-tranche`에서 실행하라.
