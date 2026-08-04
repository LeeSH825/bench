# Write API, Dash Controls, and Playwright Workflow Implementation Prompt

당신은 AI-ADCS Neural Kalman Filter Benchmark 저장소의 Benchmark Execution Visualization **Write API / Dash Controls Integration** tranche를 수행하는 시니어 ML 플랫폼 엔지니어이다.

현재 backend는 실제 `WorkerManager` OS process에서 다음까지 인증되었다.

- production graceful stop
- validated interrupt Checkpoint v2
- KNet/Split parent→child exact-resume bitwise parity
- checkpoint-write failure → `FAILED`, exit 50
- child restore/training failure → `FAILED`, exit 40
- coordinator restart 중 동일 child 복구
- Stop/Resume idempotency
- parent immutability와 child lineage

이번 작업은 이 backend를 다시 구현하는 것이 아니다. 목표는 다음뿐이다.

```text
explicit local-only write mode
→ POST Stop/Resume API
→ action resource 조회
→ Dash Stop safely / Resume training
→ 실제 Playwright browser workflow
```

최종 목표 판정:

```text
READY_FOR_CONFIG_GUI_LAUNCH_TRANCHE
```

---

## 0. Worktree 정책

### User working tree — 보존 대상

```text
/home/dss-pc-05/bench
```

여기에서는 source 수정, test 실행, artifact 생성을 하지 마라.

### Feature worktree — 모든 작업 기준

```text
/tmp/bench-wc-tranche
branch: benchmark-viz/write-control
```

모든 inspection, 구현, test, report 작성은 여기서 수행한다.

보고서 상단에 반드시 기록:

```yaml
git_common_repository: /home/dss-pc-05/bench/.git
user_working_tree: /home/dss-pc-05/bench
feature_worktree: /tmp/bench-wc-tranche
branch: benchmark-viz/write-control
baseline_commit: <exact sha>
head_commit: <exact sha>
control_root: <temporary root>
verification_layer: api_ui_browser
```

금지:

- `git add -A`
- `git commit -am`
- `git clean`
- `git reset --hard`
- unrelated checkout/revert
- verification artifact commit
- production `runs/`, `reports/`, registry 사용
- third-party tracked source 수정
- remote push

---

## 1. Preflight와 baseline

`/tmp/bench-wc-tranche`에서 실행하고 기록하라.

```bash
pwd
git rev-parse --show-toplevel
git rev-parse --git-common-dir
git branch --show-current
git rev-parse HEAD
git status --short
git log --oneline --decorate -25
git worktree list
git submodule status --recursive
```

다음 의미의 구현이 ancestry에 있는지 확인하라.

- production graceful-stop wiring
- WorkerManager KNet/Split bitwise parity
- real-process fault/restart hardening
- F-3 fail-closed `SuiteExecutor` handling
- F-4 resumable-path budget accounting
- Checkpoint v2
- registry migration 3
- durable Stop/Resume action
- child allocation/lineage
- `control_resumable_v1`

최근 관련 commit은 `ba65524`, `7296294`, `cf09078`, `a869b28`, `2133709` 등을 포함할 수 있으나 Git에서 실제 ancestry를 확인하라.

변경 전:

```bash
python -m pytest --collect-only -q
python -m pytest -q
```

최근 보고 기준은 약 `547 collected / 546 passed / 1 skipped`이나 실제 branch tip을 기준으로 한다.

구현을 찾지 못하면:

```text
WRONG_CHECKOUT_OR_LOST_CERTIFIED_BACKEND
```

으로 종료하라.

---

## 2. 반드시 읽을 최신 문서

```text
docs/benchmark_visualization/
  write_control_real_process_fault_contract.md
  write_control_fault_restart_report.md
  write_control_fault_restart_summary.json
  production_worker_graceful_stop_contract.md
  worker_level_exact_resume_certification.md
  write_control_worker_wiring_report.md
  write_control_worker_wiring_summary.json
  control_plane_training_path_contract.md
  resume_child_worker_contract.md
  durable_resume_action_contract.md
  checkpoint_v2_schema.md
  exact_resume_certification_matrix.md
  known_limitations.md
  operator_quickstart.md
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
  21_write_api_dash_controls_playwright_prompt.md
```

최신 코드와 실행 결과가 source of truth이다. 과거 report는 수정하지 마라.

---

## 3. Backend 재작성 금지

다음을 다시 설계하거나 별도 구현으로 대체하지 마라.

- Stop/Resume persistence
- StopCoordinator / graceful settlement
- Checkpoint v1/v2 및 validation/reconciliation
- exact-resume certification
- `control_resumable_v1`
- WorkerManager / worker lifecycle
- child allocation / lineage
- action idempotency / reconciliation
- normal/failure/restart semantics
- KNet/Split parity
- 기존 read-only FastAPI / Dash

API handler나 Dash callback 안에서 training, checkpoint restore, `Popen`, direct SQLite mutation을 하지 마라. 기존 service/manager를 호출하는 얇은 계층으로 유지하라.

---

# A. Explicit Write Mode

## 4. 기본은 계속 read-only

```text
BENCH_CONTROL_ENABLE_WRITES unset/0/false
→ POST write routes 없음
→ OpenAPI에 POST 없음
→ FastAPI GET/HEAD only
→ Dash action button 0개
```

Write mode:

```bash
export BENCH_CONTROL_ENABLE_WRITES=1
```

Boolean parsing은 엄격하게 구현한다.

## 5. Write mode는 loopback-only

허용:

```text
127.0.0.1
localhost(loopback)
::1
```

거부:

```text
0.0.0.0
::
LAN/public interface
BENCH_CONTROL_ALLOW_PUBLIC_BIND=1 + writes enabled
```

Startup error:

```text
Write control requires a loopback bind because authentication is not implemented.
```

CORS wildcard를 사용하지 마라. Write POST는 JSON과 다음 custom header를 요구한다.

```text
X-Bench-Control-Request: 1
```

브라우저 JS가 API를 직접 호출하지 않고 Dash server callback이 API client를 통해 호출하도록 유지한다.

---

# B. Write API

## 6. 최소 endpoint

Write mode에서만 등록:

```text
POST /api/v1/runs/{run_id}/actions/stop
POST /api/v1/checkpoints/{checkpoint_id}/actions/resume
GET  /api/v1/actions/{action_id}
```

## 7. Stop request

Headers:

```text
Idempotency-Key: <required>
X-Bench-Control-Request: 1
Content-Type: application/json
```

Body:

```json
{"expected_state_version": 12}
```

조건:

- run exists
- state `RUNNING`
- worker identity exists and alive
- `training_path_id == control_resumable_v1`
- graceful-stop eligible
- no conflicting active action
- state version matches

## 8. Resume request

Headers 동일.

Body:

```json
{"expected_parent_state_version": 19}
```

조건:

- checkpoint exists
- schema v2
- `VALID`
- `COMPATIBLE`
- launch-eligible
- parent terminal and allowed
- parent state version matches
- full certification tuple matches
- `control_resumable_v1`
- no conflicting active Resume action

모델명만 보고 Resume를 허용하지 마라.

## 9. 응답

성공 또는 동일 idempotent action 재사용:

```http
202 Accepted
```

예:

```json
{
  "schema_version": 1,
  "action_id": "...",
  "action_type": "STOP_GRACEFUL",
  "state": "REQUESTED",
  "run_id": "...",
  "checkpoint_id": null,
  "child_run_id": null,
  "idempotency_reused": false,
  "status_url": "/api/v1/actions/<id>"
}
```

동일 요청 재시도는 기존 action을 반환하고 `idempotency_reused=true`.

`GET /actions/{id}`는 최소한 다음을 반환한다.

- action type/state/version
- source run/checkpoint
- child run ID
- timestamps
- error code/message
- terminal 여부
- result checkpoint ID
- idempotency key는 원문 노출 금지

다음 의미를 유지하라.

```text
RESUME_EXACT action COMPLETED = child launch 완료
child run COMPLETED = resumed training 완료
```

## 10. Error mapping

| 상황 | HTTP |
|---|---:|
| malformed body/header | 400 |
| missing resource | 404 |
| stale state/version | 409 |
| invalid current state | 409 |
| same key + different request | 409 |
| corrupt checkpoint | 409 |
| uncertified envelope | 422 |
| unsupported training path/model | 422 |
| manager unavailable | 503 |
| unexpected error | 500 |

Machine-readable reason code를 제공한다.

API handler는 request validation → existing action service 호출 → durable action 반환만 수행한다.

---

# C. Capabilities and Eligibility

## 11. Capability fields

다음을 명시한다.

```text
write_control_available
write_control_enabled
graceful_stop_api
exact_resume_api
dash_stop_control
dash_resume_control
authentication
write_mode_loopback_only
```

기본 mode에서는 API/UI capability false, enabled loopback mode에서만 true.

Coarse model-name resume flag를 true로 만들지 말고 keyed certification rows를 사용한다.

## 12. Eligibility read model

Dash가 조건을 재구현하지 않도록 API가 eligibility와 reason을 반환한다.

```json
{
  "stop_action": {"eligible": true, "reason_code": null, "reason": null},
  "resume_action": {
    "eligible": false,
    "reason_code": "DEVICE_NOT_CERTIFIED",
    "reason": "This run used CUDA; certified exact resume requires CPU/fp32/0 workers."
  }
}
```

Eligibility 계산은 backend service와 단일 source를 공유한다.

---

# D. Dash Controls

## 13. Run Detail controls

Write mode에서만:

```text
Stop safely
Resume training
```

Dash는 SQLite를 직접 열지 않고 `ApiClient`만 사용한다.

## 14. Stop safely UX

설명:

```text
현재 optimizer update를 완료한 뒤 검증된 interrupt checkpoint를 저장하고 종료합니다.
즉시 종료되지 않을 수 있습니다.
```

Confirmation에 run/model, graceful stop이지 kill이 아님, checkpoint 실패 시 `FAILED`가 됨을 표시한다.

클릭 후:

- button 즉시 disable
- stable idempotency key 재사용
- action status panel
- polling으로 `STOP_REQUESTED`, `CHECKPOINTING`, `INTERRUPTED`, `FAILED`
- checkpoint link
- double-click/rerender가 새 key를 만들지 않음

## 15. Resume training UX

선택된 checkpoint에 대해 동작한다.

설명:

```text
검증된 checkpoint에서 새 child run을 생성해 학습을 이어갑니다.
기존 parent run과 checkpoint는 변경되지 않습니다.
```

Confirmation:

- parent run ID
- checkpoint ID/kind/cursor
- model/implementation
- certified envelope
- new child run 생성
- warm start가 아닌 exact resume

클릭 후:

- button disable
- stable idempotency key
- action 상태
- `child_run_id` link
- action completion과 child completion 분리
- child `RUNNING`/`COMPLETED`/`FAILED`
- lineage navigation

## 16. Ineligible reason

버튼을 숨기기만 하지 말고 구체적인 이유를 표시한다.

예:

```text
Exact resume unavailable: Checkpoint v1 is not write-control launch eligible.
```

```text
Safe stop unavailable: This run used legacy_train_v1.
```

이번 tranche에서 Force terminate, Warm start, Evaluate, Delete, Clone, Launch 버튼을 추가하지 마라.

---

# E. Fault/Restart UX

## 17. 기존 backend semantics를 정확히 표시

Checkpoint write failure:

```text
Stop action FAILED
Run FAILED
exit 50
No resumable checkpoint
```

Child restore failure:

```text
Resume launch action COMPLETED
Child FAILED
0 updates
```

Child training failure:

```text
Resume launch action COMPLETED
Child FAILED
updates > resume cursor
```

API/coordinator restart 후 durable state를 다시 불러온다. frontend memory는 authority가 아니다.

API unreachable 시 같은 idempotency key로 outcome을 다시 조회하고 새 key를 만들지 마라.

---

# F. Tests

## 18. Read-only mode regression

Writes disabled:

- OpenAPI POST 없음
- POST Stop/Resume 404/405
- capability false
- Dash buttons 0
- 기존 read-only Playwright smoke PASS

## 19. Write-enabled API tests

필수:

- loopback + writes enabled routes present
- non-loopback + writes enabled startup refused
- missing custom header rejected
- missing idempotency key rejected
- malformed/stale version rejected
- same key/same request ×5 → same action
- same key/different body/target → 409
- GPU/AMP/multi-worker/legacy path → 422
- corrupt checkpoint rejected before child allocation
- missing resource → 404
- manager unavailable → 503
- action GET round-trip
- API restart 후 accepted action 복구

## 20. Actual API-to-worker E2E

Mock service로 대체하지 마라.

### KNet

```text
real parent worker
→ POST Stop
→ 202/action
→ INTERRUPTED + Checkpoint v2
→ POST Resume
→ 202/action
→ child run
→ child COMPLETED
```

기존 bitwise parity를 회귀 확인한다.

### Split

실제 POST Stop/Resume가 `bench_split_adapter_v1`을 통과해 child `COMPLETED`가 되는 것을 확인하고 기존 parity를 유지한다.

## 21. Playwright real-browser workflow

실제 FastAPI/Dash/browser DOM을 사용한다.

Mandatory KNet workflow:

```text
Run Detail
→ Stop safely
→ confirmation
→ action status
→ STOP_REQUESTED/CHECKPOINTING
→ INTERRUPTED
→ Checkpoint v2
→ Resume training
→ confirmation
→ child link
→ child RUNNING
→ child COMPLETED
```

검증:

- double-click에도 1 action
- refresh 후 same action
- API restart 후 action/child 복구
- parent/child navigation
- action completion과 child completion 분리

Unsupported DOM case도 최소 1개 확인한다: CUDA, legacy path, Checkpoint v1, corrupt checkpoint 중 하나.

Failure UI도 최소 1개 확인한다: checkpoint-write 또는 child restore failure.

## 22. Layering audit

- Dash callback은 API client만 호출
- Dash가 registry/adapter/trainer import하지 않음
- API router가 Dash/Streamlit import하지 않음
- API handler가 worker를 wait하지 않음
- UI 종료 후 worker 지속
- API restart 후 action 복구

---

# G. Mandatory Regression

제외 옵션 없이:

```bash
python -m pytest --collect-only -q
python -m pytest -q
```

별도 기록:

- disabled/enabled OpenAPI audit
- API idempotency/concurrency
- KNet/Split API-to-worker E2E
- Playwright write workflow
- real-process fault/restart regression
- KNet/Split WorkerManager parity
- Checkpoint v1/v2
- observer/telemetry parity
- 28 init-provenance
- Streamlit Inspector import/load
- third-party tracked diff

신규 output은 `tmp_path` 또는 임시 `BENCH_CONTROL_ROOT`만 사용한다.

---

# H. Explicit Exclusions

구현하지 마라.

- Config GUI/editor
- GUI benchmark launch
- Force terminate
- Warm-start launch
- Evaluate checkpoint action
- GPU queue/scheduler
- shared GPU
- GPU/AMP/multi-worker exact resume
- Adaptive/MAML/ME-Split exact resume
- authentication/authorization
- remote worker/multi-user
- WebSocket/SSE
- arbitrary batch-midpoint resume

Polling과 기존 event cursor를 사용한다.

---

# I. Documents and Artifacts

새 문서:

```text
docs/benchmark_visualization/
  write_control_api_contract.md
  write_control_ui_contract.md
  write_control_operator_guide.md
  write_api_ui_tranche_report.md
  write_api_ui_tranche_summary.json
```

갱신:

```text
operator_quickstart.md
known_limitations.md
implementation_status_phase0_phase1_phase3.md
production_worker_graceful_stop_contract.md
worker_level_exact_resume_certification.md
exact_resume_certification_matrix.md
resume_child_worker_contract.md
durable_resume_action_contract.md
```

과거 tranche report는 수정하지 마라.

Evidence:

```text
artifacts/benchmark_write_api_ui/<timestamp>/
```

보존:

- Git snapshot
- disabled/enabled OpenAPI
- API requests/responses
- action/run/worker/checkpoint rows
- process identity
- server logs
- Playwright screenshots/console/network
- pytest logs
- KNet/Split hashes
- restart/idempotency evidence

commit하지 마라.

---

# J. Commit Policy

권장:

```text
1. feat: local-only write mode + POST action API
2. feat: Dash Stop/Resume controls
3. test: API/worker/Playwright/security/idempotency
4. docs: contracts/operator/report/summary
```

명시적 path만 stage한다. Push하지 마라.

---

# K. Final Verdict

## `READY_FOR_CONFIG_GUI_LAUNCH_TRANCHE`

다음을 모두 만족한 경우에만:

- default read-only preserved
- loopback-only write mode PASS
- POST Stop/Resume PASS
- GET action PASS
- HTTP idempotency/concurrency PASS
- KNet API-to-worker PASS
- Split API-to-worker PASS
- Dash controls PASS
- ineligible reason UX PASS
- Playwright full workflow PASS
- API restart recovery PASS
- backend fault semantics correctly rendered
- full pytest PASS
- 28 init-provenance PASS
- third-party diff empty

그 외:

- `READY_AFTER_SPECIFIC_FIXES`
- `NOT_READY`
- `WRONG_CHECKOUT_OR_LOST_CERTIFIED_BACKEND`

---

# L. Completion Output

1. Git common repository
2. user working tree untouched 여부
3. feature worktree
4. branch / baseline / new HEAD
5. new commits
6. production files changed
7. read-only default result
8. write-mode security result
9. Stop POST result
10. Resume POST result
11. HTTP idempotency/restart
12. KNet API-to-worker E2E
13. Split API-to-worker E2E
14. Dash controls
15. Playwright workflow
16. full pytest
17. 28 init-provenance
18. OpenAPI methods disabled/enabled
19. third-party diff
20. final verdict
21. report/summary/artifact paths

이제 이 prompt 전체를 `/tmp/bench-wc-tranche`에서 실행하라.
