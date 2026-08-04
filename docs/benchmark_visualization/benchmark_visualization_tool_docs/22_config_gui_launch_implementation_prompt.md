# Config GUI and Benchmark Launch Implementation Prompt

당신은 AI-ADCS Neural Kalman Filter Benchmark 저장소의
Benchmark Execution Visualization **Config GUI and Launch Integration** tranche를 수행하는
시니어 ML 플랫폼 엔지니어이다.

현재 시스템은 다음 수준까지 실제 process와 browser에서 검증되었다.

- immutable run identity와 typed `ResolvedRunSpec`
- SQLite registry와 JSONL event journal
- detached `WorkerManager`
- live metric/log/resource dashboard
- Checkpoint v2
- KNet/Split production graceful stop
- KNet/Split worker-level exact resume
- real-process failure/restart hardening
- explicit local-only write mode
- POST Stop/Resume API
- Dash Stop safely / Resume training
- Playwright Stop→Checkpoint→Resume→Child workflow

이번 tranche의 목표는 사용자가 **기존 benchmark preset을 GUI에서 선택하고,
안전한 범위에서 편집·검증·resolve한 뒤 새 immutable run을 시작하는 workflow**를
추가하는 것이다.

```text
Preset 선택
→ Clone
→ Form / Raw YAML 편집
→ typed validation
→ ResolvedRunSpec preview
→ structural/operational diff
→ launch confirmation
→ durable launch action
→ WorkerManager run
→ Run Detail
```

이번 작업은 config system, runner, worker, Stop/Resume를 다시 만드는 작업이 아니다.
기존 typed config resolver와 durable lifecycle을 GUI/API에 연결하는 작업이다.

최종 목표 판정:

```text
READY_FOR_GPU_QUEUE_AND_EXECUTION_POLICY_TRANCHE
```

---

## 0. Instruction provenance gate

이번 prompt는 구현 시작 전에 repository에 tracked 상태로 존재해야 한다.

Expected path:

```text
docs/benchmark_visualization/benchmark_visualization_tool_docs/
  22_config_gui_launch_implementation_prompt.md
```

`/tmp/bench-wc-tranche`에서 작업을 시작하기 전에 반드시 다음을 실행한다.

```bash
git ls-files --error-unmatch \
  docs/benchmark_visualization/benchmark_visualization_tool_docs/22_config_gui_launch_implementation_prompt.md

sha256sum \
  docs/benchmark_visualization/benchmark_visualization_tool_docs/22_config_gui_launch_implementation_prompt.md

git status --short -- \
  docs/benchmark_visualization/benchmark_visualization_tool_docs/22_config_gui_launch_implementation_prompt.md
```

필수 조건:

- 파일이 Git index에 tracked 상태여야 한다.
- tranche 시작 시 파일이 modified/untracked 상태이면 안 된다.
- SHA-256을 report와 summary JSON에 기록한다.
- 외부 채팅 지시와 repository prompt가 다르면 repository의 tracked prompt를 기준으로 하되 차이를 보고한다.
- 파일이 없거나 tracked 상태가 아니면 구현을 시작하지 말고 다음으로 종료한다.

```text
INSTRUCTION_PROVENANCE_NOT_REPRODUCIBLE
```

---

## 1. Repository와 worktree 역할

### Common Git repository / user working tree

```text
/home/dss-pc-05/bench
```

- 사용자의 기존 연구 작업을 보존하는 working tree이다.
- source 수정, test 실행, artifact 생성 대상으로 사용하지 마라.
- 최신 visualization implementation checkout으로 간주하지 마라.

### Feature implementation worktree

```text
/tmp/bench-wc-tranche
```

- `benchmark-viz/write-control` branch의 최신 구현 기준이다.
- 모든 inspection, implementation, test, report 작성은 이 worktree에서 수행한다.
- worktree가 없으면 Git branch와 ancestry를 확인한 뒤 복구한다.
- `/home/dss-pc-05/bench`의 dirty 파일을 복사·이동·삭제하지 마라.

모든 새 보고서 상단에 다음을 기록하라.

```yaml
git_common_repository: /home/dss-pc-05/bench/.git
user_working_tree: /home/dss-pc-05/bench
feature_worktree: /tmp/bench-wc-tranche
branch: benchmark-viz/write-control
baseline_commit: <exact sha before this tranche>
head_commit: <exact sha after this tranche>
instruction_document:
  path: docs/benchmark_visualization/benchmark_visualization_tool_docs/22_config_gui_launch_implementation_prompt.md
  tracked_at_start: true
  sha256: <sha256>
control_root: <temporary test root>
verification_layer: config_api_ui_browser
```

---

## 2. Preflight와 baseline

`/tmp/bench-wc-tranche`에서 다음을 실행하라.

```bash
pwd
git rev-parse --show-toplevel
git rev-parse --git-common-dir
git branch --show-current
git rev-parse HEAD
git status --short
git log --oneline --decorate -30
git worktree list
git submodule status --recursive
```

다음 구현이 ancestry에 있는지 확인하라.

- real-process fault/restart certification
- local-only write mode
- Stop/Resume POST API
- Dash Stop/Resume controls
- Playwright write workflow
- Checkpoint v2
- `control_resumable_v1`
- typed config resolver
- immutable run allocation
- WorkerManager and SuiteExecutor

최근 write API/UI 결과 기준 implementation head는 `ca05ece` 계열일 수 있으나,
정확한 branch tip과 docs commit을 Git에서 확인하라.
존재하지 않는 hash를 추측하지 마라.

변경 전에 다음을 실행한다.

```bash
python -m pytest --collect-only -q
python -m pytest -q
```

최근 보고 기준은 약:

```text
582 collected
581 passed
1 skipped
0 failed
```

이나 실제 branch tip 결과를 baseline으로 기록하라.

인증된 backend를 찾지 못하면:

```text
WRONG_CHECKOUT_OR_LOST_WRITE_CONTROL_BACKEND
```

으로 종료하라.

---

## 3. 반드시 읽을 최신 문서

### Write-control 결과

```text
docs/benchmark_visualization/
  write_control_api_contract.md
  write_control_ui_contract.md
  write_control_operator_guide.md
  write_api_ui_tranche_report.md
  write_api_ui_tranche_summary.json
  write_control_real_process_fault_contract.md
  write_control_fault_restart_report.md
  write_control_fault_restart_summary.json
  production_worker_graceful_stop_contract.md
  worker_level_exact_resume_certification.md
  checkpoint_v2_schema.md
  exact_resume_certification_matrix.md
```

### Config/architecture 기준

```text
docs/benchmark_visualization/benchmark_visualization_tool_docs/
  02_target_architecture_and_mvp_plan.md
  03_backend_contracts_and_data_schemas.md
  04_ui_information_architecture_and_workflows.md
  05_do_not_do_risks_and_migration.md
  06_acceptance_tests_and_definition_of_done.md
  13_write_control_tranche_plan.md
  14_write_control_constraints_and_decisions.md
  15_write_control_acceptance_and_test_plan.md
  21_write_api_dash_controls_playwright_prompt.md
  22_config_gui_launch_implementation_prompt.md
```

최신 코드와 실행 결과가 source of truth이다.
과거 문서가 현재 상태와 다르면 현재 상태 문서에 correction을 기록하고 historical report는 덮어쓰지 마라.

---

## 4. 이미 완료된 기능 — 재작성 금지

다음을 새로 만들거나 우회하지 마라.

- typed config dataclasses
- suite-to-draft compatibility layer
- `ResolvedRunSpec`
- structural/operational config hash
- `variant_id`
- immutable run allocation
- SQLite registry
- JSONL events
- WorkerManager
- SuiteExecutor
- worker CLI
- Stop/Resume actions
- write mode security
- Checkpoint v2
- exact-resume eligibility
- Dash Run Detail controls
- 기존 Run Inspector integration

GUI와 API는 기존 resolver와 service를 사용해야 한다.
Form 전용 schema와 CLI 전용 schema를 별도로 만들지 마라.

---

# Part A — MVP Launch Scope

## 5. Launch 지원 범위

### Certified trainable models

```text
kalmannet_tsp
split_knet
```

기본 launch envelope:

```text
device_class: cpu
precision: fp32
num_workers: 0
training_path_id: control_resumable_v1
```

### Model-based baselines

기존 registry가 지원하는 model-based KF aliases를 launch할 수 있다.
이들은:

```text
training_path_id = not_applicable
```

이며 Stop/Resume가 없어야 한다.

### Deferred adapters

다음은 preset 목록에 보일 수 있으나 launch를 활성화하지 않는다.

```text
adaptive_knet
maml_knet
me_split_knet_v0
그 외 GUI launch용 lifecycle이 인증되지 않은 adapter
```

API가 machine-readable 비지원 이유를 제공하고 UI가 표시한다.
모델 이름을 임의 hard-code하지 말고 기존 capability/implementation registry를 사용하라.

---

# Part B — Preset Catalog

## 6. Preset source of truth

Preset은 repository가 승인한 config root 아래의 **tracked YAML**만 대상으로 한다.

권장 root:

```text
bench/configs/
```

Preset catalog 최소 필드:

- stable `preset_id`
- display name
- tracked relative path
- content digest
- suite name/version
- task/model/plan summary
- launch support status
- unsupported reason
- schema compatibility result

`preset_id`를 raw filesystem path로 사용하지 마라.
absolute path를 request identity로 사용하지 마라.

## 7. Preset discovery security

금지:

- user-supplied arbitrary path
- `..` traversal
- absolute path
- symlink escape
- untracked config 자동 노출
- arbitrary repository file 읽기
- YAML custom tag construction

`yaml.safe_load` 또는 현재 안전 parser만 사용한다.
Content size, nesting depth, alias expansion에 합리적 제한을 둔다.

## 8. Preset API

최소 read endpoint:

```text
GET /api/v1/config/presets
GET /api/v1/config/presets/{preset_id}
GET /api/v1/config/schema
```

Read-only mode에서도 preset 조회와 validation preview는 허용할 수 있다.
Launch endpoint는 write mode에서만 존재한다.

---

# Part C — Config Editing Contract

## 9. 단일 typed source of truth

```text
Preset YAML
→ safe parse
→ typed RunSpecDraft
→ semantic validation
→ ResolvedRunSpec
→ canonical serialization/hashes
```

Form과 Raw YAML은 동일한 draft를 편집하는 두 view이다.
Form/Raw YAML/CLI용 resolver를 각각 만들지 마라.

## 10. Schema descriptor

기존 dataclass config에서 GUI용 machine-readable descriptor를 제공한다.

최소 metadata:

- field path
- type
- label/help
- default/required
- enum/min/max
- read-only 여부
- structural/operational 분류
- visibility condition
- capability dependency
- path/sensitive field 여부

가능하면 versioned JSON Schema 또는 동등한 descriptor를 제공한다.

## 11. MVP form 영역

### Experiment

- preset-derived experiment name
- user run label/description
- seed
- task
- model
- init/training plan

### Training

- max optimizer updates
- validation interval
- early-stop patience
- batch size
- supported learning rate
- deterministic mode

### Runtime

- device
- precision
- `num_workers`
- telemetry interval
- log level

### Output/observation

- visualization artifact emission
- supported logging/artifact flags

Resolver가 해석하지 못하는 key는 `unsupported_fields`에 표시하고 raw YAML에서 보존하되,
GUI가 지원한다고 광고하지 마라.

## 12. Conditional fields

- model-based baseline → training fields 비활성
- KNet/Split trained plan → training fields 활성
- pretrained/loaded init → checkpoint/reference field 필요
- device에 따라 precision 제한
- task family에 따라 scenario fields 변경
- 미지원 조합은 validation 단계에서 이유와 함께 거부

## 13. Form/YAML synchronization

```text
Preset load
→ Form edit
→ Raw YAML view
→ YAML edit
→ Parse/validate
→ Form refresh
```

요구사항:

- parse error는 line/column 표시
- unknown key는 field-level warning/error
- unsupported preserved key 별도 표시
- Form 전환 시 key가 조용히 삭제되지 않음
- original preset file은 절대 수정하지 않음

---

# Part D — Validation and Preview API

## 14. Validation endpoint

```text
POST /api/v1/config/validate
```

Read-only computation이므로 write mode 없이도 허용할 수 있다.
Request는 raw YAML content 또는 typed draft를 받으며 filesystem path는 받지 않는다.

Response 최소 필드:

```json
{
  "schema_version": 1,
  "valid": true,
  "issues": [],
  "unsupported_fields": [],
  "resolved_run_spec": {},
  "canonical_yaml": "...",
  "structural_config_hash": "...",
  "operational_config_hash": "...",
  "variant_id": "...",
  "training_path_id": "control_resumable_v1",
  "launch_eligibility": {
    "eligible": true,
    "reason_code": null,
    "reason": null
  }
}
```

## 15. Validation requirements

반드시 검증한다.

- unknown keys
- types/enums/ranges
- cross-field conditions
- model/init compatibility
- trainability
- device/precision availability
- `num_workers`
- budget/interval consistency
- checkpoint/reference trust root
- output root containment
- absolute/parent/symlink escape
- dynamic repo/class path policy
- uncertified launch adapter
- training-path decision
- dataset/scenario identity

Validation 오류 시 run allocation이나 DB mutation이 없어야 한다.

## 16. Preview

UI는 launch 전에 다음을 보여준다.

- original preset summary
- edited draft
- resolved config
- changed fields
- structural/operational diff
- unsupported preserved fields
- hashes
- `variant_id`
- `implementation_id`
- `training_path_id`
- capability summary
- command-equivalent summary

Run ID와 final run directory는 launch allocation 전 미리 고정하지 마라.

---

# Part E — Durable Launch API

## 17. Launch endpoint

Write mode에서만:

```text
POST /api/v1/runs/launch
```

Required headers:

```text
Idempotency-Key: <opaque stable key>
X-Bench-Control-Request: 1
Content-Type: application/json
```

Request 최소 필드:

```json
{
  "preset_id": "...",
  "preset_digest": "...",
  "draft": {},
  "expected_structural_config_hash": "...",
  "expected_operational_config_hash": "..."
}
```

## 18. Durable launch semantics

API handler는 직접 `Popen`하거나 worker를 wait하지 않는다.

```text
request validation
→ preset digest 확인
→ draft resolve 재실행
→ preview hash 일치 확인
→ durable LAUNCH_RUN action
→ immutable run allocation
→ original/resolved config snapshot
→ WorkerManager launch
→ 202 action/run resource
```

가능하면 기존 action infrastructure를 재사용한다.
새 DB나 별도 launch queue를 만들지 마라.

```text
LAUNCH_RUN action COMPLETED = 정확히 하나의 run/worker launch 완료
run COMPLETED = benchmark workload 완료
```

## 19. Idempotency와 concurrency

동일 key + 동일 request:

```text
action 1개
run 1개
worker 1개
```

동일 key + 다른 request:

```text
409 Conflict
side effects 0
```

동일 config + 다른 key:

```text
서로 다른 run_id
서로 다른 immutable directory
```

API crash/restart 시 action/allocation/launch 각 경계에서 duplicate run/worker가 없어야 한다.

## 20. Preset drift

Preview 뒤 preset digest가 바뀌면:

```text
409 Conflict
revalidate required
```

Preview hash와 launch-time resolve hash가 다르면 거부한다.

## 21. Launch response

권장:

```http
202 Accepted
```

```json
{
  "schema_version": 1,
  "action_id": "...",
  "action_type": "LAUNCH_RUN",
  "state": "REQUESTED",
  "run_id": "...",
  "status_url": "/api/v1/actions/...",
  "run_url": "/api/v1/runs/...",
  "idempotency_reused": false
}
```

---

# Part F — Dash New Run Workflow

## 22. New Run page

새 route:

```text
/new-run
```

Read-only mode에서는 preset 탐색·validation preview만 허용할 수 있다.
Launch는 없거나 disabled여야 하며 write mode 필요를 표시한다.

## 23. Wizard

```text
1. Choose Preset
2. Configure
3. Validate
4. Review
5. Launch
```

### Preset

- 검색/filter
- task/model/support 상태
- preset digest/version
- unsupported 이유

### Configure

- schema-driven form
- Raw YAML toggle
- inline help
- conditional fields
- unsupported fields panel

### Validate

- field-level issue summary
- YAML line/column error
- launch eligibility

### Review

- diff
- resolved config
- hashes
- variant/implementation/training path
- launch confirmation

### Launch

- stable idempotency key
- action status
- allocated run ID
- Run Detail link
- action과 run state 분리

## 24. Launch confirmation

표시:

- model/task/preset
- device/precision/workers
- training budget
- `training_path_id`
- 새 immutable run 생성
- preset 원본 불변
- Stop/Resume eligibility 예상
- GPU scheduling이 아직 없다는 경고

## 25. UI layering

- Dash callback은 `ApiClient`만 사용
- registry/resolver/adapter/WorkerManager 직접 import 금지
- browser가 FastAPI에 직접 POST하지 않음
- write header/idempotency key는 Dash server-side에서 설정
- UI eligibility는 API response만 사용

---

# Part G — Output and Provenance

## 26. GUI-launched run artifacts

최소:

```text
original_preset.yaml
submitted_draft.yaml 또는 json
resolved_run_spec.json
config_validation.json
launch_request.json
```

idempotency key 원문은 저장하지 마라.

## 27. Run provenance

- launch source: `gui`, `cli`, `resume`
- preset ID/digest
- config schema version
- submitted/resolved hashes
- Git/submodule revisions
- user label
- training path
- implementation identity
- unsupported preserved fields

CLI launch의 기존 semantics를 바꾸지 마라.

---

# Part H — Testing

## 28. Config tests

- preset stable ID/digest
- untracked preset 제외
- traversal/absolute/symlink escape
- YAML custom tag 거부
- oversized/deep/alias-heavy YAML 제한
- form descriptor correctness
- conditional fields
- unknown/unsupported key behavior
- form→YAML→form round-trip
- original preset 불변
- structural vs operational hash
- training-path decision
- model/init/device validation
- output containment

## 29. Validation API tests

- valid KNet/Split
- model-based baseline
- unsupported Adaptive/MAML/ME-Split
- malformed YAML
- unknown key
- unsafe path
- invalid enum/range
- cross-field conflict
- unavailable device
- preview hash stability
- invalid config side effect 0

## 30. Launch API tests

- write-disabled → route absent
- loopback write-enabled → route present
- missing header/key → 400
- invalid config → side effect 0
- stale preset digest → 409
- preview hash mismatch → 409
- same key same request ×5 → action/run/worker 각 1
- same key different draft → 409
- same config different keys → unique runs
- API restart at action/allocation/launch boundaries
- manager unavailable → 503
- original/resolved snapshots 존재

## 31. CLI parity

동일 resolved config에 대해 GUI/API launch와 기존 CLI/control launch의 다음을 비교한다.

- `ResolvedRunSpec`
- structural/operational hashes
- variant ID
- training path
- dataset identity
- update budget
- deterministic tiny KNet/Split final model/metrics

GUI가 다른 parser/default를 사용하면 실패다.

## 32. Actual launch E2E

### KNet

```text
GUI/API preset
→ tiny budget edit
→ validate
→ launch
→ RUNNING
→ Run Detail
→ Stop safely
→ INTERRUPTED
→ Resume training
→ child COMPLETED
```

기존 bitwise parity를 유지한다.

### Split

```text
preset validate
→ GUI/API launch
→ control_resumable_v1
→ WorkerManager COMPLETED 또는 Stop/Resume
```

`implementation_id == bench_split_adapter_v1` guard 유지.

### Model-based baseline

```text
preset launch
→ train phase skipped
→ COMPLETED
→ Stop/Resume 없음
```

## 33. Playwright workflow

```text
/new-run
→ preset 선택
→ form edit
→ raw YAML 전환/복귀
→ validation
→ resolved preview/diff
→ Launch
→ action status
→ Run Detail
→ live state/metric
→ Stop safely
→ INTERRUPTED
→ Resume training
→ child COMPLETED
```

검증:

- double-click launch → run 1개
- refresh 후 action/run link 복구
- invalid config에서 Launch 불가
- read-only mode에서 Launch 불가
- unsupported adapter reason 표시
- no console errors
- original preset digest unchanged

## 34. Regression

제외 옵션 없이:

```bash
python -m pytest --collect-only -q
python -m pytest -q
```

별도 기록:

- Stop/Resume API tests
- write-control Playwright
- real-process fault/restart
- KNet/Split worker parity
- checkpoint tests
- observer/telemetry parity
- 28 init-provenance regression
- Streamlit Inspector import/load
- third-party tracked diff
- disabled/enabled OpenAPI audit

기존 테스트 삭제/skip/xfail/ignore 금지.
모든 output은 `tmp_path` 또는 임시 `BENCH_CONTROL_ROOT`를 사용한다.

---

# Part I — Security and Failure Handling

## 35. Security

- launch endpoint는 write mode + loopback-only
- custom write header와 idempotency key 필수
- wildcard CORS 금지
- raw YAML은 data이지 path가 아님
- arbitrary command/class/repository path 입력 금지
- shell invocation 금지
- body size 제한
- error response에 stack/absolute path 노출 금지
- API handler에서 untrusted pickle load 금지
- preset catalog는 tracked allowlist만 사용

## 36. Failure semantics

### Validation failure

```text
HTTP 400/422
run/action allocation 없음
```

### Launch allocation failure

```text
LAUNCH_RUN action FAILED
run 없음 또는 명시적 terminal state
```

### Worker launch failure

```text
action FAILED
allocated run CANCELLED 또는 contract state
```

### Worker training failure

```text
launch action COMPLETED
run FAILED
```

Action completion과 workload completion을 합치지 마라.

---

# Part J — Explicit Exclusions

이번 tranche에서는 구현하지 마라.

- sweep/grid search GUI
- multiple-run batch launch
- GPU queue/scheduler/lease enforcement
- shared GPU execution
- Force terminate
- Warm-start launch
- Evaluate checkpoint
- persistent draft library
- arbitrary filesystem config upload
- authentication/authorization
- multi-user
- remote worker
- GPU/AMP/multi-worker exact resume
- Adaptive/MAML/ME-Split GUI launch
- WebSocket/SSE

---

# Part K — Documents and Artifacts

## 37. 새 문서

```text
docs/benchmark_visualization/
  config_gui_contract.md
  launch_api_contract.md
  config_gui_operator_guide.md
  config_gui_launch_tranche_report.md
  config_gui_launch_tranche_summary.json
```

## 38. 갱신 문서

```text
operator_quickstart.md
known_limitations.md
implementation_status_phase0_phase1_phase3.md
write_control_operator_guide.md
write_control_ui_contract.md
write_control_api_contract.md
```

과거 tranche report는 수정하지 마라.

## 39. Evidence

```text
artifacts/benchmark_config_gui_launch/<timestamp>/
```

보존:

- instruction hash
- preflight Git snapshot
- preset catalog/schema snapshot
- validation/launch API requests/responses
- action/run/worker rows
- original/resolved config snapshots
- CLI parity hashes
- Playwright screenshots/network/console logs
- pytest logs

commit하지 마라.

---

# Part L — Commit Policy

권장:

```text
1. feat: preset catalog + schema/validation/launch API
2. feat: Dash New Run wizard and launch UX
3. test: config/API/CLI parity/Playwright/E2E/security
4. docs: contracts/operator/report/summary
```

명시적 path만 stage하라.

금지:

- `git add -A`
- `git commit -am`
- `git clean`
- `git reset --hard`
- verification artifact commit
- temp DB/run/log/screenshot commit
- user working tree unrelated changes commit
- third-party generated files commit
- remote push

---

# Part M — Final Verdict

## `READY_FOR_GPU_QUEUE_AND_EXECUTION_POLICY_TRANCHE`

다음을 모두 만족한 경우에만 사용한다.

```text
instruction provenance tracked/hash recorded
preset catalog safe and reproducible
single typed config source of truth
form/YAML round-trip PASS
validation and preview PASS
unsafe config/path rejection PASS
durable launch API PASS
launch idempotency/restart PASS
same config different request → unique immutable runs
CLI/GUI resolved-config parity PASS
KNet GUI launch E2E PASS
Split GUI launch E2E PASS
model-based baseline launch PASS
existing Stop/Resume workflow PASS
Playwright New Run→Launch→Stop→Resume PASS
full pytest PASS
28 init-provenance PASS
third-party tracked diff empty
```

그 외:

- `READY_AFTER_SPECIFIC_FIXES`
- `NOT_READY`
- `WRONG_CHECKOUT_OR_LOST_WRITE_CONTROL_BACKEND`
- `INSTRUCTION_PROVENANCE_NOT_REPRODUCIBLE`

---

# Part N — Completion Output

완료 시 터미널에는 다음만 구조적으로 출력하라.

1. Git common repository
2. user working tree untouched 여부
3. feature worktree
4. branch / baseline / new HEAD
5. instruction path / tracked status / SHA-256
6. new commits
7. production files changed
8. preset catalog result
9. schema/form/YAML round-trip result
10. validation/security result
11. launch API/idempotency/restart result
12. CLI parity result
13. KNet GUI launch E2E
14. Split GUI launch E2E
15. model-based baseline result
16. Playwright workflow
17. existing Stop/Resume regression
18. full pytest
19. 28 init-provenance
20. third-party tracked diff
21. final verdict
22. reports/summary/artifact paths

이제 이 prompt 전체를 `/tmp/bench-wc-tranche`에서 실행하라.
