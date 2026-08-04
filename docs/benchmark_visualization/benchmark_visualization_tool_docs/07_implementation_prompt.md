# CLI Coding Agent Prompt — Benchmark Execution Visualization Foundation

아래 프롬프트를 저장소 루트에서 실행되는 CLI coding agent에 전달한다.

---

당신은 AI-ADCS / Neural Kalman Filter Benchmark 저장소의 시니어 플랫폼 엔지니어이다.

이번 작업의 목표는 화면을 급히 만드는 것이 아니라, 현재 benchmark를 안전하게 실행·관측할 수 있는 **frontend-agnostic backend foundation과 read-only Dash dashboard**를 구현하는 것이다.

## 0. 기준 정보

- 저장소 루트 예상: `/home/sung-lee/bench`
- 감사 기준 commit: `d1cc4b035597bb029e0bce95a546b29b4664b5c6`
- 기준 감사 문서: `docs/benchmark_gui_current_state_audit.md`
- 추가 설계 문서가 저장소에 없다면, 사용자로부터 제공된 다음 문서를 `docs/benchmark_visualization/` 아래에 먼저 복사하라.
  - `00_README.md`
  - `01_frontend_framework_evaluation.md`
  - `02_target_architecture_and_mvp_plan.md`
  - `03_backend_contracts_and_data_schemas.md`
  - `04_ui_information_architecture_and_workflows.md`
  - `05_do_not_do_risks_and_migration.md`
  - `06_acceptance_tests_and_definition_of_done.md`

## 1. 이번 구현 범위

이번 작업에서 반드시 구현할 범위:

1. baseline test/environment snapshot
2. canonical identity types
3. typed config/ResolvedRunSpec의 최소 실행 가능한 기반
4. immutable run allocation
5. SQLite WAL run registry와 migration
6. append-only JSONL event journal
7. run별 subprocess/process group worker
8. heartbeat, stdout/stderr capture, ordinary failure와 abrupt death 기록
9. CPU/RAM 및 가능한 경우 NVIDIA telemetry
10. adapter/runner observer hook의 최소 공통 contract
11. Dash + FastAPI read-only dashboard
12. 기존 Streamlit Run Inspector deep link 또는 mapping
13. unit/integration/browser smoke tests
14. 구현 상태 보고서

이번 작업에서 **활성화하지 말아야 할 기능**:

- exact resume
- Resume 버튼
- graceful Stop 버튼
- config GUI launch
- multi-GPU queue
- shared GPU execution
- multi-user/auth
- React frontend
- 모든 adapter instrumentation 완성

단, 이후 구현이 가능하도록 schema와 capability field는 정의하라. 기능이 없는데 버튼만 만드는 것을 금지한다.

## 2. 절대 원칙

- training을 Dash callback, FastAPI request handler, Streamlit session, UI thread에서 실행하지 마라.
- 각 run은 별도 subprocess 및 process group에서 실행하라.
- UI는 run state를 직접 변경하지 마라.
- filesystem path, model_id, variant_label을 run identity로 사용하지 마라.
- 기존 deterministic run directory를 덮어쓰지 마라.
- `model.pt` load를 resume라고 부르지 마라.
- stdout parsing을 metric source of truth로 사용하지 마라.
- third-party repository를 임의로 수정하지 마라.
- 기존 run, checkpoint, dataset을 삭제하거나 수정하지 마라.
- 현재 dirty worktree/submodule을 보존하고 보고하라.
- 구현한 것과 향후 계획을 혼동하지 마라.

## 3. 첫 단계: 저장소 기준선 확보

먼저 다음을 수행하고 `docs/benchmark_visualization/implementation_baseline.md`에 기록하라.

- Git branch, HEAD, status, dirty diff summary
- submodule revisions와 dirty 여부
- Python/PyTorch/CUDA/GPU
- dependency source
- pytest 설치 여부
- full test collection 수
- baseline pass/fail
- 기존 28개 variant identity regression 재실행
- representative run/artifact/checkpoint 위치

pytest가 dev extra에 선언되어 있으나 설치되지 않았다면 저장소 정책에 맞게 dev dependency를 설치하고 lockfile 변경 필요성을 보고하라. 무단으로 대규모 dependency upgrade를 하지 마라.

기존 실패는 baseline failure로 기록하고, 이번 변경으로 새 실패를 만들지 마라.

## 4. 구현 세부 요구사항

### 4.1 Identity

새 package 예시:

```text
bench/control/identity.py
```

최소 타입:

- ExperimentId
- RunId
- ModelId
- ImplementationId
- InitId
- VariantId
- CheckpointId
- ArtifactId

요구사항:

- persistent ID에 Python `hash()` 사용 금지
- canonical JSON serialization + SHA-256
- stable across process restart
- run_id는 UUIDv7/ULID 또는 표준 UUID 기반 immutable ID
- `variant_label`은 presentation-only helper

테스트:

- 같은 input stable
- 같은 model, 다른 init/implementation distinct
- process restart stable

### 4.2 Typed config와 ResolvedRunSpec

기존 suite YAML을 즉시 전면 교체하지 말고 compatibility layer를 만든다.

```text
bench/control/config/schema.py
bench/control/config/resolver.py
bench/control/config/compatibility.py
```

요구사항:

- 기존 YAML dict → typed object
- unknown-key policy 명시
- field-level validation error
- canonical JSON/YAML serialization
- structural hash와 operational hash
- original config attachment 보존
- existing CLI와 동일 config를 resolve하는 test

MVP에서는 모든 모델의 모든 필드를 완벽히 schema화할 필요는 없다. 지원 범위를 명시하고 unsupported field를 silent drop하지 마라.

### 4.3 Immutable run allocator

```text
runs/<experiment_id>/<run_id>/
```

- 새 run은 반드시 새 directory
- atomic create
- concurrent identical config collision test
- legacy deterministic path는 importer/read-only mapping으로만 취급

### 4.4 SQLite registry

```text
bench/control/registry/schema.py
bench/control/registry/sqlite.py
bench/control/registry/migrations/
```

최소 table:

- schema_migrations
- experiments
- runs
- run_state_transitions
- run_actions
- workers
- gpu_leases
- checkpoints
- artifacts
- legacy_run_mappings

설정:

- WAL
- foreign keys
- busy timeout
- short transaction
- state_version optimistic concurrency

최소 state:

```text
CREATED, VALIDATING, QUEUED, STARTING, RUNNING,
COMPLETED, FAILED, CANCELLED, ORPHANED
```

`STOP_REQUESTED`, `CHECKPOINTING`, `INTERRUPTED`, `RESUMING`은 schema에 포함하되 이번 tranche에서 control 기능을 활성화하지 마라.

### 4.5 Event journal

```text
bench/control/events/schema.py
bench/control/events/writer.py
bench/control/events/reader.py
```

event type:

- status
- metric
- log
- resource
- checkpoint
- artifact
- warning
- failure

요구사항:

- run별 monotonic event_id
- JSONL append
- partial last line recovery
- bounded query by cursor
- large payload artifact indirection
- terminal/status event flush policy

### 4.6 Worker process

```text
bench/control/process/manager.py
bench/control/process/worker_cli.py
bench/control/process/signals.py
```

worker CLI 예시:

```text
python -m bench.control.process.worker_cli \
  --run-id ... \
  --registry ... \
  --run-spec ...
```

요구사항:

- `shell=True` 금지
- argv list 사용
- new process group/session
- PID, PGID, host, process start time, worker token 기록
- stdout/stderr file redirection
- heartbeat
- exit code contract
- ordinary exception → FAILED + traceback artifact
- abrupt death → ORPHANED candidate
- UI/server 종료와 독립

현재 `bench/runners/orchestrate.py`의 scaffold를 재사용할 수 있는지 검토하되, active runner에 무리하게 끼워 맞추지 마라.

### 4.7 Observer/instrumentation

```text
bench/control/events/observer.py
```

최소 interface 예시:

```python
class RunObserver(Protocol):
    def status(...): ...
    def metric(...): ...
    def log(...): ...
    def artifact(...): ...
```

우선 적용:

- runner phase boundary
- `kalmannet_tsp`
- `split_knet`
- model-based baseline

최소 metric:

- train loss
- validation loss/metric
- global step
- final test metric

adapter가 event를 지원하지 않으면 capability/coverage에 명시하고 stdout parsing으로 대체하지 마라.

### 4.8 Telemetry

```text
bench/control/telemetry/base.py
bench/control/telemetry/cpu.py
bench/control/telemetry/nvidia.py
```

- `psutil` 필요 여부를 dependency 정책에 맞게 결정
- NVML 또는 `nvidia-smi` fallback
- torch allocator metric은 별도 label
- whole GPU와 process GPU memory 구분
- CPU-only null-safe
- configurable interval
- collector failure가 run failure를 일으키지 않음

### 4.9 FastAPI service/API

```text
bench/control/api/app.py
bench/control/api/routers/
```

최소 endpoint:

```text
GET /api/v1/system/health
GET /api/v1/system/gpus
GET /api/v1/runs
GET /api/v1/runs/{run_id}
GET /api/v1/runs/{run_id}/events
GET /api/v1/runs/{run_id}/artifacts
GET /api/v1/capabilities
```

read-only dashboard 범위에서는 destructive action endpoint를 노출하지 않아도 된다.

### 4.10 Dash dashboard

```text
bench/ui/dash_app.py
bench/ui/pages/runs.py
bench/ui/pages/run_detail.py
bench/ui/pages/system.py
```

요구사항:

- Dash with FastAPI backend
- multi-page routing
- run table
- state/heartbeat/phase/step
- live metric chart
- resource chart
- bounded log tail
- artifacts/failure/provenance summary
- legacy Inspector deep link
- run_id가 row/route key
- capability/fidelity badge

초기 live update는 bounded polling으로 구현하라. WebSocket은 interface 또는 future note만 남기고, event cursor/reconnect test 없이 premature하게 도입하지 마라.

### 4.11 Legacy mapping

- 기존 `meta.json` discovery를 read-only importer에서 재사용 가능한지 검토
- 원 directory 수정 금지
- synthetic run record는 `legacy=true`
- exact resume capability false
- status confidence/unknown field 표시

## 5. UI에서 표시할 모델 정보

최소:

- model_id
- implementation_id
- init_id
- variant_id short form
- paper_fidelity_status
- trainable
- exact-resume certified 여부

현재 adapter가 실행된다는 이유만으로 paper fidelity를 verified로 표시하지 마라.

## 6. 테스트 요구사항

최소 unit/integration:

- identity stability/collision
- config round-trip/unknown key
- immutable concurrent run allocation
- state transition validation
- SQLite concurrency smoke
- JSONL partial-tail recovery
- worker UI independence
- stdout/stderr capture
- heartbeat/orphan detection
- CPU-only telemetry
- NVIDIA telemetry availability가 있는 환경에서는 smoke
- API list/detail/events
- Dash page browser smoke
- legacy import
- 기존 28개 variant regression

가능하면 Dash 공식 pytest fixture 또는 Playwright/Selenium 계열을 사용하되, 테스트 dependency를 불필요하게 중복 추가하지 마라.

## 7. 문서와 보고서 산출물

다음 파일을 작성하라.

```text
docs/benchmark_visualization/implementation_baseline.md
docs/benchmark_visualization/implementation_status_phase0_phase1_phase3.md
docs/benchmark_visualization/known_limitations.md
docs/benchmark_visualization/operator_quickstart.md
```

`implementation_status`에는 다음을 포함한다.

- 구현한 module과 public interface
- DB schema version
- event schema version
- dashboard 실행 명령
- worker 실행 명령
- test 결과
- 기존 CLI와의 호환성
- instrumented model coverage
- legacy import coverage
- 미구현 기능
- 발견한 blocker
- 다음 tranche 권장 순서

## 8. 완료 기준

이번 작업은 다음이 모두 충족되어야 완료다.

1. 같은 config의 두 run이 서로 다른 run_id/path를 얻는다.
2. run이 별도 process로 실행된다.
3. FastAPI/Dash를 종료해도 worker가 계속된다.
4. server 재시작 후 run state와 event를 다시 읽는다.
5. metric/log/resource가 terminal 이전에 dashboard에서 보인다.
6. worker exception이 FAILED와 traceback으로 남는다.
7. abrupt worker death가 완료로 오인되지 않는다.
8. 기존 Streamlit Inspector가 깨지지 않는다.
9. exact resume/Stop 버튼이 잘못 활성화되지 않는다.
10. 전체 신규 테스트와 기존 regression 결과가 보고된다.

## 9. 작업 방식

- 먼저 조사하고 작은 commit 단위로 구현하라.
- 기존 public behavior를 바꿀 때 compatibility test를 먼저 작성하라.
- 대규모 파일 이동이나 unrelated refactor를 피하라.
- 발견한 중요한 버그는 구현 상태 보고서에 즉시 기록하라.
- 장시간 full training 대신 tiny/synthetic smoke run을 사용하라.
- 기존 artifact를 덮어쓸 위험이 있으면 새 isolated test root를 사용하라.
- 구현할 수 없는 항목은 그럴듯하게 완료 처리하지 말고 정확한 이유와 최소 선행 변경을 적어라.

## 10. 종료 시 터미널 요약

작업 완료 후 다음만 간결히 출력하라.

1. 변경한 주요 파일
2. DB/event schema version
3. dashboard/worker 실행 명령
4. test 결과
5. 구현된 model instrumentation 범위
6. 남은 blocker 3개
7. 다음 tranche 권장 작업 3개

---

## 다음 tranche 참고

이 foundation이 acceptance test를 통과한 뒤에만 다음을 구현한다.

1. checkpoint v1 atomic schema
2. warm start/resume API 분리
3. KNet/Split exact-resume parity certification
4. graceful stop state transition
5. config GUI/launch
6. checkpoint/resume control UI
