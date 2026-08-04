# Benchmark Execution Visualization Tool — UI Information Architecture and Workflows

## 1. UI 원칙

- UI는 run state의 source of truth가 아니다.
- 버튼 label은 실제 semantics를 그대로 드러내야 한다.
- 실행 가능 여부와 exact-resume 인증 여부를 분리한다.
- model family, implementation, initialization을 동시에 표시한다.
- live data가 없거나 stale하면 0으로 표시하지 말고 `Unavailable` 또는 `Stale`로 표시한다.
- error와 warning을 숨기지 않는다.
- read-only legacy run과 controllable new run을 명확히 구분한다.

## 2. Global navigation

```text
Overview
Runs
New Run
Checkpoints
Compare
System
Documentation
```

### 2.1 Global header

- repository/commit
- registry health
- worker manager health
- GPU lease summary
- active run count
- current time/host
- legacy Inspector link

## 3. Overview page

목적: 현재 workstation 상태를 한눈에 파악한다.

### 영역

1. Active runs
2. Queued/stop requested/orphaned count
3. GPU cards
4. Recent failures
5. Recent checkpoints
6. Disk usage
7. Registry/event health

### GPU card

반드시 다음을 구분한다.

- whole device utilization
- whole device memory
- benchmark PID memory
- device temperature/power
- lease owner run
- attribution quality

## 4. Runs page

### 4.1 Table columns

- state badge
- run name
- run_id short form
- model display name
- implementation_id
- init_id
- paper fidelity
- phase / global step
- device
- start time / elapsed
- heartbeat age
- latest metric
- latest checkpoint
- parent/child indicator

### 4.2 Filters

- state
- experiment
- model
- implementation
- init
- tag
- device
- date range
- legacy/new
- resume-certified

### 4.3 Table behavior

- stable run_id가 row key
- server-side pagination
- default sort: active first, then latest
- `ORPHANED`, `FAILED`, `STOP_REQUESTED`는 색상 외 text/icon으로도 구분
- 동일 model_id의 여러 init/implementation은 별도 row

## 5. New Run page

Wizard를 권장한다.

### Step 1 — Preset

- existing preset
- clone from run
- clone from checkpoint as warm start
- blank advanced config

### Step 2 — Task and dataset

- task/scenario
- dataset source
- split/seed
- dataset fingerprint preview
- validation error

### Step 3 — Model and implementation

- model_id
- implementation_id
- paper_fidelity_status
- capability badges
- initialization mode

### Step 4 — Training/runtime

- budget/update
- optimizer
- validation/early stopping
- device/precision
- telemetry
- artifact policy

### Step 5 — Review

- original vs resolved diff
- structural hash
- output run ID/path preview
- estimated artifact policy
- warnings
- generated worker command preview

### Step 6 — Launch

- final validation
- immutable run allocation
- launch result

## 6. Form과 raw YAML editor

- typed object가 source of truth이다.
- form → typed object → canonical YAML
- raw YAML → parse/validate → typed object → form
- view switch 전에 validation한다.
- original input은 attachment로 보존한다.
- unknown key는 silent drop하지 않는다.
- comment preservation을 보장하지 못하면 UI에서 명시한다.

### Field state

- valid
- invalid
- conditionally disabled
- unsupported by implementation
- advanced
- resume-locked

## 7. Run Detail page

### 7.1 Header

- run state
- model / implementation / init
- run_id
- lineage
- started/elapsed/heartbeat
- device/GPU lease
- exact resume capability
- paper fidelity

### 7.2 Tabs

```text
Live
Metrics
Resources
Logs
Checkpoints
Artifacts
Config
Provenance
Failure
Lineage
```

### 7.3 Live tab

- phase/subphase
- global step/epoch/batch
- progress denominator가 확정된 경우에만 progress bar
- train/validation loss
- latest validation metric
- throughput/update time
- latest event timestamp
- stop/terminate controls

### 7.4 Metric chart

- event step type를 축 label에 표시
- train/validation/test를 구분
- dB와 linear MSE를 같은 axis에 혼용하지 않음
- downsampling은 visual-only이며 원 event는 보존
- NaN/divergence marker

### 7.5 Resource chart

- whole GPU and process GPU memory separate series
- CPU process tree and system CPU separate
- RAM process tree and system RAM separate
- telemetry gap/stale band
- collector error annotation

### 7.6 Log viewer

- virtualized tail
- stdout/stderr/logger source filter
- severity filter
- follow tail on/off
- event ID/timestamp
- ANSI stripping 또는 안전한 rendering
- download artifact link

## 8. Control UX

### 8.1 Stop safely

표시 조건:

- state가 RUNNING
- adapter가 graceful stop/checkpoint boundary를 지원

확인 dialog:

```text
현재 안전한 optimizer-update 경계까지 진행한 후 interrupt checkpoint를 저장합니다.
즉시 중단되지 않을 수 있습니다.
```

버튼 결과:

- action request 생성
- UI는 `STOP_REQUESTED`를 표시
- checkpoint complete event 후 `INTERRUPTED`

### 8.2 Force terminate

확인 dialog:

```text
프로세스 그룹을 종료합니다. 최신 학습 상태가 저장되지 않을 수 있으며 exact resume가
불가능할 수 있습니다.
```

두 단계 확인 또는 run name 입력을 권장한다.

### 8.3 Resume training

표시 조건:

- source checkpoint complete/hash valid
- implementation capability `supports_exact_resume=true`
- compatibility check pass

동작:

- 새 child run 생성
- parent/resumed_from 기록
- original run은 immutable terminal state 유지

### 8.4 Warm start new run

- weight만 load
- optimizer/cursor/RNG는 새로 시작
- UI에 `Warm start` badge
- resume와 같은 버튼/메뉴에 넣지 않음

## 9. Checkpoints page

### columns

- checkpoint_id
- source run
- kind: periodic/best/interrupt/final
- phase/cursor
- created time
- payload size/hash
- compatibility status
- exact-resume certified
- pinned/retention

### actions

- validate integrity
- evaluate as child run
- resume child run
- warm-start child run
- pin/unpin
- reveal artifact path

## 10. Compare page

### 비교 identity

- run_id로 row/trace를 식별
- variant_id로 grouping
- variant_label은 display only

### 비교 기능

- final metrics
- training curves
- trajectories
- resource usage
- parameter count/inference time
- config diff
- provenance diff
- implementation/fidelity notice

기존 Streamlit Run Inspector와 parity가 확보되기 전에는 deep link를 제공하고 동일 기능을 중복 구현하지 않는다.

## 11. System page

- GPU/CPU/RAM/disk
- active leases
- worker process table
- registry status/WAL size
- event lag
- orphan candidates
- dependency versions
- Git/submodule dirty status
- config/security policy

## 12. Live transport 정책

### MVP

- `dcc.Interval` 또는 equivalent bounded polling
- run list: 2~5초
- active run detail: 1~2초
- terminal run: polling 중지
- log: event cursor 이후 증분 fetch

### 이후 WebSocket

- server push는 UI latency 개선용
- source of truth는 여전히 DB/JSONL
- reconnect 시 `after_event_id`로 gap fill
- backpressure와 max batch size
- browser close 시 worker에 영향을 주지 않음

## 13. URL/deep link

권장 route:

```text
/runs
/runs/<run_id>
/runs/<run_id>?tab=checkpoints
/new-run?preset=<id>
/new-run?clone_run=<run_id>
/checkpoints/<checkpoint_id>
/compare?run=<id>&run=<id>
/system
/legacy-inspector?run_id=<id>
```

full comparison state가 길면 saved comparison ID를 사용한다.

## 14. Error와 stale 상태

### UI에서 반드시 구분

- API unavailable
- registry locked/busy
- event journal lag
- heartbeat stale
- process missing
- checkpoint corrupt
- telemetry unsupported
- GPU attribution incomplete
- legacy run incomplete
- config validation error
- resume incompatibility

`Unknown`을 `0`, `Idle`, `Completed`로 표시하지 않는다.

## 15. 접근성과 안전성

- 색상만으로 상태를 구분하지 않는다.
- destructive button은 위치/색/label을 분리한다.
- keyboard navigation과 focus order
- log/metric chart의 text summary
- long identifiers copy button
- timezone과 absolute timestamp 제공
- remote bind/auth warning 표시

## 16. Frontend component boundary

Dash callback은 다음 service만 호출한다.

```text
RunQueryService
RunCommandService
ConfigService
EventQueryService
CheckpointService
ArtifactService
SystemResourceService
```

callback 안에서 다음을 직접 하지 않는다.

- subprocess spawn
- sqlite raw schema manipulation
- checkpoint load
- third-party adapter import
- filesystem recursive scan
- YAML merge logic
