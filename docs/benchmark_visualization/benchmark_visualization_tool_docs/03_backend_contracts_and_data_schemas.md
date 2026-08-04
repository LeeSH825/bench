# Benchmark Execution Visualization Tool — Backend Contracts and Data Schemas

## 1. 목적

이 문서는 frontend, runner, adapter, worker, registry가 공유해야 할 versioned contract를 정의한다. 구체 Python 구현보다 **의미와 불변조건**이 우선이다.

권장 구현은 Pydantic v2 또는 동등한 typed validation layer이며, JSON Schema를 export해 Dash form과 API validation에서 같은 source of truth를 사용한다.

## 2. Identity contract

### 2.1 식별자

| 이름 | 의미 | 생성 방식 | 변경 가능성 |
|---|---|---|---|
| `experiment_id` | 반복 run을 묶는 논리적 실험 | user slug + UUID/ULID | immutable |
| `run_id` | 한 번의 실행 | UUIDv7 또는 ULID 권장 | immutable |
| `model_id` | 알고리즘 family | registry canonical ID | immutable per run |
| `implementation_id` | concrete adapter/upstream 구현 | versioned string | immutable per run |
| `init_id` | 초기화 provenance | enum + detail | immutable per run |
| `variant_id` | 비교 가능한 구현 variant | canonical hash | immutable |
| `checkpoint_id` | checkpoint identity | UUID/ULID | immutable |
| `artifact_id` | artifact identity | UUID/ULID 또는 content hash | immutable |
| `parent_run_id` | clone/resume 계보의 직계 parent | run_id | immutable |
| `resumed_from_checkpoint_id` | resume source | checkpoint_id | immutable |

### 2.2 금지

- directory path를 run ID로 사용하지 않는다.
- `variant_label`을 DB key로 사용하지 않는다.
- Python `hash()`를 persistent identity에 사용하지 않는다.
- model_id만으로 trained/untrained/pretrained variant를 식별하지 않는다.

### 2.3 Variant identity 입력

```json
{
  "model_id": "split_knet",
  "implementation_id": "bench_split_adapter_v1",
  "architecture_fingerprint": "sha256:...",
  "init": {
    "mode": "trained",
    "source_checkpoint_hash": null,
    "source_run_id": null
  },
  "structural_config_hash": "sha256:..."
}
```

표시 문자열은 별도 생성한다.

```text
Split-KalmanNet · bench_split_adapter_v1 · trained
```

## 3. AdapterCapabilities

UI는 모델 이름으로 기능을 추측하지 않는다.

```json
{
  "schema_version": 1,
  "model_id": "split_knet",
  "implementation_id": "bench_split_adapter_v1",
  "display_name": "Split-KalmanNet",
  "trainable": true,
  "supports_evaluation": true,
  "supports_warm_start": true,
  "supports_graceful_stop": false,
  "supports_checkpoint": true,
  "supports_exact_resume": false,
  "resume_boundary": null,
  "training_phases": ["train", "validation", "test"],
  "optimizer_slots": ["main"],
  "scheduler_slots": [],
  "amp_supported": false,
  "online_adaptation_supported": false,
  "paper_fidelity_status": "partial",
  "paper_fidelity_note": "Current adapter uses one optimizer and is not certified as paper-faithful alternating optimization."
}
```

`supports_exact_resume=true`는 parity test를 통과한 implementation/version에만 허용한다.

## 4. ResolvedRunSpec

`ResolvedRunSpec`은 CLI와 GUI launch가 공유하는 immutable 실행 명세이다.

```json
{
  "schema_version": 1,
  "experiment": {
    "experiment_id": "01J...",
    "name": "viz_attitude_bias",
    "description": "...",
    "tags": ["attitude", "bias"]
  },
  "identity": {
    "run_id": "01J...",
    "model_id": "split_knet",
    "implementation_id": "bench_split_adapter_v1",
    "init_id": "trained",
    "variant_id": "sha256:..."
  },
  "system": {
    "task_id": "...",
    "scenario_id": "...",
    "state_dim": 6,
    "observation_dim": 3,
    "scenario_config": {}
  },
  "dataset": {
    "dataset_id": "...",
    "fingerprint": "sha256:...",
    "train_uri": "...",
    "val_uri": "...",
    "test_uri": "...",
    "split_seed": 42
  },
  "training": {
    "enabled": true,
    "max_updates": 1000,
    "batch_size": 32,
    "gradient_accumulation_steps": 1,
    "validation_interval_updates": 50,
    "early_stopping": {}
  },
  "optimizer": {
    "name": "adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0
  },
  "initialization": {
    "mode": "trained",
    "checkpoint_id": null,
    "checkpoint_uri": null
  },
  "resume": {
    "mode": "none",
    "checkpoint_id": null,
    "allowed_overrides": []
  },
  "runtime": {
    "device": "cuda:0",
    "precision": "fp32",
    "deterministic": true,
    "seed": 7,
    "num_workers": 0
  },
  "telemetry": {
    "enabled": true,
    "interval_seconds": 2.0
  },
  "artifacts": {
    "root_uri": "runs/...",
    "save_predictions": true,
    "emit_visualization": true,
    "checkpoint_policy": {}
  },
  "provenance": {
    "git_commit": "...",
    "git_dirty": true,
    "submodule_revisions": {},
    "environment_fingerprint": "sha256:..."
  },
  "hashes": {
    "structural_config_hash": "sha256:...",
    "operational_config_hash": "sha256:...",
    "resolved_spec_hash": "sha256:..."
  }
}
```

### 4.1 Structural vs operational field

Structural field 예시:

- model architecture/layout
- task/system dimensions
- dataset fingerprint/split
- loss semantics
- optimizer class와 parameter grouping
- training phase 구조
- precision semantics가 model numerics에 영향을 줄 경우

Operational field 예시:

- UI label
- telemetry interval
- log verbosity
- output display preference
- device relocation이 인증된 경우의 device index

Exact resume에서는 structural hash mismatch를 거부한다. Operational override는 allowlist와 audit record가 있어야 한다.

## 5. RunRecord

```json
{
  "run_id": "01J...",
  "experiment_id": "01J...",
  "state": "RUNNING",
  "state_version": 8,
  "created_at": "2026-07-30T12:00:00Z",
  "updated_at": "2026-07-30T12:03:10Z",
  "started_at": "2026-07-30T12:00:05Z",
  "ended_at": null,
  "host": "workstation-a",
  "pid": 12345,
  "process_group_id": 12345,
  "heartbeat_at": "2026-07-30T12:03:09Z",
  "worker_instance_id": "01J...",
  "gpu_lease_id": "01J...",
  "device": "cuda:0",
  "phase": "train",
  "subphase": null,
  "global_step": 120,
  "epoch": 3,
  "batch_cursor": 8,
  "last_event_id": 344,
  "latest_checkpoint_id": "01J...",
  "best_checkpoint_id": "01J...",
  "parent_run_id": null,
  "resumed_from_run_id": null,
  "resumed_from_checkpoint_id": null,
  "exit_code": null,
  "terminal_reason": null,
  "error_summary": null
}
```

### 5.1 State invariants

- terminal state는 다시 non-terminal state로 변하지 않는다.
- resume는 parent run을 다시 RUNNING으로 바꾸지 않고 child run을 만든다.
- `state_version`은 optimistic concurrency control에 사용한다.
- worker만 `STARTING` 이후의 실행 상태를 확정할 수 있다.
- UI는 직접 상태를 바꾸지 않고 action request를 생성한다.

## 6. Run state machine

```text
CREATED
VALIDATING
QUEUED
STARTING
RUNNING
STOP_REQUESTED
CHECKPOINTING
INTERRUPTED
RESUMING
COMPLETED
FAILED
CANCELLED
ORPHANED
```

허용 전이:

```text
CREATED → VALIDATING
VALIDATING → QUEUED | FAILED | CANCELLED
QUEUED → STARTING | CANCELLED
STARTING → RUNNING | FAILED | ORPHANED
RUNNING → STOP_REQUESTED | CHECKPOINTING | COMPLETED | FAILED | ORPHANED
STOP_REQUESTED → CHECKPOINTING | CANCELLED | FAILED
CHECKPOINTING → RUNNING | INTERRUPTED | COMPLETED | FAILED
INTERRUPTED → RESUMING
RESUMING → RUNNING | FAILED
```

`ORPHANED`는 자동으로 `FAILED`나 `INTERRUPTED`가 아니다. PID, process start time, heartbeat, checkpoint integrity를 확인한 후 researcher action이 필요하다.

## 7. Event schema

```json
{
  "schema_version": 1,
  "event_id": 345,
  "run_id": "01J...",
  "timestamp": "2026-07-30T12:03:10.123Z",
  "event_type": "metric",
  "phase": "train",
  "subphase": null,
  "step_type": "global_step",
  "step": 120,
  "name": "loss/state_mse",
  "value": 0.00123,
  "unit": "mse",
  "level": null,
  "message": null,
  "payload": {
    "batch_size": 32
  }
}
```

### 7.1 Event type

- `status`
- `metric`
- `log`
- `resource`
- `checkpoint`
- `artifact`
- `control`
- `warning`
- `failure`

### 7.2 Metric naming

공통 namespace:

```text
loss/train_total
loss/validation_total
metric/test_mse
metric/test_mse_db
progress/global_step
progress/epoch
throughput/sequences_per_sec
latency/update_ms
```

모델 전용 namespace:

```text
model.split/prior_cov_loss
model.split/innovation_cov_loss
model.maml/outer_loss
model.maml/inner_loss
model.aknet/hypernetwork_loss
```

`metrics_step.csv`의 sequence time index와 optimizer global step를 혼용하지 않는다.

### 7.3 JSONL write 규칙

- UTF-8 한 줄 한 event
- event ID monotonic per run
- write + flush 정책 설정
- checkpoint/status terminal event는 fsync
- partial last line은 reader가 무시하고 recovery warning 생성
- unbounded payload 금지
- large tensor/array는 artifact로 저장하고 event에는 URI/hash만 기록

## 8. ResourceSample

```json
{
  "timestamp": "2026-07-30T12:03:10Z",
  "run_id": "01J...",
  "pid": 12345,
  "process_tree_cpu_percent": 82.1,
  "process_tree_rss_bytes": 12400000000,
  "system_cpu_percent": 65.0,
  "system_ram_used_bytes": 48000000000,
  "gpu": {
    "backend": "nvidia",
    "device_index": 0,
    "device_uuid": "GPU-...",
    "device_utilization_percent": 91,
    "device_memory_used_bytes": 32000000000,
    "device_memory_total_bytes": 102000000000,
    "process_memory_used_bytes": 28000000000,
    "temperature_c": 62,
    "power_w": 310,
    "attribution_quality": "memory_only"
  },
  "disk": {
    "run_dir_bytes": 1200000000,
    "free_bytes": 400000000000
  },
  "collector_errors": []
}
```

전체 GPU utilization과 해당 PID의 GPU memory를 명확히 구분한다. attribution이 불완전하면 UI에 표시한다.

## 9. CheckpointManifest

```json
{
  "schema_version": 1,
  "checkpoint_id": "01J...",
  "run_id": "01J...",
  "kind": "interrupt",
  "created_at": "2026-07-30T12:05:00Z",
  "model_id": "split_knet",
  "implementation_id": "bench_split_adapter_v1",
  "variant_id": "sha256:...",
  "phase": "train",
  "subphase": null,
  "resume_boundary": "optimizer_update",
  "cursor": {
    "global_step": 150,
    "epoch": 4,
    "batch_cursor": 2,
    "gradient_accumulation_step": 0
  },
  "components": {
    "models": ["main"],
    "optimizers": ["main"],
    "schedulers": [],
    "grad_scaler": false,
    "rng": true,
    "sampler": true,
    "early_stopping": true
  },
  "compatibility": {
    "structural_config_hash": "sha256:...",
    "dataset_fingerprint": "sha256:...",
    "code_fingerprint": "sha256:..."
  },
  "payload_uri": "checkpoints/01J....pt",
  "payload_sha256": "...",
  "payload_bytes": 12345678,
  "event_cursor": 530,
  "complete": true
}
```

### 9.1 Checkpoint payload 필수 후보

- all model states
- all optimizer states
- scheduler states
- GradScaler
- epoch/global step/batch/phase/subphase
- gradient accumulation cursor
- early stopping and best metric
- Python, NumPy, Torch CPU/CUDA RNG
- DataLoader generator/sampler state
- dataset/config/code identity
- event cursor
- resume lineage

모델별로 필요하지 않은 slot은 명시적으로 empty로 기록한다.

### 9.2 Atomic write

```text
serialize temporary file
→ flush
→ fsync file
→ compute hash
→ atomic replace final payload
→ fsync directory
→ insert checkpoint registry row
→ append checkpoint event
```

registry row가 payload보다 먼저 committed되면 안 된다.

## 10. ArtifactRecord

```json
{
  "artifact_id": "01J...",
  "run_id": "01J...",
  "kind": "predictions|metrics|visualization|config|log|failure|environment",
  "uri": "artifacts/preds_test.npz",
  "sha256": "...",
  "bytes": 1234,
  "media_type": "application/octet-stream",
  "created_at": "...",
  "complete": true,
  "metadata": {}
}
```

## 11. ControlActionRequest

```json
{
  "action_id": "01J...",
  "run_id": "01J...",
  "action": "STOP_GRACEFUL",
  "requested_at": "...",
  "requested_by": "local-user",
  "expected_state_version": 8,
  "parameters": {},
  "status": "PENDING",
  "handled_at": null,
  "result": null
}
```

지원 action:

- `STOP_GRACEFUL`
- `TERMINATE`
- `KILL`
- `RESUME`
- `WARM_START`
- `RETRY_EVALUATION`
- `MARK_FAILED`
- `IMPORT_LEGACY_RUN`

모든 destructive action은 idempotency key를 가진다.

## 12. SQLite 최소 schema

```text
schema_migrations
experiments
runs
run_state_transitions
run_actions
checkpoints
artifacts
model_capabilities
gpu_leases
workers
legacy_run_mappings
```

### 12.1 runs 주요 index

- state, updated_at
- experiment_id, created_at
- model_id, implementation_id, init_id
- parent_run_id
- heartbeat_at
- device

### 12.2 SQLite 운영 규칙

- WAL mode
- foreign key on
- busy timeout
- short transaction
- UI는 long read transaction 금지
- event high-rate payload 전체를 DB에 넣지 않음
- migration은 forward-only + backup

## 13. API v1

### Config/capability

```text
GET  /api/v1/capabilities
GET  /api/v1/config-presets
GET  /api/v1/config-presets/{preset_id}
POST /api/v1/run-specs/validate
POST /api/v1/run-specs/resolve
```

### Runs

```text
POST /api/v1/runs
GET  /api/v1/runs
GET  /api/v1/runs/{run_id}
GET  /api/v1/runs/{run_id}/lineage
POST /api/v1/runs/{run_id}/actions/stop
POST /api/v1/runs/{run_id}/actions/terminate
POST /api/v1/runs/{run_id}/actions/resume
POST /api/v1/runs/{run_id}/actions/warm-start
POST /api/v1/runs/{run_id}/clone
```

### Events/resources

```text
GET /api/v1/runs/{run_id}/events?after_event_id=123&limit=1000
GET /api/v1/runs/{run_id}/resources?from=...&to=...
WS  /api/v1/runs/{run_id}/stream
```

### Checkpoint/artifact

```text
GET  /api/v1/runs/{run_id}/checkpoints
GET  /api/v1/checkpoints/{checkpoint_id}
POST /api/v1/checkpoints/{checkpoint_id}/validate
GET  /api/v1/runs/{run_id}/artifacts
GET  /api/v1/artifacts/{artifact_id}
```

### System

```text
GET /api/v1/system/health
GET /api/v1/system/resources
GET /api/v1/system/gpus
GET /api/v1/system/workers
```

## 14. Worker CLI contract

```text
python -m bench.control.process.worker_cli \
  --run-id <run_id> \
  --registry <sqlite_path> \
  --run-spec <resolved_run_spec.json>
```

Exit code:

| Code | 의미 |
|---:|---|
| 0 | completed |
| 10 | gracefully interrupted with valid checkpoint |
| 20 | cancelled before execution |
| 30 | validation/config incompatibility |
| 40 | training/evaluation failure |
| 50 | checkpoint write failure |
| 60 | worker protocol failure |
| 70 | external termination detected |

Exit code만으로 상태를 정하지 않고 registry terminal transition과 함께 검증한다.

## 15. Run directory layout

```text
runs/<experiment_id>/<run_id>/
  original_config.yaml
  resolved_run_spec.json
  run_manifest.json
  events.jsonl
  stdout.log
  stderr.log
  failure.json
  checkpoints/
    <checkpoint_id>.pt
    <checkpoint_id>.manifest.json
  artifacts/
    metrics.json
    metrics_step.csv
    predictions/
    visualization/
  provenance/
    git.json
    environment.json
    pip_freeze.txt
  tmp/
```

`tmp/` 외부에 partial artifact를 노출하지 않는다.

## 16. Legacy import contract

기존 deterministic run directory를 삭제·이동하지 않는다. Importer는 다음을 수행한다.

- legacy path에서 synthetic `run_id` 생성
- read-only `legacy=true` record 생성
- config/model/init/viz metadata 추출
- status를 best-effort로 판정하고 confidence 표시
- checkpoint를 resume-certified로 표시하지 않음
- 원 path와 hash를 mapping table에 기록

## 17. Security contract

local-only라도 다음을 적용한다.

- output root allowlist
- absolute path와 `..` traversal 거부
- dynamic import allowlist
- shell string 실행 금지; argv list 사용
- uploaded YAML을 code로 평가하지 않음
- checkpoint pickle load는 trusted local artifact로 제한
- reverse proxy 없이 `0.0.0.0` bind 금지
- destructive action audit log

## 18. Versioning

- `schema_version`은 모든 top-level object에 존재
- API는 `/api/v1`
- event reader는 최소 한 버전 backward compatibility
- checkpoint migration은 명시적이며 silent conversion 금지
- unsupported newer schema는 명확히 거부
