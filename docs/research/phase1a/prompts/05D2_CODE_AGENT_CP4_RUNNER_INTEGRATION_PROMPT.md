# Phase 1A CP4 Integration Step 2 Execution Contract
## Registry, Task Dispatch, Runner, Verified Cache, and Smoke Suite

당신은 `/home/dss-pc-05/bench` repository에서 Phase 1A의 마지막 integration 작업만 수행한다.

이 작업의 이전 명칭은 Gate D2였으나, 새로운 연구 Gate가 아니다.

```text
P1A-CP4 Integration
├─ Step 1: MEKF adapter/artifact bridge  — 완료
└─ Step 2: registry/dispatch/runner      — 이번 실행
```

이번 Step이 통과하면 Phase 1A 기반 구축을 종료한다.
다음은 Phase 1B classical MEKF benchmark completion이며, neural model로 자동 진행하지 마라.

---

# 1. 반드시 먼저 읽을 문서와 source

다음을 처음부터 끝까지 읽어라.

```text
docs/research/phase1a/P1A_CP4_STEP1_FINAL_APPROVAL.md
docs/research/phase1a/P1A_MEKF_ADAPTER_ARTIFACT_CONTRACT.md
docs/research/phase1a/P1A_GATE_D1_TEST_MATRIX.md
experiments/phase1a/reports/P1A_GATE_D1_VALIDATION_REPORT.md

docs/research/phase1a/P1A_GATE_C_FINAL_APPROVAL.md
docs/research/phase1a/P1A_CANONICAL_MEKF_METRICS_CONTRACT.md
docs/research/phase1a/P1A_BASILISK_UNIT_ST_CONTRACT.md
docs/research/phase1a/P1A_EVENT_SCHEMA_CONTRACT.md
docs/research/phase1a/P1A_IMPLEMENTATION_CONTRACT.md

docs/research/phase1a/P1A_RISK_REGISTER.md
docs/research/phase1a/P1A_IMPLEMENTATION_MAP.md

bench/models/mekf.py
bench/models/base.py
bench/models/registry.py
bench/tasks/bench_generated.py
bench/runners/run_suite.py
bench/tasks/data_format.py
bench/tasks/generator/contract.py

bench/tasks/generator/mekf_events.py
bench/tasks/generator/unit_st_synthetic.py
bench/tasks/generator/basilisk_unit_st.py
bench/metrics/mekf.py
```

다음 tests도 읽어라.

```text
tests/test_mekf_adapter.py
tests/test_mekf_events.py
tests/test_mekf_replay.py
tests/test_basilisk_unit_st_generator.py
tests/test_mekf_metrics.py
```

---

# 2. 승인 baseline

다음 결과를 baseline으로 사용하라.

```text
Gate A / P1A-CP1: 55 passed
Gate B1 / P1A-CP2a: 55 passed
Gate B2 / P1A-CP2b: 67 passed
Gate C / P1A-CP3: 43 passed
D1 / P1A-CP4 Step 1: 24 passed
Legacy: 18 passed, 5 subtests passed
```

pass/fail은 exit code와 contract 보존으로 판정하며 시험 개수는 provenance로 기록한다.

---

# 3. Current-tree 및 dirty-tree 정책

실행 시작 시점의 current working tree 전체를 승인된 기준선으로 사용하라.

다음을 승인 조건으로 검토하지 마라.

```text
branch
HEAD
commit history
과거 commit delta
repository 전체 whitespace
기존 visualization artifact
```

금지:

```text
git reset
git restore
git clean
git stash
git add
git commit
git push
git merge
git rebase
git switch
git checkout
```

실행 전 recoverable snapshot, existing dirty fingerprint, frozen-file hash,
allowlist existence를 기록하고 실행 후 비교하라.

외부 unrelated non-source artifact는 읽거나 수정하지 말고 path/status ledger에만 기록하라.

---

# 4. 동결 source

다음은 읽고 import할 수 있으나 수정 금지다.

```text
bench/estimators/**
bench/tasks/generator/mekf_events.py
bench/tasks/generator/unit_st_synthetic.py
bench/tasks/generator/basilisk_unit_st.py
bench/metrics/mekf.py
bench/models/mekf.py
bench/models/base.py

tests/test_mekf_conventions.py
tests/test_mekf_core.py
tests/test_mekf_events.py
tests/test_unit_st_synthetic.py
tests/test_mekf_replay.py
tests/test_basilisk_unit_st_generator.py
tests/test_mekf_metrics.py
tests/test_mekf_adapter.py

docs/research/phase0a/**
docs/research/phase1a/P1A_IMPLEMENTATION_CONTRACT.md
docs/research/phase1a/P1A_EVENT_SCHEMA_CONTRACT.md
docs/research/phase1a/P1A_BASILISK_UNIT_ST_CONTRACT.md
docs/research/phase1a/P1A_CANONICAL_MEKF_METRICS_CONTRACT.md
docs/research/phase1a/P1A_MEKF_ADAPTER_ARTIFACT_CONTRACT.md
```

MEKF math, event replay, sensor generation, metric, bridge를 중복 구현하지 마라.

---

# 5. Exact allowlist

기존 파일 수정:

```text
bench/tasks/bench_generated.py
bench/models/registry.py
bench/runners/run_suite.py
```

신규 파일 생성:

```text
bench/configs/suite_phase1a_unit_st_smoke.yaml
tests/test_mekf_runner_integration.py

docs/research/phase1a/P1A_CP4_INTEGRATION_CONTRACT.md
docs/research/phase1a/P1A_CP4_TEST_MATRIX.md
experiments/phase1a/reports/P1A_CP4_VALIDATION_REPORT.md
```

provenance:

```text
experiments/phase1a/preflight_snapshots/05D2_*/
experiments/phase1a/agent_logs/05D2_*
```

이 목록 밖 source/test/config 변경이 필요하면 수정하지 말고:

```text
BLOCKED_CP4_SCOPE_EXTENSION_REQUIRED
```

로 중단하라.

특히 다음 두 파일이 필요하면 자동 수정하지 마라.

```text
bench/tasks/data_format.py
bench/tasks/generator/contract.py
```

sidecar 전달이 이 파일 변경 없이는 불가능하다는 executable evidence와 최소 migration 설계를 보고하라.

---

# 6. 고정 ID

정확히 다음을 사용하라.

```text
task_family = mekf_unit_st_v1
model_id    = mekf_event_replay_v1
```

기존 ID/동작/metric key를 변경하지 마라.
등록은 append-only여야 한다.

---

# 7. Task dispatch와 typed sidecar

`bench/tasks/bench_generated.py`에 `mekf_unit_st_v1` family를 append-only로 추가하라.

최소 지원 generator mode:

```text
synthetic-unit-st-v1
basilisk-unit-st-v1
```

resolved task config가 어느 producer를 사용할지 명시해야 한다.
model ID는 data generation seed/config/hash에 포함하지 마라.

typed event source of truth는 다음 exact three-file artifact다.

```text
manifest.json
truth.npz
events.npz
```

legacy dense `x/y`를 estimator input source로 사용하지 마라.
gyro/ST를 dense float32 또는 zero-filled y로 변환하지 마라.

runner에 sidecar 경로와 verified identity를 전달할 수 있는 가장 작은 append-only 경로를 사용하라.
기존 data loader API를 거짓으로 만족시키는 dummy sensor sequence를 만들지 마라.

---

# 8. Verified cache 정책

fresh generation 후와 cache hit 시 모두 strict loader를 사용하라.

```python
load_event_dataset(
    path,
    expected_generator_id=<resolved exact producer ID>,
)
```

cache hit는 다음을 모두 검증해야 한다.

```text
schema_version
generator_id
convention_id
truth_hash
sensor_payload_hash
event_order_hash
manifest_hash
dataset_hash
complete resolved config
seed policy/version
source fingerprints
Python/NumPy/SciPy/Basilisk runtime identity where applicable
whole-trajectory split membership
```

파일 존재만으로 cache hit를 승인하지 마라.

불일치 시 정책은 다음 중 하나를 명시적으로 선택하고 문서화하라.

```text
new deterministic cache namespace 생성
또는
fail-loud
```

기존 artifact를 조용히 덮어쓰거나 stale cache를 재사용하지 마라.

---

# 9. Registry

`bench/models/registry.py`에 `mekf_event_replay_v1`을 append-only 등록하라.

D1에서 explicit bridge를 legacy `Type[ModelAdapter]` 계약에 억지로 넣지 않기로 결정했다.

따라서 registry는 다음 중 repository에 가장 작은 안전한 방식을 선택하라.

1. typed-event bridge 전용 registry/table 추가
2. registry entry에 kind/capability metadata 추가
3. runner의 exact ID branch가 bridge factory를 import

기존 model registry의 public lookup behavior를 바꾸지 마라.
legacy model ID에는 영향이 없어야 한다.

bridge는 training을 지원하지 않으며 frozen/deterministic이다.

---

# 10. Runner branch 위치

`bench/runners/run_suite.py`에서 exact task/model pair를 다음보다 먼저 분기하라.

```text
_load_split_npz
_SeqDataset
_predict_batches
legacy dense ModelAdapter lifecycle
```

분기 조건은 정확해야 한다.

```text
task_family == mekf_unit_st_v1
model_id == mekf_event_replay_v1
```

다른 task/model에 이 branch가 실행되면 안 된다.

runner 순서:

```text
resolve config
→ prepare/locate typed sidecar
→ strict load and identity verification
→ choose requested whole-trajectory split
→ construct explicit initial state/time/Qc from task config
→ bridge.replay_events per trajectory
→ collect immutable artifacts
→ after estimation only: exact truth join
→ Gate C canonical metrics
→ write result metrics/artifacts/manifest
```

truth에서 initial state 또는 Qc를 생성하지 마라.

---

# 11. Initial-state 및 Qc configuration

smoke suite는 explicit config로 다음을 제공해야 한다.

```text
initial q_NB
initial b_hat
initial P
initial_time_s
S_g
S_b
또는 resolved Q_c
```

initial estimate가 truth와 같은 zero-noise test를 구성할 수는 있지만,
runner implementation이 truth를 읽어 initial state를 생성해서는 안 된다.

config의 값과 generated truth가 우연히 같도록 fixture를 구성하는 것은 허용한다.
그 equality는 test가 외부에서 확인한다.

모든 array/unit/convention을 manifest와 run artifact에 기록하라.

---

# 12. Runner artifact 저장

정확한 기본 경로:

```text
runs/.../artifacts/mekf_replay/manifest.json
runs/.../artifacts/mekf_replay/trajectory_<trajectory_id>.npz
```

각 trajectory NPZ는 `allow_pickle=False`로 읽을 수 있는 exact arrays를 저장하라.

최소 arrays:

```text
event_index
event_order
timestamp_s
sensor_code
q_hat_NB
b_hat_rad_s
P

st_event_index
st_event_order
st_timestamp_s
st_residual
st_S
```

truth는 estimator artifact NPZ에 섞지 마라.

runner evaluation 결과/metric artifact에는 exact truth-join provenance를 별도로 기록할 수 있다.

`manifest.json`은 canonical sorted compact JSON이며 최소 다음을 포함한다.

```text
task_family
model_id
adapter_id/version
dataset identity
trajectory IDs
per-trajectory counters
artifact filenames
config identity
metric contract/version
fresh_generation 또는 verified_cache_hit
```

generic `preds_test.npz`로 강제 변환하지 마라.

---

# 13. Canonical metrics

Gate C의 `bench.metrics.mekf`만 사용하라.

최소 report:

```text
attitude RMSE rad/deg
attitude P95
attitude max
bias per-axis RMSE
bias vector RMSE
ST NIS count/mean/normalized mean/chi-square interval
state NEES count/mean/normalized mean/chi-square interval
P SPD diagnostics summary
S SPD diagnostics summary
```

NIS는 실제 ST update compact rows만 사용한다.
NEES는 같은 trajectory/timestamp의 posterior q/b/P와 truth q/b를 exact join한다.

timestamp나 count가 맞지 않으면 보간/nearest lookup하지 말고 fail-loud한다.

---

# 14. Same-realization 및 equivalence

필수 세 경로:

```text
Gate B1 direct replay
D1 bridge replay
D2 runner replay
```

동일 serialized artifact, trajectory, initial state/time, Qc에서 다음이
`np.array_equal`이어야 한다.

```text
event index/order/time/sensor
q
b
P
ST residual
ST S
final state
counters
dataset identity
```

metric 결과도 direct/bridge evidence로 계산한 값과 runner 결과가 일치해야 한다.

fresh generation과 verified cache hit 양쪽에서 시험하라.

synthetic와 Basilisk producer 양쪽을 시험하라.

---

# 15. 필수 시험

최소 다음 logical tests를 `tests/test_mekf_runner_integration.py`에 구현하라.

## CP4-01 Config load

새 smoke YAML이 기존 config parser로 로드되고 ID가 정확해야 한다.

## CP4-02 Append-only dispatch/registry

새 ID가 선택되고 기존 대표 ID lookup/dispatch가 변하지 않아야 한다.

## CP4-03 Fresh synthetic end-to-end

fresh typed artifact 생성부터 runner metric/artifact까지 통과해야 한다.

## CP4-04 Fresh Basilisk end-to-end

Basilisk producer에서도 같은 runner path가 통과해야 한다.

## CP4-05 Verified cache hit

두 번째 실행에서 strict verified cache hit를 사용하고 output이 exact-equivalent해야 한다.

## CP4-06 Stale cache rejection

config, generator ID, schema, source fingerprint 또는 semantic hash가 불일치하는 cache를 거부하거나 새 namespace로 분리해야 한다.

## CP4-07 Direct/bridge/runner equivalence

synthetic와 Basilisk 각각 exact equality.

## CP4-08 Same-realization across model metadata

run ID/adapter metadata 변화가 dataset hash와 numeric replay를 바꾸지 않아야 한다.

## CP4-09 Truth boundary

runner가 bridge 호출 전에 truth를 전달하지 않음을 monkeypatch/signature로 증명하라.

## CP4-10 Exact truth join

trajectory/timestamp mismatch를 fail-loud하고, matching case metric이 direct reference와 같아야 한다.

## CP4-11 Lossless artifact round trip

저장한 manifest/NPZ를 `allow_pickle=False`로 읽고 bridge artifact와 exact equal.

## CP4-12 Compact ST evidence

gyro placeholder 없이 실제 update 수만 저장되는지 확인.

## CP4-13 q/-q invariance

raw ST sign-negated dataset에서 physical artifact와 metrics가 동일.

## CP4-14 No dense coercion

source/AST 검사로 float32 cast, zero-fill, `_SeqDataset`, `_predict_batches`,
legacy `predict(y_seq)` 경로를 사용하지 않음을 확인.

## CP4-15 Training disabled

runner가 fit/train/load checkpoint를 호출하지 않음.

## CP4-16 Failure cleanup

중간 실패 시 partial artifact를 valid cache/result로 인식하지 않음.

## CP4-17 Legacy isolation

새 branch가 아닌 기존 task/model smoke의 dispatch와 결과가 유지됨.

## CP4-18 Artifact provenance

모든 required identity/config/version/path/counter 필드가 기록됨.

---

# 16. Smoke YAML

`bench/configs/suite_phase1a_unit_st_smoke.yaml`은 최소 두 scenario를 포함하라.

```text
synthetic UNIT-ST
Basilisk UNIT-ST
```

CPU, 짧은 duration, 소수 trajectory를 사용한다.

각 scenario는:

```text
task_family = mekf_unit_st_v1
model_id = mekf_event_replay_v1
producer ID
seed
rates
noise/bias
split
explicit initial state/P/Qc
artifact/cache root
```

를 명시해야 한다.

flight-grade 수치라고 표현하지 말고 representative Tier-0 smoke임을 metadata에 기록하라.

---

# 17. Baseline regression

명시적 interpreter:

```text
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python
```

환경:

```text
PYTHONDONTWRITEBYTECODE=1
pytest -p no:cacheprovider
```

구현 전후 다음을 실행하라.

```text
Gate A tests
Gate B1 tests
Gate B2 tests
Gate C tests
D1 adapter tests
legacy regression
```

구현 후 추가:

```text
tests/test_mekf_runner_integration.py
실제 smoke YAML CLI run
fresh run
verified cache-hit run
```

전체 repository test가 현실적으로 가능한 경우 실행하되,
대규모 unrelated 실패를 기존 코드 수정으로 해결하지 마라.

baseline failure 시:

```text
BLOCKED_BASELINE_REGRESSION
```

---

# 18. Property sweep

최소:

```text
synthetic: 3 seeds
Basilisk: 3 seeds
```

각각 fresh + cache hit에서:

```text
direct/bridge/runner equality
dataset identity equality
artifact round trip
metric equality
q/-q invariance
split isolation
P/S SPD
```

를 확인하라.

---

# 19. 문서 산출물

## P1A_CP4_INTEGRATION_CONTRACT.md

포함:

```text
목적
ID
task dispatch
registry kind
runner branch 위치
typed sidecar/cache contract
initial state/Qc config
artifact schema
truth join
canonical metrics
same-realization
failure/partial artifact policy
legacy isolation
Phase 1B handoff
```

## P1A_CP4_TEST_MATRIX.md

각 test의 input/expected/tolerance/actual/evidence/status.

## P1A_CP4_VALIDATION_REPORT.md

포함:

```text
최종 판정
변경 파일
fresh/cache-hit synthetic/Basilisk
direct/bridge/runner equality
artifact/metric evidence
stale cache rejection
truth boundary
legacy regression
dirty-tree integrity
remaining Phase 1B work
```

---

# 20. 완료 판정

정상 완료 시:

```text
Status: PASS_P1A_CP4_INTEGRATION

Task dispatch: PASS
Registry append-only integration: PASS
Typed sidecar delivery: PASS
Verified cache identity: PASS
Fresh synthetic runner: PASS
Fresh Basilisk runner: PASS
Verified cache-hit replay: PASS
Stale cache rejection: PASS
Direct/bridge/runner exact equivalence: PASS
Same-realization preservation: PASS
Truth-free estimator boundary: PASS
Exact truth join: PASS
Lossless q/b/P artifact: PASS
Compact ST r/S artifact: PASS
Canonical Gate C metrics: PASS
q/-q invariance: PASS
No dense coercion: PASS
Training disabled: PASS
Failure/partial-artifact safety: PASS
Legacy isolation/regression: PASS
Gate A/B1/B2/C/D1 regressions: PASS
Dirty-tree integrity: PASS

P1A-CP4 Integration: GO
Phase 1A foundation: COMPLETE
Next authorized stage: Phase 1B classical MEKF benchmark completion
```

실패 시 최초 실패 계약, 반례, 변경 파일, frozen-file 변화, dirty integrity,
필요 scope extension을 보고하라.

Phase 1B, neural model 또는 visualization으로 자동 진행하지 마라.
