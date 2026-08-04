# CLI Coding Agent Prompt — Real Worker Exact-Resume Parity Micro-Tranche

당신은 AI-ADCS Neural Kalman Filter Benchmark 저장소의 **write-control backend를 실제 OS worker 수준에서 최종 인증하는 시니어 ML 플랫폼 엔지니어**이다.

이번 작업은 새로운 기능 계층을 넓히는 tranche가 아니다. 이전 continuation에서 이미 구현된 구조를 그대로 사용하여, 남아 있는 단 하나의 release gate를 실제 subprocess E2E로 닫는 **좁은 검증·수정 micro-tranche**이다.

```text
[이미 완료 — 재구현 금지]
control_resumable_v1 선택 및 no-silent-fallback
training_path_id 영속화
registry migration 3
Checkpoint v2 및 launch eligibility
immutable resume child allocation
WorkerManager child launch 경로
durable RESUME_EXACT action
idempotency / optimistic concurrency / restart reconciliation
CLI plan-only resume + explicit --launch

[이번 micro-tranche]
KNet real WorkerManager stop→resume-child bitwise parity
Split real WorkerManager stop→resume-child bitwise parity
그 E2E에 의존하는 두 child-failure fault case

[다음 tranche로 계속 미룸]
HTTP POST write API
Dash Stop/Resume controls
write-mode switch
Playwright write workflow
```

최종 목표는 다음을 **실제 SQLite, 실제 checkpoint package, 실제 WorkerManager, 실제 별도 OS process, 실제 third-party KNet/Split 코드**로 증명하는 것이다.

```text
continuous reference worker
vs.
parent worker → graceful stop → Checkpoint v2
→ durable RESUME_EXACT action
→ immutable child worker in a fresh interpreter
→ remaining training → COMPLETED
```

두 경로가 인증된 비교 항목에서 bitwise identical이어야 한다.

────────────────────────────────────
0. 현재 상태와 기준선 확인
────────────────────────────────────

사용자가 전달한 최신 결과는 다음과 같다. 이를 주장으로만 취급하고 Git과 코드에서 다시 확인하라.

```text
authoritative repository:
  /home/dss-pc-05/bench
  authoritative tree는 untouched

continuation branch:
  benchmark-viz/write-control

expected isolated worktree:
  /tmp/bench-wc-tranche

partial implementation ancestry:
  48b0edb
  dfb932d

latest continuation commits:
  11a451d  feat — training path/checkpoint/registry propagation
  de79d2a  feat — child coordinator + durable resume action
  a3ceff1  test — continuation tests
  9e4fd8   docs — continuation contracts/report

reported latest baseline:
  532 collected
  531 passed
  1 skipped
  0 failed
  init-provenance 28 passed
  API GET-only
  Dash action buttons 0
  third-party tracked diff empty

reported verdict:
  READY_AFTER_SPECIFIC_FIXES

reported remaining blocker:
  real-subprocess KNet/Split stop→resume-child bitwise parity E2E
  + that E2E에 의존하는 두 child-failure fault cases
```

먼저 실행하라.

```bash
pwd
git rev-parse --show-toplevel
git worktree list
git branch --all --verbose --no-abbrev
git log --oneline --decorate --graph -40 benchmark-viz/write-control
git show --stat --oneline 11a451d
git show --stat --oneline de79d2a
git show --stat --oneline a3ceff1
git show --stat --oneline 9e4fd8
git merge-base --is-ancestor c0eaaf7 benchmark-viz/write-control
git merge-base --is-ancestor 48b0edb benchmark-viz/write-control
git status --short
git submodule status --recursive
```

원칙:

- 위 hash가 실제와 다르면 branch의 실제 latest committed tip과 의미상 대응되는 commit을 식별하여 기록한다.
- `benchmark-viz/write-control`의 최신 committed tip에서 이어간다.
- `/tmp/bench-wc-tranche`가 안전하면 재사용하고, 없으면 해당 tip에서 새 isolated worktree를 만든다.
- `48b0edb` 이후 기능을 처음부터 다시 구현하지 않는다.
- authoritative dirty tree에서 직접 수정하지 않는다.
- 구조가 실제로 없거나 branch ancestry가 끊겼다면 즉시 다음 verdict로 중단한다.

```text
WRONG_CHECKOUT_OR_LOST_CONTINUATION_IMPLEMENTATION
```

────────────────────────────────────
1. 반드시 읽을 문서
────────────────────────────────────

먼저 이전 backend의 실제 상태를 읽는다.

```text
docs/benchmark_visualization/checkpoint_stop_resume_tranche_report.md
docs/benchmark_visualization/checkpoint_stop_resume_tranche_summary.json
docs/benchmark_visualization/checkpoint_resume_state_audit.md
docs/benchmark_visualization/checkpoint_v1_schema.md
docs/benchmark_visualization/checkpoint_v2_schema.md
docs/benchmark_visualization/exact_resume_certification_matrix.md
docs/benchmark_visualization/graceful_stop_operator_guide.md

docs/benchmark_visualization/control_plane_training_path_contract.md
docs/benchmark_visualization/resume_child_worker_contract.md
docs/benchmark_visualization/durable_resume_action_contract.md
docs/benchmark_visualization/write_control_backend_continuation_report.md
docs/benchmark_visualization/write_control_backend_continuation_summary.json
```

그다음 normative design 문서를 읽는다.

```text
docs/benchmark_visualization/benchmark_visualization_tool_docs/
  13_write_control_tranche_plan.md
  14_write_control_constraints_and_decisions.md
  15_write_control_acceptance_and_test_plan.md
  16_write_control_implementation_prompt.md
  17_write_control_backend_continuation_prompt.md
  18_write_control_real_worker_parity_prompt.md
```

우선순위:

```text
latest executable code
→ latest continuation report/summary/contracts
→ this prompt
→ 14 binding decisions
→ 15 acceptance semantics
→ 13/16/17 older tranche prompts
```

`write_control_backend_continuation_report.md`에서 **아직 미검증으로 명시된 정확한 두 child-failure fault case와 기대 상태**를 찾아 preflight report에 그대로 옮겨라. 임의의 다른 fault case로 대체하지 마라.

해당 보고서가 두 case를 충분히 명확히 정의하지 않았다면:

1. coordinator/action/worker contracts와 기존 tests를 조사하고,
2. ambiguity를 먼저 문서화하며,
3. 최소한 아래 두 범주를 포함하는 명시적 test contract를 작성한 후 구현한다.

```text
A. child process가 시작된 뒤 첫 optimizer update 전에 restore/protocol 실패
B. restore 성공 후 실제 child training 중 ordinary failure 또는 보고서가 지정한 post-launch failure
```

보고서가 다른 정확한 두 case를 지정하면 보고서 정의가 우선한다.

────────────────────────────────────
2. 작업 보호
────────────────────────────────────

- authoritative repository의 unrelated dirty work를 그대로 보존한다.
- isolated worktree에서만 작업한다.
- `git add -A`, `git commit -am`, `git clean`, `git reset --hard`, broad checkout/revert 금지.
- staging은 명시적 path만 사용한다.
- `runs/`, `reports/`, shared dataset, 기존 production registry/checkpoint를 test output으로 사용하지 않는다.
- 모든 새 E2E는 `tmp_path` 또는 timestamped temporary `BENCH_CONTROL_ROOT`를 사용한다.
- third-party tracked source를 수정하지 않는다.
- remote push를 하지 않는다.
- verifier가 시작하지 않은 PID/process group에는 signal을 보내지 않는다.

preflight evidence:

```text
artifacts/benchmark_write_control_real_worker/<UTC_TIMESTAMP>/preflight/
```

최소 포함:

- branch/HEAD/worktree/status
- commit ancestry
- submodule revisions/diff
- Python/Torch/device 정보
- registry/checkpoint/config schema versions
- current certification rows
- current full pytest result
- continuation report에서 추출한 남은 fault cases

artifact는 commit하지 않는다.

────────────────────────────────────
3. Baseline gate
────────────────────────────────────

변경 전에 다음을 실행한다.

```bash
python -m pytest --collect-only -q
python -m pytest -q
python -m pytest -q tests/test_viz_init_provenance_comparison.py
```

그리고 repository에서 실제 파일명을 찾아 다음 targeted 영역도 실행한다.

```text
training path selection
checkpoint v1/v2
exact resume certification
graceful stop
resume child coordinator
durable action/idempotency/reconciliation
read-only API/Dash regression
```

기대 baseline은 대략 다음이다.

```text
532 collected
531 passed
1 skipped
0 failed
28 init-provenance passed
```

숫자가 달라도 test 삭제 여부와 branch tip 차이를 조사한다. 기존 test를 ignore, xfail, unconditional skip 또는 삭제하여 baseline을 맞추지 마라.

baseline이 깨져 있으면 원인을 분류하고, 해결되지 않은 상태에서 E2E를 추가하지 마라.

────────────────────────────────────
4. 이번 작업의 엄격한 범위
────────────────────────────────────

이번 micro-tranche는 **먼저 현재 production code를 수정하지 않고 real-worker E2E를 실행**해야 한다.

순서:

```text
1. 현재 committed code로 E2E harness 작성/실행
2. 통과하면 production change 없이 tests/docs만 추가
3. 실패하면 first divergence를 국소화
4. 필요한 최소 production fix만 수행
5. 동일 E2E와 full regression 재실행
```

다음을 재설계하지 마라.

- training path selection
- Checkpoint v2 format
- registry migration 3
- resume action semantics
- child allocation/lineage
- idempotency model
- WorkerManager architecture
- exact-resume certified envelope

기존 구현을 우회하는 별도 worker, 별도 DB, 별도 checkpoint format, test-only in-process resume path를 만들지 마라.

────────────────────────────────────
5. Real Worker E2E의 정의
────────────────────────────────────

E2E는 다음 조건을 모두 만족해야 한다.

### 5.1 실제 production process 경로

- parent와 child 모두 `WorkerManager`를 통해 launch한다.
- 둘 다 별도 OS process이며 fresh Python interpreter이다.
- 실제 `bench.control.process.worker_cli`와 production executor/runner 경로를 사용한다.
- `SyntheticExecutor`, direct adapter call, in-process coordinator-only call, mocked `Popen`만으로 승인하지 않는다.
- KNet/Split의 실제 adapter와 실제 pinned third-party module을 load한다.

증명할 항목:

```text
parent PID / PGID / process start time / worker token
child PID / PGID / process start time / worker token
parent PID != child PID
child PID != test/API/coordinator PID
/proc cmdline 또는 psutil cmdline에 worker entry와 child run_id 존재
stdout/stderr가 child run directory에 기록
정상 exit code와 terminal state 일치
```

POSIX `/proc`가 없으면 equivalent verified process identity를 사용하고 근거를 기록한다.

### 5.2 인증 envelope

두 모델 모두 다음 exact tuple만 사용한다.

```text
device_class = cpu
precision = fp32
num_workers = 0
resume_boundary = optimizer_update
training_path_id = control_resumable_v1
checkpoint_schema_version = 2
```

추가 deterministic 설정:

- deterministic algorithms 활성화
- intra/inter-op thread 수 고정
- 동일 dataset fingerprint
- 동일 structural config hash
- 동일 implementation_id
- 동일 variant_id
- 동일 total optimizer-update target
- shared production run/report tree를 사용하지 않음

GPU, AMP, multi-worker, Adaptive/MAML/ME-Split은 이번 gate에 포함하지 않는다.

### 5.3 Real tiny suite fixture

- KNet과 Split 각각 실제 suite/adapter가 실행되는 self-contained tiny fixture를 만든다.
- dataset은 test-controlled temporary location에 생성·고정한다.
- fixture는 model setup, optimizer step, validation, checkpoint, final evaluation을 실제로 수행해야 한다.
- 모델을 대체하는 fake network를 사용하지 않는다.
- runtime은 bounded해야 하지만 너무 짧아 Stop request 전에 완료되는 flaky fixture를 만들지 않는다.

Stop 시점을 hard-code하지 마라. 외부 test controller가 worker의 `RUNNING`과 completed update/event를 관찰한 뒤 persistent graceful-stop action을 요청한다. 실제 interrupt checkpoint cursor를 `K_actual`로 사용한다.

필요하면 workload update 수를 늘리되, production training loop에 user-facing sleep 옵션을 추가하지 마라. 기존 observer/barrier seam을 이용한 test-only synchronization이 필요하면:

- public RunSpec/config option으로 노출하지 말고,
- numerical path를 변경하지 않으며,
- test에서만 활성화되고,
- 별도 test가 비활성 시 완전히 inert함을 증명해야 한다.

가급적 registry/event polling으로 해결한다.

────────────────────────────────────
6. KNet real-worker parity
────────────────────────────────────

### 6.1 Continuous reference A

```text
new immutable run A
training_path_id = control_resumable_v1
WorkerManager launch
N completed optimizer updates
normal final evaluation/report
COMPLETED
```

### 6.2 Stop/resume lineage B

```text
new immutable parent B
same structural experiment as A
WorkerManager launch
RUNNING 확인
persistent graceful Stop request
safe boundary에서 K_actual updates 완료
STOP_REQUESTED → CHECKPOINTING → INTERRUPTED
validated, launch-eligible Checkpoint v2

durable RESUME_EXACT action
immutable child C allocation
WorkerManager child launch
fresh process restore
K_actual 다음 update부터 N까지 계속
normal final evaluation/report
child COMPLETED
```

### 6.3 비교

A의 전체 결과와 `B prefix + C suffix`를 다음에서 bitwise 비교한다.

```text
final model state_dict tensor bytes
Adam optimizer state
full per-update train-loss sequence
full validation history
global update count
BatchPlan id and final cursor
best model state
best step
best metric
final prediction tensor/array
final benchmark metrics
structural_config_hash
dataset_fingerprint
variant_id
training_path_id
implementation_id
```

시간, PID, run_id, directory, event timestamp, telemetry sample 등 operational 차이만 allow-list한다. 허용 목록을 보고서에 명시한다.

Loss/event sequence 요구:

- parent steps `1..K_actual`
- child steps `K_actual+1..N`
- duplicate 없음
- gap 없음
- child에서 step이 1로 reset되지 않음
- validation cadence가 continuous reference와 동일

Parent immutability는 `resume_child_worker_contract.md`의 정확한 정의를 따른다. Resume action row와 child lineage처럼 의도적으로 추가되는 관계는 parent artifact mutation으로 오판하지 마라.

────────────────────────────────────
7. Split real-worker parity
────────────────────────────────────

KNet과 동일한 WorkerManager E2E를 수행한다.

추가 필수 조건:

- child adapter construction은 parent와 다른 pre-restore RNG/seed 상태에서 이루어져야 한다.
- 이 검증을 위해 기존 fresh-process exact-resume harness의 test seam을 재사용한다.
- public/user-facing seed override를 추가하지 않는다.
- resolved structural RunSpec의 experiment seed를 바꾸어 다른 experiment를 만드는 방식으로 속이지 않는다.
- checkpoint restore가 Split의 `hn1_init` / `hn2_init` extra state를 실제로 덮어써야 한다.

증명:

```text
pre-restore child construction state != parent construction state
post-restore required Split extra state == checkpoint state
첫 post-resume update부터 continuous reference와 bitwise 동일
```

Mutation probe:

- v2 payload 또는 inventory에서 required Split extra state를 제거한 변형은 restore 단계에서 거부되어야 한다.
- guard를 우회한 diagnostic mutation에서는 first post-resume divergence가 검출되어야 한다.
- production third-party file을 수정하지 않는다.

────────────────────────────────────
8. First-divergence diagnosis
────────────────────────────────────

Parity가 실패하면 tolerance를 넓히거나 `allclose`로 바꾸지 마라.

다음 순서로 first divergence를 국소화한다.

```text
1. parent interrupt manifest/registry/spec consistency
2. child pre-restore construction hashes
3. child post-restore model hash
4. optimizer state hash
5. RNG state hashes: Python / NumPy / torch CPU
6. BatchPlan id/cursor와 next batch indices
7. best/validation state
8. Split conditional extra state
9. first child forward output
10. first child loss
11. first child gradient hash
12. first child optimizer-step result
13. subsequent update sequence
```

진단 artifact에 민감한 tensor 전체를 무제한 dump하지 말고, shape/dtype/hash와 bounded sample을 기록한다.

차이의 원인이 current implementation bug이면 최소 fix를 수행한다. 인증 범위를 넓히거나 legacy path로 fallback하지 않는다.

bitwise parity를 확보하지 못하면 API/UI 단계로 넘어가지 말고 정확한 blocker를 남긴다.

────────────────────────────────────
9. 두 child-failure fault case
────────────────────────────────────

`write_control_backend_continuation_report.md`가 지정한 정확한 두 case를 실제 WorkerManager child process로 실행한다.

공통 검증:

```text
action state
child run state/state_version
worker row/state/PID/exit code
failure event와 traceback
parent state/artifact/checkpoint 불변
duplicate child/worker 없음
reconcile 재실행 idempotent
same idempotency request 재시도 결과
```

상태 의미 원칙:

- launch 전 한 번도 실행되지 않은 child는 기존 contract가 정한 `CANCELLED` semantics를 유지한다.
- worker가 실제 시작된 뒤 restore/protocol에서 실패하면 실행된 child의 실제 failure semantics를 사용한다.
- Resume action이 worker launch 성공 시 이미 `COMPLETED`가 되는 contract라면, child의 이후 학습 실패 때문에 action을 소급 `FAILED`로 바꾸지 않는다.
- child training outcome은 child run state가 source of truth이다.
- SIGKILL case가 포함되면 PID/start-time/token을 검증한 verifier-owned process에만 signal을 보내고, expected result는 false `COMPLETED`가 아닌 `ORPHANED`이다.

보고서의 기존 contract와 다른 임의 state transition을 추가하지 마라.

────────────────────────────────────
10. Idempotency와 restart regression
────────────────────────────────────

이번 micro-tranche는 기존 구조를 다시 최소한 다음으로 확인한다.

```text
same Resume request 5회
→ action 1
→ child 1
→ actual worker process 1

same key + different checkpoint
→ conflict

stale parent state_version
→ no side effect

coordinator process restart while child RUNNING
→ child continues
→ same child/worker recovered
→ second launch 없음

reconcile-actions 두 번
→ 두 번째 no-op
```

API는 실행하지 않아도 된다. API가 없는 상태에서 worker와 action lifecycle이 완료되어야 한다.

────────────────────────────────────
11. 이번 micro-tranche에서 금지되는 구현
────────────────────────────────────

다음을 구현하지 마라.

- POST write API
- `BENCH_CONTROL_ENABLE_WRITES` public switch
- Dash Stop safely button
- Dash Resume training button
- Playwright write workflow
- config GUI launch
- force terminate API/UI
- warm-start API/UI
- GPU queue/scheduler
- GPU/AMP/multi-worker exact resume
- Adaptive/MAML/ME-Split exact resume
- authentication/multi-user
- WebSocket/SSE
- third-party source patch
- Checkpoint v3 또는 registry migration 4 — real E2E가 실제로 이를 요구하고 기존 additive contract로 해결할 수 없다는 증거가 없는 한 금지

기존 FastAPI는 계속 GET/HEAD-only여야 한다. Dash action button 수는 계속 0개여야 한다.

────────────────────────────────────
12. Mandatory tests and execution evidence
────────────────────────────────────

새 test는 mock-only로 끝내지 않는다. 최소 test groups:

```text
real_worker_knet_continuous_vs_resume
real_worker_split_continuous_vs_resume
real_worker_resume_child_failure_cases
real_worker_resume_idempotency_restart
```

실제 파일명은 repository convention에 맞춘다.

각 real-worker test는 다음을 가져야 한다.

- bounded timeout
- verifier-owned process cleanup in `finally`
- PID identity verification before any signal
- isolated control root
- stdout/stderr/evidence retention on failure
- no production `runs/`/`reports/` write
- no hidden skip on slow/GPU/platform condition when CPU/POSIX requirements are present

최종적으로 제외 옵션 없이 실행한다.

```bash
python -m pytest --collect-only -q
python -m pytest -q
python -m pytest -q tests/test_viz_init_provenance_comparison.py
```

별도 기록:

- KNet real worker test
- Split real worker test
- child fault tests
- idempotency/restart tests
- checkpoint v1/v2 tests
- graceful-stop tests
- direct adapter/service exact-resume tests
- observer/telemetry parity
- normal/failure/orphan lifecycle tests
- read-only API method audit
- Dash action-buttons-0 audit
- Streamlit Inspector import/load
- third-party tracked diff

기존 test를 삭제, ignore, unconditional skip, xfail하여 green으로 만들지 않는다.

────────────────────────────────────
13. Certification과 documentation update
────────────────────────────────────

Real-worker parity가 통과하기 전에는 worker-level certification을 완료로 기록하지 마라.

통과 후 다음을 갱신한다.

### 새 문서

```text
docs/benchmark_visualization/
  worker_level_exact_resume_certification.md
  write_control_real_worker_parity_report.md
  write_control_real_worker_parity_summary.json
```

### 실제 동작에 맞게 갱신

```text
docs/benchmark_visualization/resume_child_worker_contract.md
docs/benchmark_visualization/durable_resume_action_contract.md
docs/benchmark_visualization/exact_resume_certification_matrix.md
docs/benchmark_visualization/known_limitations.md
docs/benchmark_visualization/implementation_status_phase0_phase1_phase3.md
```

과거 tranche report와 summary는 historical evidence이므로 덮어쓰지 않는다.

```text
checkpoint_stop_resume_tranche_report.md
write_control_tranche_report.md
write_control_backend_continuation_report.md
각 기존 summary JSON
```

필요하면 상단에 successor pointer만 추가하되 과거 verdict를 소급 변경하지 않는다.

### Certification 표현

다음을 구분한다.

```text
adapter/service fresh-process parity
WorkerManager real-subprocess parity
public API/UI exposure
```

이번 micro-tranche가 성공하면 앞의 두 개가 인증되고 세 번째는 아직 미구현이다.

Machine-readable summary에는 최소 다음을 포함한다.

```json
{
  "verdict": "READY_FOR_WRITE_API_UI_TRANCHE",
  "kalmannet_tsp_worker_parity": "BITWISE_PASS",
  "split_knet_worker_parity": "BITWISE_PASS",
  "child_process_verified": true,
  "checkpoint_schema_version": 2,
  "training_path_id": "control_resumable_v1",
  "failure_cases": {},
  "full_pytest": {},
  "api_methods": ["GET", "HEAD"],
  "dash_action_buttons": 0,
  "third_party_tracked_diff": false
}
```

실제 field naming은 repository convention에 맞춘다.

Evidence artifact:

```text
artifacts/benchmark_write_control_real_worker/<UTC_TIMESTAMP>/
```

최소 포함:

- environment/commit snapshot
- parent/child process identities
- state-transition histories
- action/lineage rows
- checkpoint manifests and validation summaries
- parity hashes and first-divergence diagnostics if any
- fault injection results
- pytest logs

checkpoint payload 전체를 불필요하게 복제하지 않는다. artifact는 commit하지 않는다.

────────────────────────────────────
14. Commit discipline
────────────────────────────────────

기존 latest continuation commit 이후에만 추가한다.

통과하고 production fix가 불필요하면:

```text
test: certify KNet/Split exact resume through real WorkerManager processes
docs: record worker-level parity and remaining API/UI scope
```

버그가 발견되어 fix가 필요하면:

```text
fix: <minimal real-worker resume issue>
test: add real WorkerManager parity and dependent fault gates
docs: record certification evidence and limitation
```

명시적 path만 stage한다.

commit 전:

```bash
git diff --cached --check
git status --short
git diff --submodule=log -- third_party
```

verification artifacts, temp DB/WAL/SHM, temp runs, logs, screenshots, cache를 commit하지 않는다.

push하지 않는다.

────────────────────────────────────
15. Final gate
────────────────────────────────────

`READY_FOR_WRITE_API_UI_TRANCHE`는 다음을 모두 만족할 때만 가능하다.

- current continuation implementation is preserved
- KNet parent/child are actual WorkerManager OS processes
- KNet continuous vs lineage result is bitwise identical
- Split parent/child are actual WorkerManager OS processes
- Split continuous vs lineage result is bitwise identical
- Split different-construction-state and required extra-state checks pass
- parent artifacts/checkpoints remain immutable after resume
- child lineage is complete
- loss/validation/global-step sequences have no reset, duplicate, or gap
- the exact two dependent child-failure cases pass with correct states
- idempotency creates one action/child/worker under real process launch
- coordinator restart does not launch a second child
- full pytest passes without exclusions
- 28 init-provenance tests pass
- existing direct exact-resume and graceful-stop tests pass
- API remains GET/HEAD-only
- Dash remains zero-action-button
- third-party tracked diff is empty
- implementation/tests/docs are reproducible from local commits in a clean worktree

최종 verdict는 다음 중 하나이다.

```text
READY_FOR_WRITE_API_UI_TRANCHE
READY_AFTER_SPECIFIC_FIXES
NOT_READY
WRONG_CHECKOUT_OR_LOST_CONTINUATION_IMPLEMENTATION
```

`READY_FOR_WRITE_API_UI_TRANCHE`는 POST API나 Dash controls가 이미 구현됐다는 뜻이 아니다. 그 계층을 추가하기 위한 real-worker numerical gate가 닫혔다는 뜻이다.

────────────────────────────────────
16. 완료 시 터미널 요약
────────────────────────────────────

완료 시 다음만 구조적으로 출력한다.

1. authoritative repository와 isolated worktree
2. continuation baseline branch/commit
3. 기존 11a451d/de79d2a/a3ceff1/9e4fd8 확인 결과
4. 새 fix/test/docs commit hashes
5. KNet actual parent/child process identity
6. KNet bitwise parity 결과
7. Split actual parent/child process identity
8. Split bitwise parity 및 extra-state 결과
9. 두 child-failure fault case 결과
10. idempotency/restart 결과
11. full pytest 결과
12. 28 init-provenance 결과
13. API GET-only / Dash 0-button 결과
14. third-party tracked diff
15. 남은 API/UI scope
16. 최종 verdict
17. 생성한 report/summary/evidence 경로

이제 위 문서를 모두 읽고, `benchmark-viz/write-control` 최신 tip에서 **real WorkerManager numerical parity micro-tranche만 완료하라.**
