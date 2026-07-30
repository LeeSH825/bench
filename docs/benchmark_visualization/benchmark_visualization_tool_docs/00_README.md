# Benchmark Execution Visualization Tool — 문서 묶음 안내

작성 기준일: 2026-07-30  
기준 저장소 감사 문서: `docs/benchmark_gui_current_state_audit.md`  
감사 기준 Git commit: `d1cc4b035597bb029e0bce95a546b29b4664b5c6`

## 1. 이 문서 묶음의 목적

이 문서 묶음은 현재의 post-hoc 결과 시각화 도구를 넘어, 다음 기능을 갖춘 **benchmark 실행·관측·제어 도구**를 안전하게 구현하기 위한 기준 문서이다.

- GUI에서 config preset을 선택하고 수정·검증한다.
- benchmark run을 별도 worker process로 시작한다.
- 실행 중 loss, validation metric, 진행률, 로그, GPU/CPU/RAM을 본다.
- 실행을 graceful stop 또는 force terminate한다.
- interrupt checkpoint를 만들고, 지원되는 모델은 exact resume한다.
- 완료·실패·중단 run을 다시 열고 비교한다.
- 기존 Run Inspector와 새 orchestration dashboard를 연결한다.
- third-party KalmanNet 계열 코드는 가능한 한 수정하지 않는다.

현재 감사 결과상 저장소는 post-hoc Streamlit Run Inspector와 재사용 가능한 adapter/runner 일부는 갖추고 있지만, exact resume, durable process lifecycle, typed config, event contract, run registry는 없다. 따라서 화면부터 만드는 대신 backend contract와 process boundary를 먼저 고정해야 한다.

## 2. 핵심 결정 요약

### 2.1 프런트엔드

- **1순위: Dash + FastAPI**
- **2순위: Panel**
- 기존 **Streamlit Run Inspector는 초기 단계에서 유지**하고, 새 dashboard에서 deep link로 연결한다.
- 장기적으로 UI 요구가 크게 복잡해지면 같은 FastAPI API 위에 React/Vue frontend를 추가하거나 교체할 수 있게 한다.

Dash를 선택한 이유는 다음과 같다.

- Python 중심으로 구현 가능하다.
- 복잡한 multi-page dashboard, form, table, Plotly chart에 적합하다.
- FastAPI backend와 한 서버에서 API 및 dashboard를 함께 운영할 수 있다.
- WebSocket callback과 async callback을 이용한 live update가 가능하다.
- 공식 테스트 fixture를 제공한다.
- 나중에 React frontend와 같은 API를 공유하기 쉽다.

### 2.2 실행 구조

- frontend process 안에서 training을 실행하지 않는다.
- **한 run당 독립 subprocess 및 process group**을 사용한다.
- UI는 action을 요청하고 상태를 읽을 뿐, run lifecycle의 소유자가 아니다.
- worker는 UI가 꺼져도 계속 실행되어야 한다.

### 2.3 영속화

- SQLite: run 현재 상태, identity, lineage, checkpoint/artifact index, action request, GPU lease의 source of truth
- per-run JSONL: metric, log, status, resource, checkpoint, artifact event의 append-only journal
- filesystem: checkpoint, predictions, config snapshot, logs, visualization artifact의 저장 위치

### 2.4 MVP 정책

- local-first, single-user
- 한 GPU당 trainable run 하나
- MVP certified model: `split_knet`, `kalmannet_tsp`
- model-based baseline: launch/evaluate 지원, learning resume는 해당 없음
- `adaptive_knet`, `maml_knet`, `me_split_knet_v0`: 처음에는 browse/read-only 또는 experimental
- exact resume 경계: **완료된 optimizer-update boundary**
- weight load는 `warm start`, exact resume와 절대 혼용하지 않는다.

## 3. 문서 목록과 읽는 순서

1. [`01_frontend_framework_evaluation.md`](01_frontend_framework_evaluation.md)  
   후보 프런트엔드 비교, 점수, 선택 근거, 대안과 재검토 조건

2. [`02_target_architecture_and_mvp_plan.md`](02_target_architecture_and_mvp_plan.md)  
   권장 전체 구조, ADR, 단계별 구현 계획, 첫 end-to-end milestone

3. [`03_backend_contracts_and_data_schemas.md`](03_backend_contracts_and_data_schemas.md)  
   RunSpec, run/event/checkpoint/artifact/resource schema, SQLite/API contract

4. [`04_ui_information_architecture_and_workflows.md`](04_ui_information_architecture_and_workflows.md)  
   화면 구조, 사용자 workflow, 버튼 의미, live update, 오류 및 안전 UX

5. [`05_do_not_do_risks_and_migration.md`](05_do_not_do_risks_and_migration.md)  
   금지사항, anti-pattern, 위험 목록, 기존 Streamlit에서의 점진적 이행

6. [`06_acceptance_tests_and_definition_of_done.md`](06_acceptance_tests_and_definition_of_done.md)  
   단계별 acceptance test, failure injection, exact-resume 인증, Definition of Done

7. [`07_implementation_prompt.md`](07_implementation_prompt.md)  
   CLI coding agent에 전달할 구현 프롬프트

## 4. 구현 순서

```text
Baseline 확보
  ↓
Identity / Typed RunSpec / Immutable run allocation
  ↓
SQLite registry / JSONL event / worker subprocess / heartbeat
  ↓
Read-only Dash dashboard
  ↓
Config validation / clone / launch
  ↓
Checkpoint v1 / graceful stop / exact resume 인증
  ↓
기존 Run Inspector 통합 또는 선택적 포팅
  ↓
Queue / multi-GPU / remote / multi-user
```

화면 구현이 backend contract보다 앞서면 안 된다. 특히 Stop 버튼, Resume 버튼, live progress bar는 의미가 정확히 정의된 lifecycle과 checkpoint가 생긴 뒤 활성화해야 한다.

## 5. 구현 전 선결 확인

- 현재 dirty working tree와 dirty submodule 상태를 별도 branch 또는 patch로 보존한다.
- `pytest`를 설치하고 전체 baseline test 결과를 기록한다.
- 대표 run artifact를 golden fixture로 보존한다.
- 현재 adapter의 실행 가능 여부와 논문 충실도는 별도 필드로 관리한다.
- `split_knet`의 현재 adapter가 논문식 alternating optimization과 다른지 별도 fidelity audit를 유지한다.

## 6. 산출물 관리 원칙

새 문서를 저장소에 반영할 때 권장 경로는 다음과 같다.

```text
docs/benchmark_visualization/
  00_README.md
  01_frontend_framework_evaluation.md
  02_target_architecture_and_mvp_plan.md
  03_backend_contracts_and_data_schemas.md
  04_ui_information_architecture_and_workflows.md
  05_do_not_do_risks_and_migration.md
  06_acceptance_tests_and_definition_of_done.md
  07_implementation_prompt.md
```

모든 문서는 versioned schema와 ADR 번호를 사용한다. 구현 중 결정이 바뀌면 기존 결정을 소급 수정하기보다 superseded 상태와 대체 ADR을 남긴다.
