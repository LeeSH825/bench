# Benchmark Execution Visualization Tool — Frontend Framework Evaluation

작성 기준일: 2026-07-30  
평가 대상: Streamlit, Dash, Panel, NiceGUI, Shiny for Python, Taipy, React + FastAPI, PySide6, Gradio

## 1. 평가 목적

이 프로젝트에 필요한 것은 단순 결과 dashboard가 아니라 다음을 동시에 다루는 experiment control surface이다.

- typed config form과 raw YAML editor
- run 목록, 상태, lineage, checkpoint catalog
- live metric, log, GPU/CPU/RAM telemetry
- graceful stop, force terminate, resume, warm start
- 여러 model/init/implementation variant 비교
- local workstation과 SSH tunnel 기반 remote 접근
- 장시간 실행과 UI 재시작에 대한 내구성
- 기존 Python/Plotly/Streamlit visualization 자산 재사용

프런트엔드는 run lifecycle의 source of truth가 될 수 없다. 어떤 프레임워크를 선택하더라도 worker subprocess, SQLite registry, JSONL event journal, atomic checkpoint는 별도 backend 계층으로 유지한다.

## 2. 현재 코드베이스가 주는 제약

감사 보고서에서 확인된 현재 상태는 다음과 같다.

- 기존 Streamlit 앱은 completed artifact 중심의 post-hoc Run Inspector이다.
- `run_one()`은 재사용 가능하지만 filesystem, logging, adapter lifecycle이 결합되어 있다.
- training은 synchronous in-process 방식이다.
- durable PID/heartbeat/status state machine이 없다.
- live event contract와 resource telemetry가 없다.
- checkpoint는 weight/summary 중심이며 exact resume가 아니다.
- config는 YAML dict와 ad-hoc access이며 UI schema가 없다.
- run identity는 deterministic path에 의존하고 overwrite 가능성이 있다.

따라서 프런트엔드 선택의 핵심은 “training을 프런트엔드 callback에서 돌릴 수 있는가”가 아니라, **외부 orchestration service를 명확하게 호출하고 live state를 안정적으로 표시할 수 있는가**이다.

## 3. 평가 기준

| 기준 | 가중치 | 의미 |
|---|---:|---|
| Backend/process 분리 적합성 | 15 | 외부 API, worker, DB와 자연스럽게 연결 가능한가 |
| Live update | 15 | polling, async, WebSocket, incremental update가 가능한가 |
| 복잡한 form/routing | 12 | multi-page, 조건부 form, validation UX가 가능한가 |
| Chart/table/log 표현력 | 10 | Plotly, 대형 table, log tail, status card 구현 편의성 |
| 테스트 가능성 | 10 | unit/UI/E2E 공식 지원과 자동화 용이성 |
| Python 자산 재사용 | 10 | 기존 Python loader, Plotly figure, adapter metadata 활용성 |
| Remote/headless | 10 | Linux workstation, SSH tunnel, reverse proxy 적합성 |
| 장기 확장성 | 10 | API 분리, multi-user, frontend 교체 경로 |
| MVP 구현 비용 | 8 | 현재 팀이 빠르게 안전한 MVP를 만들 수 있는가 |

점수는 이 프로젝트에 대한 1~5의 상대 평가이며, 범용 프레임워크 순위가 아니다.

## 4. 프로젝트별 평가 점수

| 후보 | 분리 | Live | Form | Viz | Test | Python | Remote | 확장 | 비용 | 가중 점수 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Dash + FastAPI** | 5 | 5 | 5 | 5 | 5 | 4 | 5 | 5 | 4 | **96.4** |
| **Panel** | 4 | 5 | 4 | 5 | 4 | 5 | 5 | 4 | 4 | **89.0** |
| React + FastAPI | 5 | 5 | 5 | 5 | 5 | 2 | 5 | 5 | 1 | 87.6 |
| Shiny for Python | 4 | 4 | 4 | 4 | 4 | 5 | 5 | 4 | 3 | 82.4 |
| NiceGUI | 4 | 4 | 4 | 4 | 4 | 5 | 4 | 3 | 5 | 81.6 |
| Streamlit | 3 | 4 | 3 | 4 | 4 | 5 | 5 | 3 | 5 | 78.2 |
| PySide6 | 5 | 4 | 4 | 4 | 4 | 4 | 1 | 3 | 2 | 71.8 |
| Taipy GUI/Core | 3 | 4 | 4 | 4 | 3 | 4 | 4 | 3 | 3 | 71.4 |
| Gradio | 2 | 3 | 2 | 3 | 3 | 5 | 4 | 2 | 5 | 61.8 |

## 5. 후보별 분석

### 5.1 Dash + FastAPI — 권장

#### 장점

- Dash는 Plotly 중심의 dashboard에 적합하고 multi-page routing과 명시적 callback graph를 제공한다.
- 현재 공식 문서 기준 Dash는 FastAPI backend를 지원한다.
- FastAPI backend에서 async endpoint와 async callback을 같이 둘 수 있다.
- Dash 4.2 이상은 WebSocket callback을 제공하며 live stream, progress, log display를 공식 use case로 제시한다.
- `dash.testing`은 pytest 기반 unit/E2E fixture를 제공한다.
- 동일 FastAPI server에서 `/dashboard`, `/api`, 향후 React 정적 frontend를 함께 제공할 수 있다.
- 기존 Plotly figure를 재사용하기 쉽다.

#### 주의점

- Dash background callback이나 WebSocket callback 자체를 benchmark worker로 사용하면 안 된다.
- live callback은 SQLite/JSONL을 읽어 UI에 전달하는 transport일 뿐이다.
- callback dependency가 커지면 복잡해질 수 있으므로 service/API layer를 UI callback에서 분리해야 한다.
- 최신 WebSocket 기능을 채택할 경우 Dash/FastAPI 버전을 lock하고 browser reconnect test를 추가해야 한다.

#### 이 프로젝트에서의 사용 방식

```text
Dash page/callback
  → FastAPI control service
  → SQLite registry / JSONL event
  → process manager
  → worker subprocess
```

MVP에서는 1~2초 polling으로 시작하고, backend contract가 안정된 뒤 log/event panel만 WebSocket으로 전환하는 방식을 권장한다.

### 5.2 Panel — 강력한 2순위

#### 장점

- Python/PyData 중심으로 dashboard와 복잡한 application을 만들 수 있다.
- async callback과 periodic callback을 공식 지원한다.
- Plotly, Bokeh, HoloViews, Tabulator, Perspective 등 데이터 시각화 구성요소가 강하다.
- large/real-time dataset visualization과 Python 객체 표현에 유리하다.
- multi-page app과 browser UI test가 가능하다.

#### 단점

- Bokeh/Tornado session model을 이해해야 한다.
- 프로젝트가 장기적으로 REST/WebSocket API와 별도 frontend로 확장될 경우 Dash+FastAPI보다 경계가 덜 직접적이다.
- 현재 기존 앱이 Streamlit이므로 Panel로도 사실상 UI 재작성은 필요하다.

#### 선택 조건

- 팀이 HoloViz/Bokeh 생태계를 선호한다.
- plotting/data exploration이 control form보다 훨씬 중요하다.
- 향후 React 분리 가능성이 낮다.

### 5.3 React + FastAPI — 장기 확장 옵션

#### 장점

- UX, routing, state management, large table, virtualized log, design system에서 가장 큰 자유도를 가진다.
- backend와 frontend를 명확히 분리할 수 있다.
- multi-user, auth, remote worker, complex workflow로 확장하기 좋다.

#### 단점

- JavaScript/TypeScript build pipeline과 별도 테스트 체계가 필요하다.
- 현재 Python 중심 팀과 코드베이스에서 MVP 비용이 가장 높다.
- 기존 visualization panel을 직접 재사용하기 어렵다.

#### 권장 사용 시점

다음 중 두 가지 이상이 실제 요구가 되면 재검토한다.

- 여러 사용자가 동시에 접속한다.
- remote worker cluster가 생긴다.
- 사용자별 권한과 audit log가 필요하다.
- 복잡한 drag-and-drop workflow builder가 필요하다.
- UI 품질과 custom interaction이 연구 속도보다 중요해진다.

### 5.4 NiceGUI — 빠른 local web/native prototype 후보

#### 장점

- Python만으로 browser UI를 만들기 쉽다.
- timer, binding, Plotly, table, log와 native mode를 제공한다.
- local desktop-like UX를 빠르게 만들 수 있다.
- 공식 pytest 기반 UI testing framework가 있다.

#### 단점

- 대규모 experiment management UI에 대한 검증 사례와 component ecosystem은 Dash/React보다 작다.
- application state와 backend registry를 혼용하기 쉬우므로 discipline이 필요하다.
- 장기적인 frontend 분리 경로가 Dash+FastAPI보다 덜 명시적이다.

#### 결론

짧은 UI spike에는 좋지만 본 프로젝트의 공식 control plane 1순위로는 선택하지 않는다.

### 5.5 Shiny for Python — reactive model이 강한 대안

#### 장점

- reactive graph와 scheduled invalidation이 명확하다.
- `ExtendedTask`로 긴 작업을 reactive graph 밖에서 비동기로 실행할 수 있다.
- data science dashboard에 강하고 remote web deployment에 적합하다.

#### 단점

- 현재 팀과 코드베이스에 Shiny 자산이 없다.
- exact-resume/process manager는 어차피 별도 구현해야 한다.
- Plotly/Dash 기반 구성보다 현재 visualization code 재사용 이점이 작다.

### 5.6 Taipy GUI/Core — 기능은 매력적이지만 구조 중복 위험

#### 장점

- scenario, task, job submission, status subscription, scenario comparison 같은 개념을 제공한다.
- GUI와 orchestration을 한 생태계에서 구성할 수 있다.

#### 단점

- 이미 존재하는 benchmark의 run/model/init/track/scenario 개념과 Taipy의 Scenario/Task/DataNode 모델이 충돌하거나 중복될 수 있다.
- exact resume, third-party adapter, benchmark-specific artifact contract를 Taipy 모델에 맞추는 migration이 필요할 수 있다.
- UI만 사용할 경우 Taipy Core의 장점이 줄어든다.

#### 결론

새 프로젝트라면 검토할 수 있지만, 현재 adapter 기반 benchmark를 보존해야 하는 본 프로젝트에는 도입 비용과 lock-in 위험이 크다.

### 5.7 Streamlit — 기존 Inspector 유지, 새 control plane에는 비권장

#### 장점

- 현재 Run Inspector가 이미 존재한다.
- Plotly, dataframe, form 개발 속도가 빠르다.
- fragment의 주기적 rerun으로 live panel 구현은 가능하다.

#### 한계

- interaction마다 script rerun이라는 execution model을 가진다.
- Session State는 browser tab/session에 종속되며 server crash 후 복구되지 않는다.
- process control, durable lifecycle, complex multi-page form을 잘못 구현하면 UI state와 run state가 섞이기 쉽다.

#### 결론

- 기존 post-hoc Inspector는 유지한다.
- 새 orchestration backend와 연결된 read-only page를 임시로 추가하는 것은 가능하다.
- 장기 실행의 authoritative control plane으로는 사용하지 않는다.

### 5.8 PySide6 — local desktop 전용 대안

#### 장점

- `QProcess`가 process 시작, stdout/stderr, exit code, error signal을 직접 지원한다.
- desktop UI와 native file dialog, system tray 등에 강하다.
- browser/server 없이 local-only tool을 만들 수 있다.

#### 단점

- SSH tunnel, headless Linux, remote workstation 접근성이 크게 떨어진다.
- web-based 기존 Inspector와 통합이 불편하다.
- packaging과 cross-platform 배포 비용이 증가한다.

#### 선택 조건

보안상 browser 접근을 금지하고, 단일 local workstation GUI만 허용하는 정책으로 바뀔 때만 재검토한다.

### 5.9 Gradio — 본 용도에는 비권장

Gradio는 ML model demo와 간단한 interaction에 매우 빠르지만, 복잡한 config editor, durable run registry, checkpoint lineage, 여러 page와 제어 권한을 갖춘 experiment platform에는 구조적 이점이 적다.

## 6. 최종 결정

### ADR-FE-001

```text
The benchmark execution visualization MVP shall use Dash with a FastAPI backend.
The frontend shall be replaceable and shall communicate only through versioned
service/API contracts.
```

### ADR-FE-002

```text
The existing Streamlit Run Inspector shall remain available during the MVP.
The new dashboard shall link to it by run_id or legacy artifact path.
No new lifecycle authority shall be added to Streamlit session state.
```

### ADR-FE-003

```text
Initial live updates shall use bounded polling over SQLite/JSONL.
WebSocket push may be introduced after event ordering, reconnect, and backpressure
acceptance tests pass.
```

### ADR-FE-004

```text
Dash background callbacks, Panel tasks, Shiny ExtendedTask, NiceGUI background
callbacks, or any equivalent frontend task mechanism shall not execute benchmark
training. Benchmark training shall always run in a dedicated worker process.
```

## 7. 재검토 조건

- Dash의 pinned version에서 WebSocket/reconnect 문제가 반복된다 → Panel 또는 React frontend spike
- UI code가 callback dependency로 유지 불가능해진다 → React + FastAPI 전환
- local desktop만 허용되고 remote browser가 금지된다 → PySide6 전환
- scenario/workflow orchestration 자체를 전면 재설계한다 → Taipy 평가 재개
- 기존 Inspector의 panel code 재사용이 핵심이고 Bokeh 생태계가 더 적합하다 → Panel 재평가

## 8. 공식 문서 출처

접근일: 2026-07-30

- Dash WebSocket callbacks: https://dash.plotly.com/websocket-callbacks
- Dash server backends / FastAPI: https://dash.plotly.com/server-backends
- Dash testing: https://dash.plotly.com/testing
- Dash multi-page apps: https://dash.plotly.com/urls
- Panel async callbacks: https://panel.holoviz.org/how_to/callbacks/async.html
- Panel periodic callbacks: https://panel.holoviz.org/how_to/callbacks/periodic.html
- Panel Plotly pane: https://panel.holoviz.org/reference/panes/Plotly.html
- Panel Tabulator: https://panel.holoviz.org/reference/widgets/Tabulator.html
- Streamlit fragments: https://docs.streamlit.io/develop/concepts/architecture/fragments
- Streamlit Session State caveats: https://docs.streamlit.io/develop/concepts/architecture/session-state
- NiceGUI documentation: https://nicegui.io/documentation
- Shiny for Python non-blocking operations: https://shiny.posit.co/py/docs/nonblocking.html
- Taipy Scenario API: https://docs.taipy.io/en/latest/refmans/reference/pkg_taipy/pkg_core/Scenario/
- Qt for Python QProcess: https://doc.qt.io/qtforpython-6/PySide6/QtCore/QProcess.html
- FastAPI WebSockets: https://fastapi.tiangolo.com/advanced/websockets/
