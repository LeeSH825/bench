# Code CLI Agent Prompt 2 — Visualization-Coexistence Dirty-Tree-Safe Phase 1A Gate A MEKF Math/Core

당신은 AI-ADCS-SNN benchmark 저장소의 **Phase 1A Gate A 구현 담당자**다.
같은 repository/working tree에서는 별도의 **benchmark 실행용 visualization tool 개발 작업**도 진행 중이다.
이번 실행의 목적은 그 visualization 작업과 기존 대규모 dirty working tree를 훼손하거나 정리하지 않은 상태에서,
Phase 0A에서 잠근 quaternion/MEKF 수학 계약을 따르는 **runner 비의존 NumPy/SciPy reference core**와
그 수학 검증 시험만 추가하는 것이다.

이번 실행은 전체 Phase 1A 구현이 아니다.
**Basilisk sensor generator, typed event schema, metric, model adapter, registry, runner, YAML,
Package C, neural network, SNN, FPGA는 구현하지 않는다.**

이 프롬프트의 dirty-tree 보호 규칙과 허용 파일 목록은 이번 실행에서 최우선이다.
Phase 0A의 LOCK 수학 계약과 충돌하는 지시가 발견되면 임의로 절충하지 말고 `BLOCKED`로 종료한다.

---

## V0. Visualization 동시 개발 공존 규칙 — 최우선 오버레이

이 절은 아래의 dirty-tree 보호 규칙, 종료 provenance 규칙, 완료 조건과 충돌할 경우 **우선 적용**한다.
이번 실행은 visualization tool을 구현·수정·검증하는 실행이 아니다. visualization workstream의 기존 변경과
실행 중 발생할 수 있는 별도 변경을 보존하면서, MEKF Gate A의 agent-owned diff만 분리해야 한다.

### V0.1 기본 원칙

1. visualization workstream은 이번 agent의 작업 범위 밖이며, 그 파일을 수정·정리·format·stage·commit하지 않는다.
2. MEKF Gate A는 아래 §6의 exact allowlist에만 새 파일을 만든다.
3. visualization 전용 경로의 **별도 작업에 의한 변경**은 Gate A 실패로 자동 간주하지 않는다.
4. 다만 visualization과 benchmark가 공유하는 핵심 파일이 실행 중 바뀌면 baseline과 provenance를 신뢰할 수 없으므로 즉시 중단한다.
5. visualization 변경을 이번 agent가 만들었다고 주장하지 말고, 원인을 식별할 수 없으면 `concurrent/unattributed visualization delta`로 기록한다.
6. 같은 working tree를 사용하더라도 visualization 실행 서버, GUI, dashboard, plot 생성, screenshot, browser automation은 시작하지 않는다.

### V0.2 경로 분류

실행 전에 repository-relative path를 다음 네 집합으로 분류하고 snapshot manifest에 기록한다.

#### A. `PHASE1A_AGENT_OWNED`

§6에 명시된 구현·시험·문서 파일과 다음 provenance 경로만 포함한다.

```text
experiments/phase1a/agent_logs/02_math_core_*
experiments/phase1a/preflight_snapshots/02_math_core_*/**
```

이 집합은 이번 agent만 새로 만들 수 있다. 실행 중 다른 process가 target을 먼저 만들거나 수정하면
`BLOCKED_TARGET_RACE`로 종료한다.

#### B. `VISUALIZATION_ISOLATED_PROTECTED`

기본적으로 다음을 포함한다.

```text
viz/**
visualization/**
bench/viz/**
bench/visualization/**
tests/test_viz*.py
tests/test_visual*.py
tests/**/viz/**
tests/**/visualization/**
bench/tests/test_viz*.py
bench/tests/test_visual*.py
docs/**/*viz*.md
docs/**/*visual*.md
scripts/**/*viz*
scripts/**/*visual*
```

추가로 다음 optional manifest가 존재하면 처음부터 끝까지 읽되 수정하지 않는다.

```text
docs/research/phase1a/VISUALIZATION_WORKSTREAM_PATHS.txt
```

manifest 형식은 빈 줄과 `#` 주석을 허용하며, 나머지 각 줄은 repository-relative path 또는 glob 하나다.
manifest가 없어도 기본 패턴으로 계속 진행한다.

다음 경로는 이름에 `viz`, `visual`, `plot`, `figure`, `dashboard`, `panel`이 포함되더라도
`VISUALIZATION_ISOLATED_PROTECTED`로 자동 분류하지 않는다.

```text
bench/runners/**
bench/metrics/**
bench/models/**
bench/tasks/**
bench/configs/**
pyproject.toml
uv.lock
requirements*
```

이들은 아래 shared-critical 집합이다.

#### C. `SHARED_CRITICAL_PROTECTED`

visualization과 estimator/benchmark가 함께 의존할 수 있어 실행 중 불변이어야 하는 경로다.

```text
bench/runners/**
bench/metrics/**
bench/models/registry.py
bench/tasks/bench_generated.py
bench/tasks/data_format.py
bench/tasks/generator/contract.py
bench/configs/**
bench/models/basilisk_mrp_ekf.py
bench/tasks/generator/basilisk_adcs.py
bench/tasks/generator/basilisk_imu_adcs.py
pyproject.toml
uv.lock
requirements*.txt
requirements*.lock
tests/test_basilisk_imu_generator.py
tests/test_basilisk_mrp_ekf.py
bench/tests/test_generator_contract_tg0.py
bench/tests/test_adcs_event_metrics.py
```

optional visualization manifest가 이 집합 또는 `PHASE1A_AGENT_OWNED`와 겹치면 자동으로 visualization 소유로
간주하지 말고 `BLOCKED_VIZ_MANIFEST_CONFLICT`로 종료한다.

#### D. `OTHER_PREEXISTING_DIRTY`

위 세 집합에 포함되지 않으면서 실행 전에 이미 modified/deleted/untracked인 경로다.
이 집합은 기존 dirty-tree 규칙대로 hash/삭제 상태가 변하지 않아야 한다.

### V0.3 visualization 파일을 읽고 실행하는 범위

- Gate A 구현에 visualization module은 필요하지 않으므로 `viz.*`, `visualization.*`, dashboard/frontend module을 import하지 않는다.
- visualization source를 formatter, linter fixer, code generator 대상으로 삼지 않는다.
- visualization test를 수집하거나 실행하지 않는다.
- repository 전체 test collection으로 인해 visualization dependency가 import되지 않도록, §5와 §11의 exact test list만 실행한다.
- 새 MEKF core가 visualization helper를 import하거나 visualization helper에 의존하면
  `BLOCKED_VIZ_IMPORT_COUPLING`으로 종료한다.
- visualization이 나중에 소비할 API를 이번 Gate A에서 미리 만들거나 shared integration file을 수정하지 않는다.

### V0.4 preflight snapshot과 visualization workstream의 분리

기존 snapshot artifact에 다음을 추가한다.

```text
PATH_CLASSIFICATION.tsv
VIZ_PATHS_BEFORE.z
VIZ_HASHES_BEFORE.tsv
SHARED_CRITICAL_HASHES_BEFORE.tsv
OTHER_DIRTY_HASHES_BEFORE.tsv
TARGET_PATHS_BEFORE.tsv
```

`VISUALIZATION_ISOLATED_PROTECTED` 파일은 현재 상태를 hash/size/mtime으로 기록한다.
이 Phase 1A snapshot은 visualization workstream의 공식 backup이라고 주장하지 않는다.
기존 untracked archive를 만들 때 visualization 경로가 포함되어도 읽기만 하며 수정하지 않는다.
archive가 크기/읽기 문제로 안전하게 생성되지 않으면 기존 §3.2에 따라 중단한다.

snapshot 직후 동일 집합을 한 번 더 fingerprint한다. 두 fingerprint 사이에:

- `PHASE1A_AGENT_OWNED` target이 생기거나 변하면 `BLOCKED_TARGET_RACE`
- `SHARED_CRITICAL_PROTECTED`가 변하면 `BLOCKED_CONCURRENT_SHARED_CHANGE`
- `OTHER_PREEXISTING_DIRTY`가 변하면 `BLOCKED_CONCURRENT_OTHER_CHANGE`
- `VISUALIZATION_ISOLATED_PROTECTED`만 변하면 `VIZ_CONCURRENT_ACTIVITY_DETECTED=YES`로 기록하고 계속할 수 있다

visualization 전용 변경이 계속되더라도 agent-owned target 및 shared-critical 경로와 겹치지 않으면 Gate A를 진행할 수 있다.

### V0.5 실행 중 checkpoint

다음 각 시점에 path fingerprint를 다시 계산한다.

1. 구현 전 baseline regression 직전
2. 구현 전 baseline regression 직후
3. agent-owned 파일 생성 직후
4. 신규 Gate A test 직전/직후
5. legacy regression 직전/직후
6. 최종 보고서 작성 직전

판정:

- visualization-isolated delta만 존재: 계속 진행하고 별도 ledger에 기록
- shared-critical delta: 즉시 `BLOCKED_CONCURRENT_SHARED_CHANGE`
- other pre-existing dirty delta: 즉시 `BLOCKED_CONCURRENT_OTHER_CHANGE`
- agent target의 외부 생성/변경: 즉시 `BLOCKED_TARGET_RACE`

shared-critical 변경을 발견한 뒤 baseline test를 다시 돌려 우회하지 않는다. 현재 run의 provenance가 깨진 것이므로 중단한다.

### V0.6 agent-only diff 계산

agent-only patch/stat에는 오직 `PHASE1A_AGENT_OWNED` 파일만 포함한다.
실행 중 변화한 visualization-isolated 파일은 global Git status에는 나타날 수 있지만 agent-only diff에 넣지 않는다.
다음 artifact를 추가한다.

```text
experiments/phase1a/agent_logs/02_math_core_viz_paths_before.tsv
experiments/phase1a/agent_logs/02_math_core_viz_paths_after.tsv
experiments/phase1a/agent_logs/02_math_core_viz_external_delta.tsv
experiments/phase1a/agent_logs/02_math_core_shared_critical_integrity.tsv
experiments/phase1a/agent_logs/02_math_core_path_classification.tsv
```

`viz_external_delta.tsv`에는 path, before hash/state, after hash/state, 최초 감지 checkpoint를 기록한다.
원인을 증명할 수 없으므로 작성 주체를 특정하지 않는다.

### V0.7 완료 조건 오버라이드

아래 §13과 §15의 `pre-existing dirty files unchanged` 조건은 다음처럼 해석한다.

```text
PHASE1A_AGENT_OWNED:               이번 agent가 allowlist 안에서만 생성
SHARED_CRITICAL_PROTECTED:         실행 내내 완전 불변
OTHER_PREEXISTING_DIRTY:           실행 내내 완전 불변
VISUALIZATION_ISOLATED_PROTECTED:  agent가 수정하지 않음; 별도 concurrent delta는 허용·기록
```

따라서 visualization-isolated external delta가 있다는 사실만으로 Gate A를 FAIL 처리하지 않는다.
그러나 다음 중 하나면 Gate A는 반드시 STOP이다.

- visualization manifest가 target/shared-critical 경로와 겹침
- visualization 작업이 shared-critical 경로를 실행 중 변경
- visualization 작업이 agent-owned target을 생성·변경
- MEKF core/test가 visualization module을 import
- agent가 visualization-isolated 경로를 직접 수정

### V0.8 최종 응답에 추가할 항목

최종 응답과 `P1A_MATH_VALIDATION_REPORT.md`에 다음을 추가한다.

```text
Visualization isolation: PASS/FAIL
Visualization concurrent activity: NONE/DETECTED
Visualization paths changed during run: <count>
Shared-critical stability: PASS/FAIL
Target race: NONE/DETECTED
Visualization import coupling: NONE/DETECTED
```

visualization concurrent activity가 감지된 경우, 변경 경로 수와 ledger 위치만 보고하고
그 변경 내용을 Phase 1A 성과나 agent diff로 요약하지 않는다.

### V0.9 동일 working tree 사용이 불가능한 조건

다음 중 하나면 prompt를 확장하여 억지로 공존하지 않는다.

1. visualization workstream이 `bench/runners/**`, `bench/metrics/**`, registry, generator contract 등 shared-critical 파일을 계속 수정해야 함
2. visualization workstream이 `bench/estimators/**` 또는 `tests/test_mekf_*`를 동시에 수정함
3. visualization test 실행이 shared config/dependency를 변경함
4. target race 또는 shared-critical concurrent change가 반복됨

이 경우 최소 사용자 행동은 visualization 실행을 잠시 멈추거나 별도 Git worktree로 분리하는 것이다.

---

## 0. 작업 위치와 예상 감사 기준

현재 repository root는 감사 당시 다음이었다.

```text
/home/dss-pc-05/bench
```

감사 당시 기준은 다음이었다.

```text
branch: benchmark-viz/stabilize-release-baseline
HEAD:   ee862a2acc368fb631c45ef0b33a8f4feb5c28c0
```

그러나 위 값을 무조건 가정하지 말고, 현재 실행 시점의 root/branch/HEAD를 직접 확인하여 기록한다.

- `docs/research/phase1a/approvals/01_AUDIT_APPROVED.md`가 존재하고 그 안에 승인된
  `baseline_commit`이 명시되어 있으면 그 값을 우선 기준으로 사용한다.
- 승인 문서가 없으면 `P1A_REPOSITORY_AUDIT.md`에 기록된 branch/HEAD와 현재 값을 비교한다.
- 승인된 기준과 현재 HEAD가 다르면 구현하지 말고 `BLOCKED_BASELINE_MISMATCH`로 종료한다.
- working tree가 dirty인 것 자체는 이번 실행의 중단 사유가 아니다. 단, 아래 snapshot과 provenance
  검증을 통과해야 한다.

---

## 1. 반드시 먼저 읽을 문서

다음 문서를 처음부터 끝까지 읽고, 서로 다른 내용을 임의로 합치지 않는다.

### Phase 0A locked source of truth

1. `docs/research/phase0a/decision_lock/P0A_PHASE_0A_SYNTHESIS.md`
2. `docs/research/phase0a/decision_lock/P0_01_DECISION_LEDGER.md`
3. `docs/research/phase0a/decision_lock/P0_02_TRUTH_SENSOR_ESTIMATOR_BOUNDARY.md`
4. `docs/research/phase0a/decision_lock/P0_05_MEKF_MATH_CONTRACT.md`
5. `docs/research/phase0a/decision_lock/P0_05_MEKF_CONVENTION_TEST_VECTORS.md`
6. `docs/research/phase0a/decision_lock/P0A_IMMEDIATE_TEST_SPEC.md`

### Phase 1A audit source

7. `docs/research/phase1a/P1A_REPOSITORY_AUDIT.md`
8. `docs/research/phase1a/P1A_IMPLEMENTATION_MAP.md`
9. `docs/research/phase1a/P1A_RISK_REGISTER.md`

### 현재 실행 프롬프트

10. 이 프롬프트 파일 자체

필수 문서가 없거나 읽을 수 없으면 코드 구현을 시작하지 말고
`BLOCKED_MISSING_INPUT`으로 종료한다.

---

## 2. 변경 불가능한 수학 계약

다음 계약은 이번 실행에서 변경할 수 없다.

```text
filter:                 6D kinematic MEKF
nominal state:          [q_NB, b_g]
local error state:      [delta_theta, delta_b_g] in R^6
quaternion ordering:    scalar-first [w, x, y, z]
quaternion algebra:     Hamilton
attitude meaning:       active body-to-navigation q_NB
multiplicative error:   right-multiplicative
truth relation:         q_true = q_hat ⊗ Exp_q(delta_theta)
gyro model:             omega_m = omega_true + b_g + n_g
numeric reference:      float64
first update primitive: star-tracker tangent residual
covariance update:      Joseph form
covariance solve:       Cholesky/triangular solve 우선
```

세부 부호, frame, Jacobian, injection, reset 식은
`P0_05_MEKF_MATH_CONTRACT.md`와 convention test vectors를 그대로 따른다.
문서와 구현 사이에 모순이 보이면 수식을 임의 수정하지 말고 해당 식, 영향 범위,
재현 test를 기록한 뒤 `BLOCKED_MATH_CONTRACT_CONFLICT`로 종료한다.

---

## 3. Dirty working tree 보호 정책

감사 당시 working tree에는 수백 개의 modified/deleted/untracked 항목이 있었다.
이 기존 변경은 이번 agent의 작업 대상이 아니며, **삭제·복원·정리·stage·commit하면 안 된다.**

### 3.1 절대 실행 금지 Git 명령/동작

다음은 사용하지 않는다.

```text
git reset
git clean
git restore
git checkout -- <path>
git switch
git stash
git add
git rm
git commit
git merge
git rebase
git cherry-pick
git push
```

동일한 효과를 내는 파일 일괄 삭제, repository-wide formatter/fixer도 금지한다.

다음도 금지한다.

```text
pip install / uninstall
uv sync / lock 변경
poetry/pipenv dependency 변경
apt 또는 system package 변경
pyproject.toml 수정
uv.lock 수정
requirements*.txt 또는 requirements*.lock 수정
repository 전체 ruff --fix / black / isort
```

### 3.2 구현 전 recoverable snapshot

어떤 source/test/report 파일도 만들기 전에 현재 dirty 상태를 snapshot한다.

권장 snapshot 위치:

```text
experiments/phase1a/preflight_snapshots/02_math_core_<UTC_TIMESTAMP>/
```

snapshot 디렉터리를 만들기 전에 `/tmp`의 임시 디렉터리에서 현재 상태를 먼저 캡처하여,
snapshot 자체가 pre-existing untracked 목록에 포함되지 않게 한다.

최소한 다음 artifact를 보존한다.

```text
REPO_ROOT.txt
BRANCH.txt
HEAD.txt
STATUS_BEFORE.txt
STATUS_BEFORE.z
WORKTREE_TRACKED.patch        # git diff --binary
INDEX_STAGED.patch             # git diff --cached --binary
UNTRACKED_BEFORE.z             # git ls-files --others --exclude-standard -z
PREEXISTING_DIRTY_HASHES.tsv   # 기존 dirty/untracked 파일의 SHA-256 또는 DELETED 상태
SNAPSHOT_MANIFEST.md
```

가능하면 `UNTRACKED_BEFORE.z`에 기록된 기존 untracked 파일을 null-safe 방식으로
`UNTRACKED_BEFORE.tar.gz`에 보존한다.

- archive 예상 크기가 1 GiB를 초과하거나, 읽을 수 없는 파일이 있거나,
  안전하게 archive할 수 없으면 구현을 시작하지 않는다.
- 이 경우 현재 상태와 원인을 로그에 남기고 `BLOCKED_DIRTY_SNAPSHOT`으로 종료한다.
- ignored cache/data는 archive 대상이 아니지만, snapshot manifest에 ignored 대용량 artifact를
  보존하지 않았음을 명시한다.
- 기존 deleted tracked 파일은 binary patch에 의해 복구 가능해야 한다.

### 3.3 기존 dirty 파일 보호용 hash baseline

`git status --porcelain=v1 -z -uall`을 안전하게 파싱하여, 실행 전부터 dirty/untracked였던
모든 실제 파일의 SHA-256을 기록한다. 삭제된 경로는 `DELETED`, rename은 old/new path를 모두 기록한다.

실행 종료 시 다음을 검증한다.

1. 이번 실행 전부터 dirty였던 파일은 허용 대상 파일이 아닌 한 hash/삭제 상태가 변하지 않았다.
2. 실행 전 clean이었던 tracked 파일이 허용 목록 밖에서 새로 modified/deleted되지 않았다.
3. 실행 전 없던 untracked 파일은 허용된 output/log/snapshot 경로에만 생겼다.

한 항목이라도 어기면 자동 복원하지 말고 `BLOCKED_UNINTENDED_CHANGE`로 종료하며,
변경 경로와 before/after hash를 보고한다.

---

## 4. Target collision 검사

아래 §6의 구현·시험·문서 target이 실행 전에 이미 존재하는지 확인한다.

- target이 하나라도 이미 존재하면 내용을 덮어쓰거나 합치지 않는다.
- existing target의 path, Git 상태, size, SHA-256을 기록한다.
- 구현을 시작하지 않고 `BLOCKED_TARGET_EXISTS`로 종료한다.

단, 이번 실행이 생성하는 agent log와 preflight snapshot 디렉터리는 기존에 없어야 하며,
동일 timestamp 충돌 시 새 timestamp를 사용한다.

---

## 5. 구현 전 baseline 회귀 시험

snapshot과 target collision 검사를 통과한 뒤, source를 작성하기 전에 다음 interpreter를 사용한다.

```text
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python
```

먼저 다음을 기록한다.

```text
Python version
NumPy version
SciPy version
pytest version
repository root
branch
HEAD
```

기본 `python` shim에 의존하지 않는다.

다음 baseline regression subset을 **구현 전에 먼저 실행**한다.

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_basilisk_imu_generator.py \
  tests/test_basilisk_mrp_ekf.py \
  bench/tests/test_generator_contract_tg0.py \
  bench/tests/test_adcs_event_metrics.py
```

결과 전체를 다음과 같은 agent log에 저장한다.

```text
experiments/phase1a/agent_logs/02_math_core_baseline_before.txt
```

조건:

- exit code가 0이어야 한다.
- 실패, collection error, import error가 하나라도 있으면 구현하지 않는다.
- 감사 당시 통과 결과와 달라진 경우 원인을 임의 수정하지 않고
  `BLOCKED_BASELINE_REGRESSION`으로 종료한다.
- 기존 test, fixture, expectation, tolerance를 수정해서 통과시키지 않는다.

---

## 6. 이번 실행의 허용 파일 목록

### 6.1 새로 생성할 구현 파일

```text
bench/estimators/__init__.py
bench/estimators/mekf.py
```

### 6.2 새로 생성할 시험 파일

```text
tests/test_mekf_conventions.py
tests/test_mekf_core.py
```

### 6.3 새로 생성할 계약/검증 문서

```text
docs/research/phase1a/P1A_IMPLEMENTATION_CONTRACT.md
docs/research/phase1a/P1A_TEST_MATRIX.md
experiments/phase1a/reports/P1A_MATH_VALIDATION_REPORT.md
```

### 6.4 실행 provenance/log artifact

다음 하위 경로의 새 파일만 허용한다.

```text
experiments/phase1a/agent_logs/02_math_core_*
experiments/phase1a/preflight_snapshots/02_math_core_*/**
```

위 목록 밖의 파일이 필요하다고 판단되면 실제로 수정하지 않는다.
필요한 파일, 이유, public contract 영향, 대안을 보고하고 `BLOCKED_SCOPE_EXPANSION`으로 종료한다.

---

## 7. 명시적 수정 금지 경로

다음 경로는 읽을 수 있지만 수정하지 않는다.

```text
bench/tasks/**
bench/models/**
bench/metrics/**
bench/runners/**
bench/configs/**
bench/tasks/generator/**
bench/models/basilisk_mrp_ekf.py
bench/tasks/generator/basilisk_adcs.py
bench/tasks/generator/basilisk_imu_adcs.py
bench/metrics/adcs_event.py
bench/tasks/data_format.py
bench/tasks/generator/contract.py
pyproject.toml
uv.lock
requirements*.txt
requirements*.lock
docs/research/phase0a/**
docs/research/phase1a/prompts/**
third_party/**
기존 test 파일 및 기존 기대값
기존 suite YAML
```

repository-wide `__init__.py`, registry, packaging 설정도 수정하지 않는다.
`bench/estimators`를 built wheel에 포함하는 문제는 이번 Gate A 범위 밖이다.

---

## 8. 이번 구현 범위 — Gate A pure math/core only

`bench/estimators/mekf.py`는 Python standard library, NumPy, SciPy만 사용하는
float64 reference implementation이어야 한다.

### 8.1 허용 import

```text
Python standard library
numpy
scipy.linalg
```

### 8.2 금지 import

```text
Basilisk
torch
bench.runners.*
bench.models.*
bench.tasks.*
bench.metrics.*
YAML/config loader
visualization module
training/neural/SNN/FPGA module
```

### 8.3 구현할 기능

최소한 다음 기능을 구현한다. 함수명과 class 구조는 저장소 스타일에 맞게 정할 수 있지만,
`P1A_IMPLEMENTATION_CONTRACT.md`에서 수식과 실제 함수 경로를 1:1로 대응시킨다.

#### Quaternion / SO(3)

- scalar-first Hamilton quaternion normalize
- conjugate, inverse, product
- quaternion ↔ DCM
- SO(3) quaternion Exp/Log
- hemisphere/sign alignment
- skew-symmetric matrix
- right Jacobian `J_r`와 필요한 안정적 small-angle branch
- `q`와 `-q`의 물리적 동치 처리

#### Kinematic MEKF core

- nominal state `[q_NB, b_g]`
- local error state `[delta_theta, delta_b_g]`
- gyro propagation using `omega_m - b_g`
- continuous error matrices `F`, `G`, `Q_c`
- discrete transition `Phi`
- discrete process covariance `Q_d`
- Phase 0A 계약에 맞는 exact/Van-Loan 또는 승인된 reference discretization
- body-vector measurement prediction
- body-vector analytic Jacobian
- star-tracker quaternion residual의 3D tangent mapping primitive
- Kalman innovation covariance와 gain solve primitive
- Joseph covariance update
- multiplicative attitude injection
- gyro-bias correction
- local error reset
- right-reset covariance transport
- symmetry/SPD diagnostics

### 8.4 입력/출력 안전성

- public numeric input은 shape, finite value, unit/frame 의미를 명확히 검증한다.
- reference path는 `float64`를 사용한다.
- quaternion은 계약에서 정한 위치에서만 normalize한다.
- estimator core API는 truth attitude, true bias, event label, true Q/R scale을 입력으로 받지 않는다.
- measurement/event packet schema는 이번 실행에서 만들지 않는다.
- latency/OOSM, magnetometer/sun/ST generator는 구현하지 않는다.

---

## 9. 수치 안전 정책

다음 방식으로 오류를 숨기지 않는다.

```text
pseudo-inverse fallback 금지
eigenvalue clipping 금지
silent diagonal jitter 금지
non-SPD covariance 자동 보정 금지
NaN/Inf 무시 금지
```

허용되는 수치 처리:

- quaternion normalization
- floating-point roundoff 수준의 covariance symmetrization `0.5*(P+P.T)`
- small-angle analytic series

단, covariance symmetrization은 correction norm을 진단 가능하게 만들고,
계약 또는 test tolerance를 넘는 비대칭은 실패시킨다.

`P`, `Q_d`, `R`, `S`의 SPD/PSD 요구를 구분하고,
Cholesky가 필요한 행렬은 실패를 명시적으로 발생시킨다.
의도적으로 non-SPD인 test에서는 fail-loud 동작을 검증한다.

---

## 10. 필수 시험

모든 시험은 deterministic해야 한다. 난수를 사용할 경우 seed를 고정하고 report에 기록한다.
실패를 `skip`, `xfail`, `xpass`, tolerance 완화로 숨기지 않는다.

### 10.1 `tests/test_mekf_conventions.py`

최소 포함:

1. Phase 0A convention vector 전부
2. identity quaternion/DCM
3. body x/y/z 축의 +90° 회전과 basis-vector mapping
4. Hamilton composition order
5. inverse/conjugate consistency
6. Exp/Log small-angle round trip
7. quaternion normalization
8. `q`와 `-q`의 DCM/geodesic/residual 동치
9. right-multiplicative injection order
10. near-zero 및 계약상 필요한 near-pi 경계 동작

### 10.2 `tests/test_mekf_core.py`

최소 포함:

#### B1 — propagation/discretization/bias sign

- zero motion
- constant angular rate
- known constant gyro bias cancellation
- gyro-bias sign test
- continuous local dynamics shape/unit check
- analytic local transition과 finite-difference/reference 비교
- `Phi`와 `Q_d`의 shape/symmetry
- 허용된 exact/first-order relation 확인

#### B3 — magnetometer/body-vector Jacobian

- analytic Jacobian vs central finite difference
- locked frame/sign convention 검증

#### B4 — sun-vector tangent Jacobian primitive

- tangent basis가 포함된 analytic Jacobian vs central finite difference
- rank/shape 검증

#### B5 — injection/reset

- known small attitude correction
- bias correction
- injection 후 nominal state와 residual relation
- reset Jacobian과 finite-difference/reference 비교
- covariance reset symmetry/SPD

#### B6 — sign invariance

- ST measurement `q`와 `-q`가 같은 innovation/update를 생성
- nominal quaternion sign이 바뀌어도 물리적 posterior와 covariance가 동일

#### Numerical safety

- Joseph update symmetry
- valid SPD covariance Cholesky success
- deliberate non-SPD `P`/`S`가 명시적으로 실패
- pseudo-inverse/jitter/clipping fallback 부재
- nonfinite input fail-loud

#### Import boundary

- `bench.estimators.mekf`를 import할 때 Basilisk, torch, runner, model, task가 import되지 않음

### 10.3 tolerance 정책

- Phase 0A 문서에 provisional tolerance가 있으면 그대로 사용한다.
- 명시값이 없으면 machine precision, finite-difference step, conditioning 근거를 사용하여
  보수적인 tolerance를 정하고 문서화한다.
- test 실패를 해결하기 위해 tolerance를 반복적으로 넓히지 않는다.
- tolerance 변경이 필요하면 원인과 민감도 evidence 없이 진행하지 말고 `BLOCKED_TOLERANCE`로 종료한다.

---

## 11. 시험 실행 순서

구현 후 다음 순서로 실행한다.

### 11.1 신규 Gate A test

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_mekf_conventions.py \
  tests/test_mekf_core.py
```

### 11.2 기존 regression subset 재실행

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_basilisk_imu_generator.py \
  tests/test_basilisk_mrp_ekf.py \
  bench/tests/test_generator_contract_tg0.py \
  bench/tests/test_adcs_event_metrics.py
```

각 명령의 stdout/stderr, exit code, duration을 다음 로그에 보존한다.

```text
experiments/phase1a/agent_logs/02_math_core_new_tests.txt
experiments/phase1a/agent_logs/02_math_core_baseline_after.txt
```

full repository test는 이번 dirty-tree Gate A에서 강제로 실행하지 않는다.
선택 regression이 실패하면 기존 기대값을 바꾸지 말고 원인과 영향 범위를 보고한다.

---

## 12. 필수 문서 산출물

### 12.1 `P1A_IMPLEMENTATION_CONTRACT.md`

최소 포함:

- 목적, 입력 근거, 결정 상태, 남은 TBD, 다음 Gate
- locked convention 요약
- 각 수식 기능과 실제 함수/class 경로 1:1 mapping
- array shape, dtype, unit, frame
- prior/posterior notation
- normalization 위치
- Cholesky/fail-loud policy
- truth/sensor/estimator 정보 경계
- 이번 Gate A에서 구현하지 않은 기능

### 12.2 `P1A_TEST_MATRIX.md`

최소 포함:

| Test ID | 수학 계약 | 시험 파일/함수 | 입력 | expected behavior | tolerance | 결과 | evidence |

B1, B3, B4, B5, B6와 numerical safety/import-boundary test를 모두 기록한다.

### 12.3 `P1A_MATH_VALIDATION_REPORT.md`

최소 포함:

1. 최종 상태: `PASS`, `FAIL`, 또는 `BLOCKED`
2. repository root, branch, HEAD
3. dirty snapshot 경로와 snapshot completeness
4. Python/NumPy/SciPy/pytest version
5. 구현 전 baseline regression 결과
6. 생성한 파일 목록
7. agent-only diff/stat
8. 신규 test exact command/result
9. 구현 후 legacy regression exact command/result
10. B1/B3/B4/B5/B6별 evidence
11. SPD/non-SPD safety evidence
12. numerical correction ledger
13. pre-existing dirty path 보호 검증 결과
14. unresolved issue
15. 다음 단계에서 허용할 범위 제안
16. **Prompt 3 또는 Gate B를 실제 실행하지 않았음**을 명시

---

## 13. Agent-only diff와 종료 provenance 검증

현재 repository 전체 `git diff --stat`은 기존 dirty 변경을 포함하므로 agent 변경량의 근거로 단독 사용하지 않는다.

실행 전 snapshot과 비교하여 이번 실행이 만든 파일만 대상으로 다음 artifact를 생성한다.

```text
experiments/phase1a/agent_logs/02_math_core_agent_only.patch
experiments/phase1a/agent_logs/02_math_core_agent_only_stat.txt
experiments/phase1a/agent_logs/02_math_core_changed_paths.txt
experiments/phase1a/agent_logs/02_math_core_status_after.txt
experiments/phase1a/agent_logs/02_math_core_dirty_integrity_check.tsv
```

새 파일은 null file(`/dev/null`)과의 `diff --no-index` 등으로 patch를 만든다.
전체 repository의 global `git diff --stat`과 `git status --short --branch`도 참고용으로 기록하되,
이를 agent-only diff라고 부르지 않는다.

종료 전에 반드시 다음을 판정한다.

```text
pre-existing non-visualization dirty files unchanged: PASS/FAIL
visualization-isolated paths untouched by this agent:   PASS/FAIL
shared-critical paths stable during run:                PASS/FAIL
concurrent visualization delta:                         NONE/DETECTED
new changes confined to allowlist:                      PASS/FAIL
target files newly created only:     PASS/FAIL
baseline regression preserved:       PASS/FAIL
Gate A tests:                         PASS/FAIL
```

하나라도 FAIL이면 최종 Gate A를 PASS로 표시하지 않는다.

---

## 14. 중단 조건

다음 중 하나라도 발생하면 범위를 확장하거나 우회하지 말고 중단한다.

```text
BLOCKED_MISSING_INPUT
BLOCKED_BASELINE_MISMATCH
BLOCKED_DIRTY_SNAPSHOT
BLOCKED_TARGET_EXISTS
BLOCKED_BASELINE_REGRESSION
BLOCKED_MATH_CONTRACT_CONFLICT
BLOCKED_SCOPE_EXPANSION
BLOCKED_TOLERANCE
BLOCKED_UNINTENDED_CHANGE
BLOCKED_ENVIRONMENT
BLOCKED_VIZ_MANIFEST_CONFLICT
BLOCKED_CONCURRENT_SHARED_CHANGE
BLOCKED_CONCURRENT_OTHER_CHANGE
BLOCKED_TARGET_RACE
BLOCKED_VIZ_IMPORT_COUPLING
```

중단 시:

- 기존 dirty source를 수정·복원하지 않는다.
- 가능한 경우 preflight/log artifact만 남긴다.
- target report가 실행 전에 존재하지 않았고 허용 범위 안이라면
  `P1A_MATH_VALIDATION_REPORT.md`에 `BLOCKED` 상태와 정확한 원인을 기록한다.
- 해결에 필요한 최소 사용자 행동을 한 가지로 압축하여 보고한다.

---

## 15. 완료 조건

다음이 모두 충족돼야 Gate A를 `PASS`로 판정한다.

1. recoverable dirty snapshot 생성 완료
2. 승인된 branch/HEAD 기준 일치
3. 구현 전 baseline regression exit code 0
4. target collision 없음
5. 허용된 7개 구현/시험/문서 파일만 새로 생성
6. B1/B3/B4/B5/B6 및 convention tests 전부 통과
7. deliberate non-SPD fail-loud test 통과
8. 구현 후 legacy regression exit code 0
9. pre-existing non-visualization dirty/untracked 파일 hash 무변경
10. shared-critical 경로가 실행 내내 불변
11. visualization-isolated 경로를 이번 agent가 수정하지 않음; 별도 concurrent delta는 ledger에 기록
12. 허용 목록 밖 신규 agent 변경 없음
13. 문서 3개 완성
14. commit, stage, push를 수행하지 않음
15. Gate B/Prompt 3을 시작하지 않음

---

## 16. 최종 응답 형식

최종 agent 응답은 다음 순서를 따른다.

### Status

```text
PASS / FAIL / BLOCKED_<REASON>
```

### Baseline and dirty-tree protection

- repository root
- branch / HEAD
- snapshot path
- initial dirty counts
- pre-existing non-visualization dirty integrity result
- visualization isolation result
- visualization concurrent activity and delta ledger
- shared-critical stability result

### Files created

- 구현
- tests
- documents
- logs/snapshot

### Test results

- pre-implementation baseline
- Gate A new tests
- post-implementation legacy regression
- exact pass/fail count와 duration

### Gate evidence

```text
B1: PASS/FAIL
B3: PASS/FAIL
B4: PASS/FAIL
B5: PASS/FAIL
B6: PASS/FAIL
Numerical safety: PASS/FAIL
Dirty-tree integrity: PASS/FAIL
Visualization isolation: PASS/FAIL
Visualization concurrent activity: NONE/DETECTED
Shared-critical stability: PASS/FAIL
Gate A: GO/STOP
```

### Diff provenance

- agent-only changed paths
- agent-only stat
- global dirty-tree stat는 참고값으로 별도 표기

### Unresolved issues

- 이번 Gate A에 남은 문제만 기록

### Next action

- Gate A가 PASS여도 Prompt 3을 실행하지 말고 종료
- 다음 단계는 Chat 검토 후 별도 승인 필요라고 명시

