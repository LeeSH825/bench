# Code CLI Agent Prompt 2 — Auto-Baseline-Reconciling, Visualization-Coexistence, Dirty-Tree-Safe Phase 1A Gate A MEKF Math/Core

당신은 AI-ADCS-SNN benchmark 저장소의 **Phase 1A Gate A 구현 담당자**다.
같은 repository/working tree에서는 별도의 **benchmark 실행용 visualization tool 개발 작업**도 진행 중이다.
이번 실행의 목적은 그 visualization 작업과 기존 대규모 dirty working tree를 훼손하거나 정리하지 않은 상태에서,
Phase 0A에서 잠근 quaternion/MEKF 수학 계약을 따르는 **runner 비의존 NumPy/SciPy reference core**와
그 수학 검증 시험만 추가하는 것이다.

이번 실행은 전체 Phase 1A 구현이 아니다.
**Basilisk sensor generator, typed event schema, metric, model adapter, registry, runner, YAML,
Package C, neural network, SNN, FPGA는 구현하지 않는다.**

이 프롬프트는 감사 이후 HEAD가 이동한 경우 단순히 `BLOCKED_BASELINE_MISMATCH`로 종료하지 않는다.
감사 commit과 현재 commit의 관계, commit delta, visualization/shared-critical 영향, 기존 회귀시험을 먼저 검토하고,
조건을 만족하면 **현재 HEAD를 Gate A 범위에 한정해 자동 승인**한 뒤 구현을 계속한다.
관계가 분기되었거나 locked contract/dependency/target 충돌이 있으면 자동 승인하지 않고 명확한 상태로 중단한다.

이 프롬프트의 baseline reconciliation, dirty-tree 보호, visualization 공존 규칙과 허용 파일 목록은 이번 실행에서 최우선이다.
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

§6에 명시된 구현·시험·문서 파일과 다음 baseline/provenance 경로만 포함한다.

```text
docs/research/phase1a/approvals/01_AUDIT_APPROVED.md
experiments/phase1a/reports/P1A_BASELINE_RECONCILIATION_*.md
experiments/phase1a/agent_logs/02_math_core_*
experiments/phase1a/preflight_snapshots/02_math_core_*/**
```

`01_AUDIT_APPROVED.md`는 이전 실행의 stale marker가 존재할 수 있으므로, §0의 reconciliation을 통과한 뒤에만
snapshot에 보존된 이전 내용을 근거로 생성 또는 갱신할 수 있다. 이 파일을 stage/commit하지 않는다.

이 집합은 이번 agent만 새로 만들거나 위 승인 marker에 한해 갱신할 수 있다. 실행 중 다른 process가
Gate A target 또는 승인 marker를 먼저 만들거나 수정하면 `BLOCKED_TARGET_RACE`로 종료한다.

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

## 0. Baseline reconciliation 및 Gate A 자동 승인 절차

이 절은 구현, snapshot, test, 문서 생성보다 먼저 수행한다. 단, repository에 파일을 쓰기 전의 조사 결과는
`/tmp/p1a_gate_a_baseline_<UTC_TIMESTAMP>/` 아래에만 임시 저장한다.

### 0.1 감사 기준과 현재 기준 수집

감사 문서에 기록된 예상 기준은 다음이다.

```text
repository root: /home/dss-pc-05/bench
branch:          benchmark-viz/stabilize-release-baseline
audit HEAD:      ee862a2acc368fb631c45ef0b33a8f4feb5c28c0
```

위 값을 맹목적으로 사용하지 않는다. 다음을 직접 확인한다.

```bash
git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git status --porcelain=v1 -z -uall
```

그리고 `P1A_REPOSITORY_AUDIT.md`에서 감사 branch와 HEAD를 다시 추출해 서로 일치하는지 확인한다.
`docs/research/phase1a/approvals/01_AUDIT_APPROVED.md`가 이미 존재하면 그 내용과 SHA-256도 읽지만,
기존 marker가 있다는 이유만으로 reconciliation을 생략하지 않는다.

다음 변수를 명시적으로 기록한다.

```text
AUDIT_BRANCH
AUDIT_HEAD
CURRENT_BRANCH
RUN_HEAD        # 실행 시작 시 CURRENT HEAD. 이후 전 과정에서 고정되어야 함
PRIOR_APPROVAL_BASELINE   # 없으면 NONE
```

### 0.2 commit 존재성과 branch 확인

1. `AUDIT_HEAD`와 `RUN_HEAD`가 유효한 local commit인지 `git cat-file -e <sha>^{commit}`으로 확인한다.
2. `AUDIT_HEAD`가 없으면 `BLOCKED_AUDIT_COMMIT_UNAVAILABLE`로 종료한다.
3. `CURRENT_BRANCH`가 `AUDIT_BRANCH`와 다르면 자동 승인하지 않고
   `BLOCKED_BASELINE_BRANCH_MISMATCH`로 종료한다.
4. detached HEAD이면 동일하게 중단한다.

### 0.3 감사 HEAD와 현재 HEAD의 관계 판정

다음 세 경우를 구분한다.

#### Case A — `RUN_HEAD == AUDIT_HEAD`

```text
BASELINE_RELATION=EXACT_MATCH
```

commit delta가 없으므로 snapshot과 구현 전 baseline regression으로 넘어간다.

#### Case B — `AUDIT_HEAD`가 `RUN_HEAD`의 ancestor

```bash
git merge-base --is-ancestor "$AUDIT_HEAD" "$RUN_HEAD"
```

성공하면:

```text
BASELINE_RELATION=FAST_FORWARD_DESCENDANT
```

으로 기록하고, 아래 commit delta 감사를 수행한다.

#### Case C — ancestor 관계가 아님

rebase, reset, branch divergence 또는 history rewrite 가능성이 있으므로 자동 승인하지 않는다.
working-tree 내용이 유사해 보여도 추측으로 진행하지 않고 다음으로 종료한다.

```text
BLOCKED_BASELINE_DIVERGED_REAUDIT_REQUIRED
```

### 0.4 commit delta 증거 생성

Case B에서는 `/tmp`에 최소 다음을 생성한다.

```text
AUDIT_HEAD.txt
RUN_HEAD.txt
HEAD_RELATION.txt
HEAD_LOG.txt
git diff --stat 결과
HEAD_DELTA_NAME_STATUS.z
HEAD_DELTA_NAME_STATUS.txt
HEAD_DELTA_CHECK.txt
```

사용할 핵심 명령:

```bash
git log --oneline --decorate "$AUDIT_HEAD..$RUN_HEAD"
git diff --stat "$AUDIT_HEAD..$RUN_HEAD"
git diff --name-status -z "$AUDIT_HEAD..$RUN_HEAD"
git diff --check "$AUDIT_HEAD..$RUN_HEAD"
```

`git diff --check`가 실패하면 자동 수정하지 말고 `BLOCKED_BASELINE_DELTA_INVALID`로 종료한다.

### 0.5 commit delta 경로 분류

`AUDIT_HEAD..RUN_HEAD`에서 변경된 모든 경로를 다음 범주로 분류하고 TSV로 기록한다.

#### A. `LOCKED_CONTRACT_DELTA` — 자동 승인 금지

```text
docs/research/phase0a/**
```

하나라도 있으면 `BLOCKED_BASELINE_LOCKED_CONTRACT_DELTA`로 종료한다.

#### B. `DEPENDENCY_OR_VENDOR_DELTA` — 자동 승인 금지

```text
pyproject.toml
uv.lock
requirements*.txt
requirements*.lock
third_party/**
```

하나라도 있으면 현재 실행환경과 audit 가정이 달라졌을 수 있으므로
`BLOCKED_BASELINE_DEPENDENCY_DELTA_REAUDIT_REQUIRED`로 종료한다.

#### C. `GATE_A_TARGET_HISTORY_COLLISION` — 자동 승인 금지

```text
bench/estimators/__init__.py
bench/estimators/mekf.py
tests/test_mekf_conventions.py
tests/test_mekf_core.py
docs/research/phase1a/P1A_IMPLEMENTATION_CONTRACT.md
docs/research/phase1a/P1A_TEST_MATRIX.md
experiments/phase1a/reports/P1A_MATH_VALIDATION_REPORT.md
```

commit history에 하나라도 이미 들어 있으면 새 구현 provenance를 분리할 수 없으므로
`BLOCKED_BASELINE_TARGET_HISTORY_COLLISION`으로 종료한다.

#### D. `VISUALIZATION_ISOLATED_DELTA`

§V0.2의 visualization-isolated 기본 패턴과 optional manifest에 해당하는 경로다.

#### E. `SHARED_CRITICAL_DELTA`

§V0.2의 shared-critical 경로다. 존재 자체만으로 즉시 중단하지는 않지만, 아래 focused review를 통과해야 한다.

#### F. `PHASE1A_DOC_OR_PROMPT_DELTA`

```text
docs/research/phase1a/**
```

단, Gate A target history collision과 locked Phase 0A 문서는 제외한다.

#### G. `OTHER_BASELINE_DELTA`

위 범주에 속하지 않는 나머지 경로다. 존재 자체만으로 즉시 중단하지 않지만 변경 목록을 전부 보고하고,
Gate A의 독립 구현 경계와 충돌하지 않는지 focused review를 수행한다.

### 0.6 FAST_FORWARD focused review

Case B에서는 changed path와 관련 diff hunk를 읽고 다음을 확인한다.

1. 새 Gate A 수학 source of truth가 아직 존재하지 않는다.
2. legacy `bench/models/basilisk_mrp_ekf.py`를 새 MEKF로 이름만 바꾸거나 혼합한 변경이 없다.
3. 현재 변경 때문에 Gate A가 runner, metric, generator, visualization module을 import해야 할 이유가 생기지 않았다.
4. `bench/estimators/mekf.py`를 독립 NumPy/SciPy core로 추가할 수 있는 package 경계가 유지된다.
5. Phase 0A locked convention과 상충하는 새 project-wide convention이 도입되지 않았다.
6. 기존 selected regression test 파일/기대값이 commit delta에서 변경됐다면, 그 변경을 상세히 기록하고
   구현 전 baseline regression 결과를 최종 승인 조건으로 둔다.
7. visualization/shared-critical delta는 이번 Gate A가 수정할 파일과 겹치지 않는다.
8. `OTHER_BASELINE_DELTA`가 Gate A target, dependency, locked contract 또는 truth leakage를 유발하지 않는다.

focused review의 provisional 판정은 다음 중 하나다.

```text
EXACT_MATCH_PROVISIONAL_ACCEPT
FAST_FORWARD_VIZ_ONLY_PROVISIONAL_ACCEPT
FAST_FORWARD_REVIEWED_PROVISIONAL_ACCEPT
BLOCKED_BASELINE_RECONCILIATION
```

`SHARED_CRITICAL_DELTA`나 `OTHER_BASELINE_DELTA`가 있더라도 위 1~8을 충족하면
**Gate A pure math/core에 한정해** provisional accept할 수 있다. 이는 runner/generator/metric 통합 승인이 아니다.
판단 근거가 부족하면 진행을 추측하지 말고 `BLOCKED_BASELINE_RECONCILIATION`으로 종료한다.

### 0.7 RUN_HEAD 잠금

`RUN_HEAD`는 이번 실행 전체에서 변하면 안 된다. 다음 checkpoint마다 `git rev-parse HEAD`를 다시 확인한다.

1. baseline reconciliation 직후
2. snapshot 직후
3. 구현 전 baseline regression 직전/직후
4. approval marker 생성 직후
5. agent-owned 파일 생성 직후
6. 신규 Gate A test 직전/직후
7. legacy regression 직전/직후
8. 최종 보고서 작성 직전

한 번이라도 HEAD가 바뀌면 visualization-only commit이라도 provenance가 끊긴 것이므로 즉시:

```text
BLOCKED_HEAD_CHANGED_DURING_RUN
```

로 종료한다. 실행 중에는 다른 workstream도 commit, merge, rebase, reset을 수행하면 안 된다.
visualization-isolated **파일 수정**은 §V0 규칙에 따라 허용될 수 있지만 HEAD 이동은 허용하지 않는다.

### 0.8 snapshot → baseline regression → 최종 자동 승인 순서

provisional accept된 경우 다음 순서를 반드시 지킨다.

```text
read-only reconciliation in /tmp
→ §3 recoverable dirty snapshot
→ path fingerprint
→ §5 구현 전 baseline regression
→ RUN_HEAD 재확인
→ 최종 Gate A baseline 승인 marker/report 생성
→ Gate A 구현 시작
```

snapshot 또는 baseline regression이 실패하면 approval marker를 만들거나 갱신하지 않는다.

### 0.9 승인 marker 자동 생성/갱신

snapshot이 완전하고 구현 전 baseline regression exit code가 0이면 다음 파일을 생성하거나 갱신할 수 있다.

```text
docs/research/phase1a/approvals/01_AUDIT_APPROVED.md
```

이 파일은 **commit하지 않은 working-tree provenance marker**다. 최소 필드:

```yaml
decision: AUTO_APPROVED_FOR_GATE_A_ONLY
approved_branch: <CURRENT_BRANCH>
baseline_commit: <RUN_HEAD>
audit_reference_commit: <AUDIT_HEAD>
baseline_relation: <EXACT_MATCH|FAST_FORWARD_DESCENDANT>
delta_classification: <EXACT|VIZ_ONLY|REVIEWED_SHARED_OR_OTHER>
approval_scope: Gate A pure MEKF math/core only
baseline_regression: PASS
snapshot_path: <path>
approved_at_utc: <timestamp>
```

기존 marker가 있으면 snapshot에 이전 내용/hash가 보존됐는지 확인한 뒤, 현재 `RUN_HEAD` 기준으로만 갱신한다.
marker를 `git add`, `git commit`, `git push`하지 않는다.

동시에 timestamped reconciliation report를 생성한다.

```text
experiments/phase1a/reports/P1A_BASELINE_RECONCILIATION_<UTC_TIMESTAMP>.md
```

최소 포함:

- audit/current branch와 HEAD
- commit relation
- commit log/stat/name-status
- 경로 분류 개수와 전체 목록 또는 ledger 경로
- focused review 결과
- shared-critical/other delta 영향 판단
- snapshot completeness
- baseline regression exact command/result
- prior approval marker 유무와 처리
- 최종 승인 범위
- RUN_HEAD lock 상태

### 0.10 최종 baseline 판정

다음 중 하나를 명시적으로 남긴다.

```text
BASELINE_APPROVAL=EXACT_MATCH_APPROVED
BASELINE_APPROVAL=FAST_FORWARD_VIZ_ONLY_APPROVED
BASELINE_APPROVAL=FAST_FORWARD_REVIEWED_GATE_A_ONLY
BASELINE_APPROVAL=BLOCKED_<REASON>
```

앞의 세 승인 상태 중 하나이고 approval marker의 `baseline_commit`이 현재 `RUN_HEAD`와 정확히 같을 때만
§8 Gate A 구현으로 넘어간다.

working tree가 dirty인 것 자체는 중단 사유가 아니다. 다만 §3 snapshot, §V0 isolation,
§13 종료 provenance를 모두 통과해야 한다.

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

§0의 read-only baseline reconciliation을 `/tmp`에서 완료한 뒤, repository 안의 source/test/report/approval 파일을
만들거나 갱신하기 전에 현재 dirty 상태를 snapshot한다.

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
   단, `docs/research/phase1a/approvals/01_AUDIT_APPROVED.md`는 §0을 통과한 경우에만
   snapshot에 보존된 이전 상태에서 현재 RUN_HEAD 기준 marker로 갱신할 수 있다.
2. 실행 전 clean이었던 tracked 파일이 허용 목록 밖에서 새로 modified/deleted되지 않았다.
3. 실행 전 없던 untracked 파일은 허용된 output/log/snapshot 경로에만 생겼다.

한 항목이라도 어기면 자동 복원하지 말고 `BLOCKED_UNINTENDED_CHANGE`로 종료하며,
변경 경로와 before/after hash를 보고한다.

---

## 4. Target collision 검사

아래 §6.1~§6.3의 **Gate A core/test/contract 고정 target**이 실행 전에 이미 존재하는지 확인한다.

- 고정 target이 하나라도 이미 존재하면 내용을 덮어쓰거나 합치지 않는다.
- existing target의 path, Git 상태, size, SHA-256을 기록한다.
- 구현을 시작하지 않고 `BLOCKED_TARGET_EXISTS`로 종료한다.

다음은 고정 target collision 검사에서 제외한다.

```text
docs/research/phase1a/approvals/01_AUDIT_APPROVED.md
experiments/phase1a/reports/P1A_BASELINE_RECONCILIATION_*.md
experiments/phase1a/agent_logs/02_math_core_*
experiments/phase1a/preflight_snapshots/02_math_core_*/**
```

approval marker는 §0에 따라 snapshot 후 생성/갱신할 수 있다. timestamped report/log/snapshot은 동일 이름이
충돌하면 새 UTC timestamp를 사용한다.

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

### 6.4 baseline reconciliation/approval artifact

```text
docs/research/phase1a/approvals/01_AUDIT_APPROVED.md
experiments/phase1a/reports/P1A_BASELINE_RECONCILIATION_*.md
```

approval marker는 §0 조건을 통과한 경우에만 생성 또는 갱신하며 stage/commit하지 않는다.

### 6.5 실행 provenance/log artifact

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
2. repository root, branch, RUN_HEAD
3. audit/current HEAD 관계와 baseline reconciliation 판정
4. commit delta 분류 및 focused review 결과
5. approval marker 및 timestamped reconciliation report 경로
6. dirty snapshot 경로와 snapshot completeness
7. Python/NumPy/SciPy/pytest version
8. 구현 전 baseline regression 결과
9. 생성·갱신한 파일 목록
10. agent-only diff/stat
11. 신규 test exact command/result
12. 구현 후 legacy regression exact command/result
13. B1/B3/B4/B5/B6별 evidence
14. SPD/non-SPD safety evidence
15. numerical correction ledger
16. pre-existing dirty path 보호 검증 결과
17. visualization/shared-critical/HEAD 안정성 결과
18. unresolved issue
19. 다음 단계에서 허용할 범위 제안
20. **Prompt 3 또는 Gate B를 실제 실행하지 않았음**을 명시

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
baseline reconciliation approved for Gate A:            PASS/FAIL
approval marker baseline == RUN_HEAD:                   PASS/FAIL
RUN_HEAD unchanged throughout execution:                PASS/FAIL
pre-existing non-visualization dirty files unchanged:   PASS/FAIL
visualization-isolated paths untouched by this agent:   PASS/FAIL
shared-critical paths stable during run:                PASS/FAIL
concurrent visualization delta:                         NONE/DETECTED
new changes confined to allowlist:                      PASS/FAIL
target files newly created only:                        PASS/FAIL
baseline regression preserved:                          PASS/FAIL
Gate A tests:                                            PASS/FAIL
```

하나라도 FAIL이면 최종 Gate A를 PASS로 표시하지 않는다.

---

## 14. 중단 조건

다음 중 하나라도 발생하면 범위를 확장하거나 우회하지 말고 중단한다.

```text
BLOCKED_MISSING_INPUT
BLOCKED_AUDIT_COMMIT_UNAVAILABLE
BLOCKED_BASELINE_BRANCH_MISMATCH
BLOCKED_BASELINE_DIVERGED_REAUDIT_REQUIRED
BLOCKED_BASELINE_DELTA_INVALID
BLOCKED_BASELINE_LOCKED_CONTRACT_DELTA
BLOCKED_BASELINE_DEPENDENCY_DELTA_REAUDIT_REQUIRED
BLOCKED_BASELINE_TARGET_HISTORY_COLLISION
BLOCKED_BASELINE_RECONCILIATION
BLOCKED_HEAD_CHANGED_DURING_RUN
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

1. 감사 HEAD와 현재 RUN_HEAD의 관계가 §0에서 검증됨
2. commit delta가 분류되고 Gate A 한정 baseline reconciliation이 승인됨
3. recoverable dirty snapshot 생성 완료
4. 구현 전 baseline regression exit code 0
5. approval marker의 `baseline_commit`이 RUN_HEAD와 정확히 일치
6. RUN_HEAD가 실행 내내 불변
7. target collision 없음
8. 허용된 Gate A 구현/시험/문서 및 baseline/provenance 파일만 생성·갱신
9. B1/B3/B4/B5/B6 및 convention tests 전부 통과
10. deliberate non-SPD fail-loud test 통과
11. 구현 후 legacy regression exit code 0
12. pre-existing non-visualization dirty/untracked 파일 hash 무변경
13. shared-critical 경로가 실행 내내 불변
14. visualization-isolated 경로를 이번 agent가 수정하지 않음; 별도 concurrent delta는 ledger에 기록
15. 허용 목록 밖 신규 agent 변경 없음
16. implementation/test 문서 3개와 baseline reconciliation report 완성
17. commit, stage, push를 수행하지 않음
18. Gate B/Prompt 3을 시작하지 않음

---

## 16. 최종 응답 형식

최종 agent 응답은 다음 순서를 따른다.

### Status

```text
PASS / FAIL / BLOCKED_<REASON>
```

### Baseline reconciliation and dirty-tree protection

- repository root
- audit branch / HEAD
- current branch / RUN_HEAD
- baseline relation: exact / fast-forward / diverged
- commit delta classification counts
- focused review decision
- approval marker path and baseline commit
- reconciliation report path
- snapshot path
- initial dirty counts
- RUN_HEAD stability result
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
Baseline reconciliation: PASS/FAIL
Baseline approval: EXACT_MATCH_APPROVED / FAST_FORWARD_VIZ_ONLY_APPROVED /
                   FAST_FORWARD_REVIEWED_GATE_A_ONLY / BLOCKED
RUN_HEAD stability: PASS/FAIL
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



---

## 17. 이 프롬프트가 해결하려는 이전 중단의 명시적 처리

이전 실행이 다음처럼 중단됐더라도:

```text
audit HEAD:   ee862a2acc368fb631c45ef0b33a8f4feb5c28c0
current HEAD: d92cd0ce590f1ebfdf3edb756064d94cba551174
approval file: 없음
status: BLOCKED_BASELINE_MISMATCH
```

이번 프롬프트는 곧바로 중단하지 않는다. `ee862...`와 현재 HEAD의 ancestor 관계와 commit delta를 조사하고,
locked/dependency/target 충돌이 없으며 focused review, snapshot, 구현 전 baseline regression을 통과하면
현재 HEAD를 `AUTO_APPROVED_FOR_GATE_A_ONLY`로 기록한 뒤 Gate A를 수행한다.

단, 현재 HEAD가 감사 HEAD의 descendant가 아니거나 Phase 0A locked 문서, dependency/vendor, Gate A target history가
변경됐다면 자동 승인을 하지 않는다. 이 제한을 우회하거나 approval marker만 임의 작성해서 진행하지 않는다.
