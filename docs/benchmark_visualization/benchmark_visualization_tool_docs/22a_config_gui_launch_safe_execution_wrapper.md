# Config GUI Launch — Safe Execution Wrapper

이 문서는 기존 `22_config_gui_launch_implementation_prompt.md`의 **기능 요구사항은 그대로 유지**하면서, 실행·보존·commit 정책만 더 안전하게 바꾸는 래퍼다.

사용자 결정:

- 구현 중간에는 commit하지 않는다.
- 전체 기능·테스트·문서가 완료된 뒤 한 번만 최종 commit한다.
- `/home/dss-pc-05/bench`의 기존 dirty working tree는 절대 건드리지 않는다.
- `/tmp/bench-wc-tranche`는 현재 인증된 write-control baseline 보존용으로 남긴다.
- uncommitted 작업 유실을 막기 위해 **영속적인 새 worktree + Git 외부 snapshot**을 사용한다.

이 결정은 22번 문서의 “시작 전에 instruction이 이미 tracked/committed여야 한다”는 조건을 대체한다.

---

## 1. 경로와 역할

### Git common repository

```text
/home/dss-pc-05/bench/.git
```

### User working tree — 보존 전용

```text
/home/dss-pc-05/bench
```

여기서는 source 수정, test 실행, artifact 생성, stage, commit을 하지 마라.

### Certified source worktree — baseline 확인 전용

```text
/tmp/bench-wc-tranche
branch: benchmark-viz/write-control
```

이번 Config GUI 구현을 여기서 직접 진행하지 마라.

### Persistent implementation worktree

```text
/home/dss-pc-05/bench-worktrees/config-gui-launch
branch: benchmark-viz/config-gui-launch
```

이번 tranche의 구현·테스트·문서·최종 commit은 여기서 수행한다.

---

## 2. Certified baseline 확인

`/tmp/bench-wc-tranche`에서 다음을 기록하라.

```bash
pwd
git rev-parse --show-toplevel
git rev-parse --git-common-dir
git branch --show-current
git rev-parse HEAD
git status --short
git log --oneline --decorate -15
git worktree list
git submodule status --recursive
```

필수:

```text
branch == benchmark-viz/write-control
```

22번 prompt 외에 예상하지 못한 uncommitted source/test 변경이 있으면 중단하고 보고하라.

현재 baseline에는 최소한 다음이 있어야 한다.

- write-control fault/restart certification
- local-only write mode
- Stop/Resume POST API
- Dash Stop/Resume controls
- Playwright write workflow
- Checkpoint v2
- KNet/Split worker-level exact resume
- full pytest green

없으면 다음으로 종료한다.

```text
WRONG_CHECKOUT_OR_LOST_WRITE_CONTROL_BACKEND
```

---

## 3. Instruction provenance — commit 전에도 재현 가능하게 고정

승인된 instruction:

```text
docs/benchmark_visualization/benchmark_visualization_tool_docs/
22_config_gui_launch_implementation_prompt.md
```

승인된 SHA-256:

```text
844c5cfeda546c4bbe23e3c2cc6258f71c9a084b4b934006121fa13317aba865
```

가능한 source:

```text
/home/dss-pc-05/bench/docs/benchmark_visualization/benchmark_visualization_tool_docs/22_config_gui_launch_implementation_prompt.md
/tmp/bench-wc-tranche/docs/benchmark_visualization/benchmark_visualization_tool_docs/22_config_gui_launch_implementation_prompt.md
```

시작 조건은 다음뿐이다.

- 최소 한 source 파일이 존재
- SHA-256이 승인값과 일치
- 두 copy가 모두 있으면 서로 byte-identical

Git tracked/committed 상태는 시작 조건이 아니다.

새 persistent worktree에 복사한 뒤 다음 시점마다 hash를 재검증하라.

- production 변경 전
- API 구현 완료 후
- UI 구현 완료 후
- full test 전
- 최종 commit 직전

hash가 바뀌면 즉시 중단한다.

최종 report/summary에는 다음을 기록한다.

```yaml
instruction_document:
  path: docs/benchmark_visualization/benchmark_visualization_tool_docs/22_config_gui_launch_implementation_prompt.md
  tracked_at_start: false
  user_authorized_untracked_instruction: true
  sha256: 844c5cfeda546c4bbe23e3c2cc6258f71c9a084b4b934006121fa13317aba865
  included_in_final_commit: true
```

---

## 4. 새 영속 branch/worktree 생성

먼저 존재 여부를 확인한다.

```bash
git -C /home/dss-pc-05/bench worktree list
git -C /home/dss-pc-05/bench branch --list benchmark-viz/config-gui-launch
test -e /home/dss-pc-05/bench-worktrees/config-gui-launch
```

둘 다 없으면:

```bash
mkdir -p /home/dss-pc-05/bench-worktrees

git -C /home/dss-pc-05/bench worktree add \
  -b benchmark-viz/config-gui-launch \
  /home/dss-pc-05/bench-worktrees/config-gui-launch \
  benchmark-viz/write-control
```

branch/path가 이미 존재하면 삭제하거나 재생성하지 마라. ancestry, current branch, status를 확인하고 안전하게 재사용 가능한 경우에만 진행한다.

새 worktree에서:

```bash
cd /home/dss-pc-05/bench-worktrees/config-gui-launch
git submodule update --init --recursive
```

third-party tracked source는 수정하지 마라.

---

## 5. Prompt 복사와 preflight backup

새 worktree에 prompt를 복사하고 SHA-256을 확인한다.

```bash
mkdir -p docs/benchmark_visualization/benchmark_visualization_tool_docs
```

복사 전후 hash가 승인값과 동일해야 한다.

Git 밖의 backup root:

```text
/home/dss-pc-05/bench-backups/config-gui-launch/<timestamp>/
```

이 경로는 repository/worktree 내부가 아니어야 한다.

Preflight snapshot에 다음을 저장한다.

```text
metadata.txt
git_status.txt
git_log.txt
git_worktrees.txt
submodules.txt
instruction.sha256
instruction_copy/
baseline_tests/
RECOVERY.md
```

`metadata.txt`에는 Git common dir, source worktree, persistent worktree, source tip, new branch/tip, Python version, prompt source/hash, timestamp를 기록한다.

새 worktree에서 변경 전:

```bash
python -m pytest --collect-only -q
python -m pytest -q
```

최근 기준은 약 `582 collected / 581 passed / 1 skipped`이지만 실제 결과를 baseline으로 기록한다. Baseline이 깨져 있으면 구현을 시작하지 않는다.

---

## 6. 중간 commit 대신 external snapshot 사용

사용자는 최종 완료 후 한 번만 commit하기를 원한다. 따라서 최소 다음 milestone마다 외부 snapshot을 만든다.

```text
S0 구현 시작 전
S1 preset catalog + schema/validation API 완료
S2 durable launch API 완료
S3 Dash New Run workflow 완료
S4 unit/integration tests 완료
S5 Playwright/E2E 완료
S6 최종 commit 직전
```

각 snapshot에 다음을 저장한다.

```text
git_status.txt
tracked.patch
staged.patch
untracked_files.txt
untracked.tar.gz
instruction.sha256
file_hashes.sha256
test_summary.txt
```

개념 명령:

```bash
git status --short > git_status.txt
git diff --binary > tracked.patch
git diff --cached --binary > staged.patch
git ls-files --others --exclude-standard > untracked_files.txt
```

Untracked 파일은 NUL-safe archive로 저장한다. Snapshot 생성 후 archive/patch가 읽히는지 확인한다.

`RECOVERY.md`에는 baseline worktree 재생성, tracked patch 적용, untracked archive 복원, prompt hash 검증, full test 실행 절차를 작성한다.

중간 Git commit은 만들지 않는다.

---

## 7. 실제 구현 범위

새 persistent worktree에서 `22_config_gui_launch_implementation_prompt.md`의 기능·보안·테스트·산출물 요구사항 전체를 수행한다.

핵심:

```text
Preset catalog
→ typed schema descriptor
→ Form / Raw YAML 편집
→ validation + ResolvedRunSpec preview
→ structural/operational diff
→ durable LAUNCH_RUN action
→ immutable run allocation
→ WorkerManager launch
→ Dash /new-run workflow
→ KNet/Split/model-based E2E
→ Playwright New Run→Launch→Stop→Resume
```

22번 문서의 다음 조건만 이 wrapper가 대체한다.

```text
“구현 시작 전에 22번 instruction이 already tracked/committed여야 한다.”
```

나머지는 그대로 유지한다.

---

## 8. 구현 중 Git 안전 규칙

최종 commit 전까지 허용:

- source/test/docs 편집
- external snapshot
- git diff/status/log
- test 실행

금지:

- `git commit`
- `git add -A`
- `git add .`
- `git commit -am`
- `git clean`
- `git reset --hard`
- unrelated path checkout/revert
- `git stash`
- rebase
- branch force move
- remote push

최종 검토 전에는 stage하지 마라.

`/home/dss-pc-05/bench`와 `/tmp/bench-wc-tranche`는 수정하지 마라.

---

## 9. Agent 중단 시 처리

작업을 완료하지 못하면 최신 external snapshot을 만들고 다음을 출력한다.

- persistent worktree
- branch
- baseline commit
- current status
- latest snapshot path
- completed scope
- remaining scope
- test status
- final commit 없음

예상하지 못한 unrelated file이 새 worktree에 생기면 삭제하지 말고 origin을 확인하고 snapshot 후 중단한다.

Backup root 또는 persistent worktree에 쓸 수 없으면 구현을 시작하지 않는다.

---

## 10. 최종 acceptance gate

22번 prompt의 모든 필수 gate를 실행한다.

최소:

```text
preset catalog safety
single typed config source of truth
Form/YAML round-trip
validation/preview
unsafe config/path rejection
durable launch API
launch idempotency/restart
same config + different request → unique immutable runs
CLI/GUI resolved-config parity
KNet GUI launch E2E
Split GUI launch E2E
model-based baseline launch
existing Stop/Resume regression
Playwright New Run→Launch→Stop→Resume
full pytest
28 init-provenance
third-party tracked diff empty
prompt hash unchanged
```

필수 gate가 실패하면 최종 commit을 만들지 않는다.

판정:

```text
READY_FOR_GPU_QUEUE_AND_EXECUTION_POLICY_TRANCHE
READY_AFTER_SPECIFIC_FIXES
NOT_READY
```

`READY_FOR_GPU_QUEUE_AND_EXECUTION_POLICY_TRANCHE`일 때만 최종 commit을 만든다.

---

## 11. 최종 한 번의 commit

### 11.1 S6 snapshot

최종 commit 전에 S6 snapshot을 먼저 생성한다.

### 11.2 Diff inventory

다음을 작성한다.

```text
final_changed_files.txt
final_diffstat.txt
final_untracked_files.txt
```

파일을 다음으로 분류한다.

```text
production
tests
docs
instruction
generated/verification
unrelated
```

`generated/verification/unrelated`는 stage하지 않는다.

### 11.3 Explicit staging

허용 파일만 명시적 path로 stage한다. 22번 prompt도 포함한다.

금지:

```text
git add -A
git add .
git commit -am
```

### 11.4 Staged audit

```bash
git diff --cached --check
git diff --cached --stat
git diff --cached --name-status
```

확인:

- user working tree 파일 없음
- `/tmp` artifact 없음
- backup archive 없음
- test DB/run/log/screenshot 없음
- third-party generated file 없음
- prompt 포함
- production/test/docs만 포함

### 11.5 Final commit

권장 message:

```text
feat(benchmark-viz): add config GUI and benchmark launch workflow
```

Commit 후:

```bash
git status --short
git rev-parse HEAD
```

남은 untracked/generated 파일은 자동 삭제하지 말고 분류해 보고한다.

---

## 12. Commit 후 추가 보존

최종 commit 뒤 external backup root에 Git bundle을 만든다.

```text
config-gui-launch-<commit>.bundle
```

포함 branch:

```text
benchmark-viz/config-gui-launch
```

검증:

```bash
git bundle verify <bundle>
```

Remote push는 하지 않는다.

---

## 13. 최종 provenance 기록

최종 report와 summary JSON에 다음을 기록한다.

```yaml
instruction_document:
  path: docs/benchmark_visualization/benchmark_visualization_tool_docs/22_config_gui_launch_implementation_prompt.md
  tracked_at_start: false
  user_authorized_untracked_instruction: true
  sha256_at_start: 844c5cfeda546c4bbe23e3c2cc6258f71c9a084b4b934006121fa13317aba865
  sha256_before_commit: 844c5cfeda546c4bbe23e3c2cc6258f71c9a084b4b934006121fa13317aba865
  included_in_final_commit: true

worktree_policy:
  source_certified_worktree: /tmp/bench-wc-tranche
  implementation_worktree: /home/dss-pc-05/bench-worktrees/config-gui-launch
  user_working_tree_untouched: true
  intermediate_git_commits: 0
  external_snapshots: [...]
  final_bundle: ...
```

---

## 14. Explicit exclusions

22번 prompt의 exclusions를 유지한다.

이번 tranche에서는 구현하지 마라.

- sweep/grid-search GUI
- multi-run batch launch
- GPU queue/scheduler/lease enforcement
- shared GPU
- Force terminate
- Warm start
- Evaluate checkpoint
- persistent draft library
- arbitrary config upload
- authentication/multi-user
- remote worker
- GPU/AMP/multi-worker exact resume
- Adaptive/MAML/ME-Split GUI launch
- WebSocket/SSE

---

## 15. 완료 시 터미널 출력

1. Git common repository
2. user working tree untouched 여부
3. certified source worktree와 baseline
4. persistent implementation worktree
5. new branch
6. instruction source/path/SHA-256
7. external backup root
8. milestone snapshot 목록
9. implementation files summary
10. preset/schema/validation 결과
11. launch API/idempotency/restart 결과
12. CLI parity
13. KNet GUI launch E2E
14. Split GUI launch E2E
15. model-based baseline
16. Playwright workflow
17. existing Stop/Resume regression
18. full pytest
19. 28 init-provenance
20. third-party diff
21. final verdict
22. final commit/hash
23. Git bundle path/verify 결과
24. reports/summary/artifact paths

이제 이 safe-execution wrapper를 적용하여,
`22_config_gui_launch_implementation_prompt.md`의 기능 요구사항 전체를 수행하라.
