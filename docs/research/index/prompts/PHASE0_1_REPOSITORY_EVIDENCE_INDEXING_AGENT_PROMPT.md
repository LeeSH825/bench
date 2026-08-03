# Phase 0–1 Repository Evidence Indexing Agent Prompt
## Canonical File Map, Numeric Evidence Catalog, and Phase 2 Lookup Handoff

당신은 `/home/dss-pc-05/bench` 저장소에서 **Phase 0–1 연구 자산을 인덱싱하는 문서·검증 agent**다.

이번 작업의 목적은 연구 내용을 새로 해석하거나 코드를 구현하는 것이 아니다.

목적은 이후 새로운 Phase 2 대화에서 다음 질문이 나왔을 때:

```text
그 수치는 어느 보고서에 있는가?
현재 canonical 결정은 어느 파일인가?
이 구현은 실제로 어느 source/test/config에 있는가?
이 결론은 어떤 실험 결과가 근거인가?
이전 결론 중 무엇이 superseded 되었는가?
agent에게 어떤 파일을 열어보라고 요청해야 하는가?
```

저장소의 정확한 파일과 locator를 즉시 찾을 수 있도록
**사람용 navigation index와 machine-readable evidence index**를 만드는 것이다.

이 작업은 새로운 Gate나 Phase가 아니다.
Phase 0–1의 frozen 결과를 수정·재튜닝·재해석하지 마라.
Phase 2 설계나 구현을 시작하지 마라.

---

# 1. 최우선 입력 문서

다음을 처음부터 끝까지 읽어라.

## Phase 0–1 handoff

```text
docs/research/phase1b/AI_ADCS_PHASE0_1_MASTER_SUMMARY_AND_PHASE2_HANDOFF.md
```

위 파일이 저장소에 없다면 같은 이름의 파일을 repository 전체에서 찾아라.
유일한 동일 이름 파일이 발견되면 그 실제 경로를 기록하고 읽어라.
여러 개가 발견되면 자동 선택하지 말고 validation report에 ambiguity로 기록하라.

## Updated P1 Exit

```text
experiments/phase1b/reports/P1_EXIT_REVIEW_UPDATED.md
experiments/phase1b/reports/P1_EXIT_REVIEW.md
```

Updated review가 현재 decision source이며,
원본 review는 historical predecessor로 표시해야 한다.

---

# 2. 반드시 조사할 저장소 범위

다음 root를 재귀적으로 조사하라.

```text
docs/research/phase0a/
docs/research/phase1a/
docs/research/phase1b/

experiments/phase1a/reports/
experiments/phase1b/reports/
experiments/phase1a/manifests/
experiments/phase1b/manifests/
experiments/phase1a/results/
experiments/phase1b/results/

bench/estimators/
bench/tasks/generator/
bench/metrics/
bench/models/
bench/experiments/
bench/configs/

tests/
bench/tests/
```

다음 provenance root는 **필요한 command/reproducibility locator를 찾을 때만** 조사하라.

```text
experiments/phase1a/agent_logs/
experiments/phase1b/agent_logs/
experiments/phase1a/preflight_snapshots/
experiments/phase1b/preflight_snapshots/
```

대규모 binary artifact, NPZ, tar, patch는 무분별하게 열지 마라.
manifest/report가 가리키는 exact key/array/file을 확인할 때만 읽어라.

---

# 3. Source-of-truth 우선순위

파일명이나 최신 수정 시간만 보고 canonical source를 판단하지 마라.

질문 유형별 우선순위를 다음처럼 적용하라.

## 3.1 현재 연구 결정

```text
updated final review/approval
> final validation report
> specialized final report
> earlier review/approval
> handoff/master summary
> command log
```

예:

```text
P1 Exit current decision:
P1_EXIT_REVIEW_UPDATED.md

historical decision:
P1_EXIT_REVIEW.md
```

## 3.2 정확한 실험 수치

```text
machine-readable frozen result/summary/manifest
> specialized report
> validation report
> test matrix
> handoff/master summary
```

단, machine-readable result가 incomplete/partial이면 final report보다 우선하지 않는다.

## 3.3 구현 위치와 API

```text
actual source
> implementation contract
> test
> validation report
> summary
```

## 3.4 재현 명령

```text
final command log
> validation report command section
> prompt example
```

## 3.5 수학·frame·schema convention

```text
final frozen contract/source
> executable proof/test matrix
> validation report
> earlier contract wording
```

Erratum이나 superseded wording이 있으면 반드시 표시하라.

---

# 4. 핵심 인덱싱 원칙

## 4.1 단순 파일 목록을 만들지 마라

각 항목은 최소 다음을 연결해야 한다.

```text
질문 또는 주장
→ 현재 canonical source
→ supporting evidence
→ 구현 source/test/config
→ exact locator
→ status/currentness
```

## 4.2 숫자는 scope 없이 기록하지 마라

모든 numeric fact에는 최소 다음 필드를 붙여라.

```text
value
unit
metric definition
phase/step
scenario/condition
policy
dataset/split
N
duration 또는 horizon
transient/settled/window 범위
canonical source path
source locator
machine-readable key/path if available
```

예를 들어 다음 값들은 같은 이름의 NEES라도 서로 다른 scope이므로 충돌로 처리하면 안 된다.

```text
MAIN-FUSION original settled F-BASE NEES/DOF = 1.873
closure independent stationary F-BASE NEES/DOF = 1.418
closure F-CALIBRATED-v1 NEES/DOF = 1.021
```

각각 dataset, seed namespace, partition과 policy를 구분하라.

## 4.3 Historical/superseded를 삭제하지 마라

다음 status 중 하나를 사용하라.

```text
current
frozen
historical
superseded
diagnostic_only
non_deployable_oracle
sensitivity_comparator
stationary_only_comparator
deferred
```

예:

```text
P1_EXIT_REVIEW.md:
  historical

P1_EXIT_REVIEW_UPDATED.md:
  current

F-TUNED:
  frozen + sensitivity_comparator

F-CALIBRATED-v1:
  frozen + stationary_only_comparator
```

## 4.4 Summary는 exact-number source가 아니다

`AI_ADCS_PHASE0_1_MASTER_SUMMARY_AND_PHASE2_HANDOFF.md`는 navigation source다.
정확한 수치나 최종 판정은 해당 final report/result까지 추적하라.

---

# 5. 필수 topic taxonomy

최소 다음 topic을 모두 인덱싱하라.

## Phase 0

```text
research objective/title
state definition
quaternion/frame convention
truth/sensor/estimator boundary
sensor role
context/oracle contract
Phase roadmap
```

## Phase 1A

```text
MEKF implementation
exact-pi/q-sign policy
state immutability
typed UNIT-ST event schema
generator identity amendment
Basilisk sigma_BN -> q_NB proof
omega_BN_B proof
project-owned gyro/ST model
canonical geodesic/bias/NIS/NEES metrics
D1 adapter
CP4 runner/cache integration
```

## Phase 1B Step 1

```text
F-BASE
F-TUNED
Q/R mismatch policies
C1
C2
C3
C5
long horizon
paired Monte Carlo
H1-H4 preliminary decisions
```

## Phase 1B Step 2

```text
fusion schema
gyro-mag-sun-ST event order
magnetometer model
sun tangent model/validity skip
MAIN-FUSION
STRESS-MAG
C4
partial/full/wrong-side oracle
sensor-specific NIS
initial P1 Exit
```

## P1 Exit Closure

```text
initial transient
full/attitude/bias NEES decomposition
whitened error
attitude-bias cross covariance
F-CALIBRATED-v1
independent stationary confirmation
C4 confirmation failure
updated P1 Exit
frozen baseline matrix
Phase 2 entry conditions
```

## Repository operation/provenance

```text
accepted dirty-tree policy
current source-of-truth paths
test suites and counts
command logs
result/manifests roots
cache/result replay commands
```

---

# 6. 반드시 생성할 파일

모든 파일은 다음 root 아래 생성하라.

```text
docs/research/index/phase0_1/
```

## 6.1 README.md

```text
docs/research/index/phase0_1/README.md
```

포함:

- 인덱스의 목적
- canonical source 우선순위
- 새 대화에서 사용하는 방법
- exact value를 찾는 절차
- historical/superseded 해석법
- machine-readable JSON 설명
- agent 요청 예시

## 6.2 SOURCE_OF_TRUTH_INDEX.md

```text
docs/research/index/phase0_1/SOURCE_OF_TRUTH_INDEX.md
```

각 topic별 표:

```text
topic ID
phase
question/claim
canonical source path
exact heading/key/locator
supporting source
implementation source
test/config
status
notes
```

## 6.3 QUESTION_TO_FILE_MAP.md

```text
docs/research/index/phase0_1/QUESTION_TO_FILE_MAP.md
```

실제 사용자가 물을 법한 질문을 최소 80개 작성하고,
각 질문을 exact source로 연결하라.

필수 질문 예:

```text
현재 P1 Exit 판정은?
F-TUNED 값은?
F-CALIBRATED 값은?
왜 F-CALIBRATED가 primary가 아닌가?
MAIN-FUSION settled NIS/NEES는?
C4 full oracle 개선량은?
C2에서 oracle Q가 attitude를 개선했는가?
C5 RMS match 값은?
Basilisk sigma_BN 변환식은?
왜 built-in star tracker를 쓰지 않았는가?
MEKF source는 어디인가?
runner integration source는 어디인가?
어떤 test가 q/-q invariance를 검증하는가?
Phase 1A smoke command는 어디에 있는가?
dirty-tree integrity evidence는 어디인가?
```

각 질문마다:

```text
first file to open
then supporting file
what to extract
warning/scope
```

를 기록하라.

## 6.4 NUMERIC_EVIDENCE_CATALOG.md

```text
docs/research/index/phase0_1/NUMERIC_EVIDENCE_CATALOG.md
```

최소 다음 숫자를 포함하라.

```text
all test counts
all Monte Carlo N
durations/runtime
F-TUNED scales
C5 alpha_R and RMS gap
long-horizon metrics
C2/C3 severity tables
C4 improvement percentages
MAIN-FUSION sensor NIS and NEES
STRESS-MAG weak/observable RMS
closure transient/marginal/full NEES
F-CALIBRATED scales
stationary confirmation values
C4 calibrated degradation
all final decisions
```

숫자를 handoff summary에서 복사하지 말고 canonical report/result와 대조하라.

## 6.5 FILE_PATH_CATALOG.md

```text
docs/research/index/phase0_1/FILE_PATH_CATALOG.md
```

분류:

```text
contracts
source
tests
configs
reports
results
manifests
command logs
integrity evidence
prompts/handoffs
```

각 파일에:

```text
path
role
phase
canonical 여부
current/historical
related topic IDs
expected reader
```

를 기록하라.

## 6.6 DECISION_AND_STATUS_LEDGER.md

```text
docs/research/index/phase0_1/DECISION_AND_STATUS_LEDGER.md
```

시간 순서:

```text
decision ID
date/time if available
decision
reason/evidence
source path
supersedes
superseded by
current effect
```

다음 관계를 반드시 명시하라.

```text
Gate 명칭 -> P1A checkpoint 재해석
initial P1 Exit CONDITIONAL_GO
closure 후 updated P1 Exit CONDITIONAL_GO
F-BASE primary 유지
F-TUNED role downgrade
F-CALIBRATED stationary-only
Phase 2 implementation not started
```

## 6.7 PHASE2_REPOSITORY_LOOKUP_HANDOFF.md

```text
docs/research/index/phase0_1/PHASE2_REPOSITORY_LOOKUP_HANDOFF.md
```

새 Phase 2 대화에 첨부할 수 있는 compact 문서다.

포함:

- “정확한 수치나 경로는 기억/summary로 답하지 말고 이 index를 통해 저장소 agent 조회”
- topic별 first-open files
- exact-number lookup template
- source-code lookup template
- experiment-reproduction lookup template
- historical decision lookup template
- Phase 2에서 자주 참조할 20개 canonical path
- frozen baseline matrix
- current P1 Exit
- known ambiguity/scope warnings

## 6.8 phase0_1_evidence_index.json

```text
docs/research/index/phase0_1/phase0_1_evidence_index.json
```

UTF-8, sorted-key canonical JSON으로 생성하라.

최소 schema:

```json
{
  "index_version": "...",
  "repository_root": "...",
  "current_phase_status": {
    "phase_0_1": "complete",
    "p1_exit": "CONDITIONAL_GO",
    "phase_2_implementation": "not_started"
  },
  "source_precedence": {},
  "topics": [
    {
      "id": "...",
      "phase": "...",
      "title": "...",
      "question_patterns": [],
      "status": [],
      "canonical_sources": [
        {
          "path": "...",
          "role": "...",
          "locator_type": "heading|line|json_key|npz_array|source_symbol",
          "locator": "..."
        }
      ],
      "supporting_sources": [],
      "implementation": [],
      "tests": [],
      "configs": [],
      "facts": [
        {
          "name": "...",
          "value": 0,
          "unit": "...",
          "scope": "...",
          "condition": "...",
          "policy": "...",
          "split": "...",
          "n": 0,
          "source_path": "...",
          "source_locator": "..."
        }
      ],
      "warnings": []
    }
  ],
  "files": [],
  "decisions": [],
  "unresolved_ambiguities": []
}
```

숫자가 문자열이어야 하는 special value가 아니면 number type으로 기록하라.

## 6.9 AGENT_LOOKUP_RECIPES.md

```text
docs/research/index/phase0_1/AGENT_LOOKUP_RECIPES.md
```

복사해 사용할 수 있는 agent 요청문을 제공하라.

최소 recipe:

1. exact numeric fact lookup
2. current decision lookup
3. source implementation lookup
4. test/evidence lookup
5. reproduce command lookup
6. compare two reported values with different scopes
7. find superseded decisions
8. trace claim -> report -> result JSON
9. find all files for one Phase
10. update index after future Phase work

## 6.10 Validation report

```text
experiments/research_index/reports/PHASE0_1_REPOSITORY_INDEX_VALIDATION_REPORT.md
```

포함:

```text
files scanned
topics indexed
questions indexed
numeric facts indexed
paths verified
missing paths
broken locators
duplicate IDs
conflicting facts
scope-resolved apparent conflicts
historical/superseded relations
unresolved ambiguities
dirty-tree integrity
final PASS/PARTIAL/BLOCKED
```

---

# 7. Locator 규칙

가능한 한 “파일만” 가리키지 말고 exact locator를 제공하라.

## Markdown

```text
heading path:
# Title > ## Section > ### Subsection

가능하면 line number range도 기록
```

Line number는 파일 수정 시 변할 수 있으므로 heading을 primary,
line range를 secondary로 사용한다.

## Source

```text
module path
class/function/constant symbol
```

예:

```text
bench/estimators/mekf.py::star_tracker_update
```

## JSON

```text
JSON key path
```

예:

```text
pilot_summary.conditions.C3_SEV.F_BASE.nees.normalized_mean
```

실제 key를 확인하고 기록하라. 예시를 그대로 만들지 마라.

## NPZ

```text
file path
array/key name
shape/dtype
```

## Command log

```text
command section heading
exact command block
```

---

# 8. Apparent conflict 처리

다음 유형을 자동 conflict로 판정하지 마라.

```text
다른 dataset
다른 seed namespace
다른 time partition
다른 policy
다른 event window
whole-horizon vs settled
validation vs test
historical vs updated decision
```

각 값에 scope를 부여한 뒤:

```text
true conflict
scope-resolved
superseded
unresolved
```

중 하나로 분류하라.

특히 다음을 검토하라.

```text
MAIN-FUSION original settled NEES 1.873
closure validation settled NEES 1.906
closure confirmation F-BASE NEES 1.418
F-CALIBRATED NEES 1.021
```

이 값들은 dataset/split이 다르므로 scope-resolved여야 한다.

---

# 9. Validation script

다음 stdlib-only validator를 생성하라.

```text
tools/research/validate_phase0_1_evidence_index.py
```

역할:

- JSON parse
- unique topic/decision IDs
- referenced repository path existence
- canonical source nonempty
- locator fields nonempty
- numeric fact source linkage
- current/historical decision consistency
- mandatory topic/question presence
- no absolute path inside portable index except repository root metadata
- sorted canonical JSON reserialization equality

다음 test를 생성하라.

```text
tests/test_phase0_1_evidence_index.py
```

test는 validator를 호출하고 representative lookup 20개를 확인한다.

외부 package를 추가하지 마라.

---

# 10. Exact allowlist

생성/수정 가능:

```text
docs/research/index/phase0_1/**
tools/research/validate_phase0_1_evidence_index.py
tests/test_phase0_1_evidence_index.py
experiments/research_index/reports/PHASE0_1_REPOSITORY_INDEX_VALIDATION_REPORT.md
experiments/research_index/agent_logs/**
experiments/research_index/preflight_snapshots/**
```

기존 Phase 0–1 source, tests, configs, reports, results, manifests는 모두 read-only다.

allowlist 밖 변경이 필요하면:

```text
BLOCKED_REPOSITORY_INDEX_SCOPE_EXTENSION_REQUIRED
```

로 중단하라.

---

# 11. Git/dirty-tree 정책

현재 working tree 전체를 승인 baseline으로 사용하라.

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

branch, HEAD, history, past commit delta를 승인 조건으로 검토하지 마라.

실행 전:

```text
status
tracked/staged patch
untracked path list
frozen file hash
allowlist existence
```

를 기록하라.

실행 후:

```text
기존 Git-visible path 제거 0
allowlist 밖 신규 path 0
기존 frozen path hash mismatch 0
staged patch unchanged
tracked pre-existing patch unchanged
```

을 검증하라.

---

# 12. 필수 검증 질문

최종 index만 사용해 다음 질문에 답할 수 있는지 self-test하라.

1. 현재 P1 Exit는 무엇이며 어느 파일이 canonical인가?
2. initial P1 Exit와 updated P1 Exit의 관계는?
3. F-BASE/F-TUNED/F-CALIBRATED의 역할과 값은?
4. MAIN-FUSION settled NIS/NEES의 source는?
5. C4 full-oracle 개선량의 source는?
6. C2 oracle Q의 제한된 결과는?
7. C5 RMS matching 값과 claim limit은?
8. STRESS-MAG weak direction 수치는?
9. Basilisk frame 변환 proof는 어디인가?
10. MEKF math source/test는?
11. event schema와 ordering source는?
12. canonical metric implementation은?
13. runner/cache integration은?
14. exact fresh/cache smoke command는?
15. Phase 1 test counts는?
16. dirty-tree integrity evidence는?
17. F-CALIBRATED가 primary가 아닌 이유는?
18. Phase 2 진입 조건은?
19. learned model로 아직 입증되지 않은 주장은?
20. 특정 numeric fact의 result JSON/manifest key는?

하나라도 index에서 추적할 수 없으면 validation PASS를 주지 마라.

---

# 13. 완료 판정

정상 완료 형식:

```text
Status: PASS_PHASE0_1_REPOSITORY_EVIDENCE_INDEX

Human-readable index: PASS
Machine-readable JSON: PASS
Question-to-file routing: PASS
Numeric evidence catalog: PASS
Decision/supersession ledger: PASS
Phase 2 lookup handoff: PASS
Path/locator validation: PASS
Representative 20 lookups: PASS
No frozen-file modification: PASS
Dirty-tree integrity: PASS

Indexed topics: <count>
Indexed questions: <count>
Indexed numeric facts: <count>
Verified file paths: <count>
Unresolved ambiguities: <count>
```

일부 source가 없거나 locator를 검증할 수 없으면:

```text
Status: PARTIAL_PHASE0_1_REPOSITORY_EVIDENCE_INDEX
```

로 보고하고 누락 항목을 정확히 기록하라.

Phase 2 설계나 구현으로 자동 진행하지 마라.
