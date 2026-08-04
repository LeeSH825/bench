# Phase 0–1 Repository Index Validation Report

- 실행일: 2026-08-02
- repository root: `/home/dss-pc-05/bench`
- 작업 성격: read-only evidence indexing; Phase 2 설계/구현 아님
- 구조/연결 validator: **PASS**
- index tests: **PASS — 5 passed**
- 초기 계약 판정: **PARTIAL_MISSING_MANDATORY_MASTER_SUMMARY** (historical)
- amendment 후 최종 판정: **PASS_PHASE0_1_REPOSITORY_EVIDENCE_INDEX**
- Phase 2 진행: **미승인, 미수행**

## Amendment timeline

```text
initial index:
  PARTIAL_MISSING_MANDATORY_MASTER_SUMMARY

amendment:
  expected-digest master artifact restored at exact path
  all mandatory claim families audited
  five documentation-only correction groups applied
  A-MISSING-MASTER-SUMMARY resolved

final:
  PASS_PHASE0_1_REPOSITORY_EVIDENCE_INDEX
```

Master summary is a current navigation/handoff document, not exact-number or
decision authority. Current P1 Exit remains `CONDITIONAL_GO`. Phase 2
implementation remains not started/not authorized; Phase 2 Design Review may
start only under a separate explicit user request.

## 1. Contract and preflight

최우선 계약
`docs/research/index/prompts/PHASE0_1_REPOSITORY_EVIDENCE_INDEXING_AGENT_PROMPT.md`
를 처음부터 끝까지 읽었다. branch, HEAD, commit history, 과거 commit delta는
검토하지 않았다. 시작 시점의 current working tree 전체를 기준선으로 사용했고,
기존 dirty 변경을 reset/restore/clean/stash/stage/commit/push하지 않았다.

entry snapshot:

```text
experiments/research_index/preflight_snapshots/01_20260802T_currentZ/
```

주요 entry facts:

- porcelain status records: 1,082
- frozen Phase 0–1 files: 4,064
- tracked unstaged patch SHA-256:
  `95ecb66407b46ad8509783da355d8d030f1aa64a1759d5ef586838ace6a5b1c6`
- staged patch: empty, SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- contract allowlist output targets: 모두 entry 시점에 absent

## 2. Input coverage

다음 evidence families를 recursive inventory와 targeted full read로 검토했다.

- Phase 0 decision lock: objective, truth boundary, truth model, sensor roles,
  MEKF math, neural options, context contract, synthesis.
- Phase 1A: Gate A, B1/A1, B2 frame proof, Gate C, D1, CP4 decisions,
  implementations, tests, configs, reports, command/provenance files.
- Phase 1B Step 1: frozen tuning, C1/C2/C3/C5, paired N=50, 600 s long
  horizon, reports/results/manifests.
- Phase 1B Step 2: fusion schema/order, benchmark mag/sun sensors,
  MAIN-FUSION, STRESS-MAG, C4, original exit review.
- P1 Exit closure: independent split, transient/marginal/whitened/cross
  diagnostics, search/freeze, stationary and C4 confirmation, updated review.

Initial-index mandatory master source search (historical):

```text
find . -type f -name AI_ADCS_PHASE0_1_MASTER_SUMMARY_AND_PHASE2_HANDOFF.md
rg --files | rg AI_ADCS_PHASE0_1_MASTER_SUMMARY_AND_PHASE2_HANDOFF.md
```

초기 indexing 당시 두 검색 모두 결과가 없었다. Amendment에서는 expected SHA와
정확히 일치한 suffixed artifact를 exact path로 복원하고 audit했다. 예상 경로
`docs/research/phase1b/AI_ADCS_PHASE0_1_MASTER_SUMMARY_AND_PHASE2_HANDOFF.md`는
현재 readable하다. 초기 `A-MISSING-MASTER-SUMMARY`는 resolved ledger로 이동했다.

## 3. Generated index coverage

Validator-reported counts:

| 항목 | count |
|---|---:|
| topics | 32 |
| question patterns | 150 |
| machine numeric facts | 73 |
| indexed files | 50 |
| decisions | 15 |
| representative lookups | 21 |
| unresolved ambiguities | 1 |
| resolved ambiguities | 1 |

Human question map에는 numbered question 103개가 있다. 각 question은 first-open
source, exact locator, next supporting evidence/implementation, status를 가진다.

## 4. Validator checks

실행 명령:

```bash
python3 tools/research/validate_phase0_1_evidence_index.py \
  --repo-root /home/dss-pc-05/bench
```

결과:

```json
{"ambiguities": 1, "decisions": 15, "files": 50, "numeric_facts": 73, "questions": 150, "representative_lookups": 21, "resolved_ambiguities": 1, "status": "PASS", "topics": 32}
```

검사 범위:

- canonical sorted-key JSON exact bytes
- schema version과 repository-root metadata
- topic/file/decision/ambiguity/lookup ID uniqueness
- mandatory topic presence
- every canonical/numeric/lookup source path existence와 non-empty locator
- implementation/test/config/result path existence
- numeric value/unit/scope/source linkage
- question count ≥80, numeric fact count ≥30
- at least 21 representative lookups, preserving L01–L20 and adding L21
- exactly one current P1 Exit decision plus historical predecessor/supersedes link
- current `CONDITIONAL_GO`, `phase2_authorized=false`
- non-root portable path에 absolute path 없음
- mandatory master path, indexed file entry, authority metadata, expected/final SHA
- master not used as a canonical numeric source
- missing-master ambiguity absent from active list and present in resolved ledger
- master-first handoff order and explicit Design Review/implementation boundary

## 5. Index tests

시스템 `/usr/bin/python3`에는 pytest가 없어 첫 test invocation은
`No module named pytest`로 종료됐다. repository의 기존 Phase 1 validation
runtime인 Python 3.10.13로 동일 tests를 실행했다.

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
  -m pytest -q tests/test_phase0_1_evidence_index.py
```

최종 결과:

```text
.....                                                                    [100%]
5 passed in 0.03s
```

초기 test 작성본에서는 current decision field가 rationale를 함께 포함했고,
test의 rounded expected NEES가 exact stored value와 맞지 않아 2개가 실패했다.
index decision은 exact enum `CONDITIONAL_GO`와 별도 rationale로 정규화했고,
NEES checks는 tolerance 확장이 아니라 machine-stored exact values에 대한 exact
equality로 수정했다. 최종에는 skip/xfail/jitter/tolerance 완화가 없다.

## 6. Twenty-one representative lookup self-tests

| ID | 질문 domain | first source | 결과 |
|---|---|---|---|
| L01 | q_NB convention | Phase 0 decision ledger D04-D08 | PASS |
| L02 | estimator forbidden inputs | truth boundary §4-5 | PASS |
| L03 | exact-pi q/-q | Gate A approval §2 | PASS |
| L04 | B1 ordering | event schema contract | PASS |
| L05 | sigma_BN→q_NB | Gate B2 approval §2 | PASS |
| L06 | canonical NIS/NEES | Gate C approval §2 | PASS |
| L07 | runner task/model pair | CP4 validation | PASS |
| L08 | F-TUNED scales | `tuning.json:/fixed_tuning/selected_policy` | PASS |
| L09 | C1 F-BASE | UNIT-ST baseline report | PASS |
| L10 | C2 process event | problem-existence report C2 | PASS |
| L11 | C3 measurement event | problem-existence report C3 | PASS |
| L12 | C5 interpretation | identifiability report | PASS |
| L13 | 600 s long horizon | UNIT-ST baseline report | PASS |
| L14 | four-sensor order | Step 1→2 handoff | PASS |
| L15 | original MAIN NEES | settled-consistency JSON pointer | PASS |
| L16 | STRESS-MAG observability | stress report decomposition | PASS |
| L17 | original C4 effects | C4 report | PASS |
| L18 | closure cause | transient diagnostic report | PASS |
| L19 | F-CALIBRATED-v1 | updated-review JSON | PASS |
| L20 | current exit/Phase 2 status | updated review | PASS |
| L21 | audited master navigation/authority | master authority note and §9 | PASS |

Test가 21개 lookup source의 실제 file 존재, locator non-empty, topic linkage를
반복 검사한다.

## 7. Numeric conflict resolution

동일해 보이는 posterior NEES를 scope로 분리했다.

| 값 | scope |
|---:|---|
| 1.8730178719854724 | original Step 2 MAIN-FUSION test N=50, F-BASE, settled |
| 1.9062451467732702 | closure independent validation N=20, F-BASE, settled |
| 1.4180268635870965 | closure independent stationary confirmation N=50, F-BASE, settled |
| 1.0206761630935368 | 같은 confirmation, F-CALIBRATED-v1, settled |

original C4 full-oracle 28.56%/32.57%와 closure confirmation
32.0873%/41.3209%도 별도 dataset/step로 cataloged했다. 어느 값도 다른 값을
수정하거나 supersede하지 않는다.

## 8. Current/historical status validation

- current canonical review:
  `experiments/phase1b/reports/P1_EXIT_REVIEW_UPDATED.md`
- historical predecessor:
  `experiments/phase1b/reports/P1_EXIT_REVIEW.md`
- machine current decision: `CONDITIONAL_GO`
- stationary closure: passed
- C4 closure: failed
- Phase 2 implementation: not started/not authorized
- Phase 2 Design Review: separate explicit user request required
- old passive q_NB prose: superseded by active B→N erratum/proof, without
  source-code/NPZ/hash-domain migration

## 9. Output scope

생성/수정은 contract exact allowlist에만 한정했다.

```text
docs/research/index/phase0_1/*.md
docs/research/index/phase0_1/phase0_1_evidence_index.json
docs/research/phase1b/AI_ADCS_PHASE0_1_MASTER_SUMMARY_AND_PHASE2_HANDOFF.md
tools/research/validate_phase0_1_evidence_index.py
tests/test_phase0_1_evidence_index.py
experiments/research_index/reports/PHASE0_1_REPOSITORY_INDEX_VALIDATION_REPORT.md
experiments/research_index/reports/PHASE0_1_MASTER_SUMMARY_AUDIT_REPORT.md
experiments/research_index/agent_logs/02_*
experiments/research_index/preflight_snapshots/02_*
```

기존 Phase 0–1 source, tests, configs, reports, results, manifests를 수정하지
않았다. Phase 2 source/design/config/result는 생성하지 않았다.

## Amendment validation and integrity

```text
validator: PASS
tests: 5 passed in 0.03s
master final SHA: 8827f4a3996b0e1f6b736de13a33a738bab06bee30e789bc1e85876e8dd40526
frozen files checked: 4,649; all equal
tracked patch SHA entry/exit: 95ecb66407b46ad8509783da355d8d030f1aa64a1759d5ef586838ace6a5b1c6
staged patch SHA entry/exit: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
non-allowlist untracked delta: 0
numbered human lookup questions: 103
```

## 10. Final disposition

Initial historical disposition was
`PARTIAL_MISSING_MANDATORY_MASTER_SUMMARY`. The mandatory file is now present at
the exact path, its expected input provenance is verified, its full claim audit
and documented corrections are complete, and all index/validator/test/integrity
requirements pass.

**PASS_PHASE0_1_REPOSITORY_EVIDENCE_INDEX**

```text
Status: PASS_PHASE0_1_REPOSITORY_EVIDENCE_INDEX

Mandatory master summary: PRESENT_AND_AUDITED
Human-readable index: PASS
Machine-readable JSON: PASS
Question-to-file routing: PASS
Numeric evidence catalog: PASS
Decision/supersession ledger: PASS
Phase 2 lookup handoff: PASS
Path/locator validation: PASS
Representative lookups: PASS
No frozen-file modification: PASS
Dirty-tree integrity: PASS

Master corrections: 5
Indexed topics: 32
Indexed questions: 150
Indexed numeric facts: 73
Verified file paths: 50
Unresolved ambiguities: 1
```

Phase 2로 진행하지 않고 종료한다.
