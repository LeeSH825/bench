# Phase 0–1 Repository Evidence Index

이 디렉터리는 Phase 0–1의 질문·주장과 저장소 증거를 연결한다. 단순 파일
목록이 아니라 다음 탐색 순서를 고정한다.

```text
질문/주장 → canonical source → exact locator → supporting evidence
          → implementation/test/config/result → current/historical/superseded
```

현재 저장소 기준 P1 Exit 판정은 **CONDITIONAL_GO**다. 독립 stationary
covariance closure는 통과했지만 독립 C4 confirmation은 사전 선언된 accuracy와
sensor-NIS 기준을 통과하지 못했다. Phase 2는 승인되거나 구현되지 않았다.

## 문서 사용 순서

1. 전체 navigation은
   `docs/research/phase1b/AI_ADCS_PHASE0_1_MASTER_SUMMARY_AND_PHASE2_HANDOFF.md`에서
   시작한다.
2. 새 대화의 repository lookup 순서는 `PHASE2_REPOSITORY_LOOKUP_HANDOFF.md`를 따른다.
3. 질문별 첫 canonical 파일은 `QUESTION_TO_FILE_MAP.md`에서 찾는다.
4. 수치는 반드시 scope와 함께 `NUMERIC_EVIDENCE_CATALOG.md`에서 확인한다.
5. 구현·test·config·result 경로는 `FILE_PATH_CATALOG.md`에서 연결한다.
6. 과거와 현재 판정은 `DECISION_AND_STATUS_LEDGER.md`에서 구분한다.
7. 명령형 조회 절차는 `AGENT_LOOKUP_RECIPES.md`를 사용한다.
8. 자동화는 `phase0_1_evidence_index.json`과 repository validator를 사용한다.

## 산출물

- `SOURCE_OF_TRUTH_INDEX.md`: topic별 canonical source와 status.
- `QUESTION_TO_FILE_MAP.md`: 103개 질문의 first-open 파일과 locator.
- `NUMERIC_EVIDENCE_CATALOG.md`: 값·단위·scope·source를 결합한 수치 catalog.
- `FILE_PATH_CATALOG.md`: source/test/config/report/result/provenance 경로.
- `DECISION_AND_STATUS_LEDGER.md`: current/historical/superseded 결정 이력.
- `PHASE2_REPOSITORY_LOOKUP_HANDOFF.md`: 다음 대화에서 열 파일의 최소 순서.
- `AGENT_LOOKUP_RECIPES.md`: 재현 가능한 read-only lookup 명령.
- `phase0_1_evidence_index.json`: canonical sorted-key machine index.

Validator:

```bash
python3 tools/research/validate_phase0_1_evidence_index.py \
  --repo-root /home/dss-pc-05/bench
```

Tests:

```bash
python3 -m pytest -q tests/test_phase0_1_evidence_index.py
```

## Master restoration과 판정

mandatory master summary는 exact contract path에 복원되어 전 claim family를
canonical evidence와 대조했다.

```text
docs/research/phase1b/AI_ADCS_PHASE0_1_MASTER_SUMMARY_AND_PHASE2_HANDOFF.md
```

제공 artifact SHA-256은 계약의 expected digest
`657b956362457472c25ba03177e521114d1d92082cc30e10cf5f4170f52b96a2`와
일치했다. evidence-based documentation correction 후 final SHA는
`8827f4a3996b0e1f6b736de13a33a738bab06bee30e789bc1e85876e8dd40526`다.
`A-MISSING-MASTER-SUMMARY`는 resolved ledger로 이동했고 전체 판정은
**PASS_PHASE0_1_REPOSITORY_EVIDENCE_INDEX**다.

Master summary는 navigation/handoff 문서이며 exact-number authority나 새로운
research decision이 아니다. 정확한 수치와 현재 결정은 각각 machine-frozen
result와 updated final review까지 추적해야 한다.

## 증거 우선순위

결정은 updated final review가 이전 review보다 우선한다. 수치는 machine-frozen
result가 specialized report보다 우선하며 반드시 experiment/split/policy/window를
같이 읽는다. 구현 사실은 actual source가 contract·test·report보다 우선한다.
같은 이름의 지표라도 scope가 다르면 서로 덮어쓰지 않는다.

Phase 2 implementation은 미착수·미승인이다. Phase 2 Design Review는 별도의
명시적 사용자 요청이 있을 때만 시작할 수 있으며 implementation 승인과 다르다.
