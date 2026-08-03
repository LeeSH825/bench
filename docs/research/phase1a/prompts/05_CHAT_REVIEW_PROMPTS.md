# Chat Review Prompts

## A. Read-only audit 결과를 가져온 뒤

다음 내용을 현재 AI-ADCS Phase 0A 문서와 대조하여 검토해줘.

첨부/붙여넣기:
- `P1A_REPOSITORY_AUDIT.md`
- `P1A_IMPLEMENTATION_MAP.md`
- `P1A_RISK_REGISTER.md`
- Code CLI agent의 command/test summary

검토 항목:
1. 기존 benchmark 재사용 범위가 적절한지
2. legacy MRP+angular-rate 경로와 새 `[q,b_g]` MEKF가 분리되는지
3. 새 파일 위치가 장기 Phase 2–7 구조에도 적합한지
4. truth/sensor/estimator/oracle 정보 경계가 강제되는지
5. Phase 0A convention과 충돌하는 항목이 있는지
6. 구현 전 수정해야 할 계획 항목
7. 승인 가능한 exact file map

최종 출력은 `승인`, `조건부 승인`, `재감사 필요` 중 하나와 Code CLI Prompt 2 수정본으로 작성해줘.

## B. Math/Core 결과를 가져온 뒤

다음 Phase 1A math/core 결과를 Phase 0A 수학 계약과 test vector에 대조해 검토해줘.

첨부/붙여넣기:
- `P1A_IMPLEMENTATION_CONTRACT.md`
- `P1A_TEST_MATRIX.md`
- `P1A_MATH_VALIDATION_REPORT.md`
- git diff 또는 changed-file summary
- pytest 전체 출력

B1/B3/B4/B5/B6를 각각 PASS/FAIL로 판단하고, 부호·frame·right-error·reset·q/-q·SPD 관점에서 implementation error 가능성을 점검해줘. 통과하지 못한 항목이 있으면 UNIT-ST/Basilisk로 넘어가지 않도록 해줘.

## C. UNIT-ST 결과를 가져온 뒤

다음 UNIT-ST/Basilisk integration 결과를 검토해줘.

첨부/붙여넣기:
- `P1A_UNIT_ST_REPORT.md`
- `P1A_C1_BASELINE_REPORT.md`
- `P1A_GATE_REPORT.md`
- sensor packet schema/manifest example
- test output와 주요 plot/table

B2/B7/B8/C1과 data leakage, deterministic replay, all-one oracle equivalence를 판단하고 Package C 진입 여부를 결정해줘.

## D. Package C 결과를 가져온 뒤

다음 Package C 결과를 검토해줘.

첨부/붙여넣기:
- `P1A_PROBLEM_EXISTENCE_REPORT.md`
- `P1A_IDENTIFIABILITY_REPORT.md`
- `P1A_ORACLE_USEFULNESS_REPORT.md`
- updated `P1A_GATE_REPORT.md`
- paired Monte Carlo summary

C2/C3/C4/C5를 기반으로 adaptation 필요성, oracle usefulness, process/measurement 분리 필요성, innovation-only 한계, robust gate 필요성을 판단해줘. 결과에 따라 다음 단계를 `고전 적응 baseline`, `ANN context`, `MEKF-Split baseline`, `연구 범위 축소` 중 하나로 결정해줘.
