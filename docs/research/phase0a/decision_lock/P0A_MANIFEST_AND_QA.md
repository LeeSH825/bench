# Phase 0A 파일 Manifest 및 QA Report

> 작성일: 2026-07-30  
> 목적: 요구 산출물 존재, 공통 header, Decision status, Markdown fence, 핵심 수학 reference test와 파일 hash를 확인한다.

## QA summary

| 검사 | 결과 |
|---|---|
| 요구된 12개 독립 산출물 | PASS |
| 모든 산출물 상단 목적/입력/상태/TBD/Gate | PASS |
| Evidence/Deprecated의 확인·미확인 header | PASS |
| Decision Ledger allowed status / rows | PASS / 26 |
| Markdown code fence balance | PASS |
| Quaternion product/propagation vectors | PASS |
| Reset Jacobian finite difference | PASS (`1.854e-10`) |
| Mag Jacobian random finite difference | PASS (`2.812e-09`) |
| Van Loan `Q_d` SPD sanity | PASS (`lambda_min=1.000e-10`) |
| 전체 QA | **PASS** |

수학 QA는 문서 수식의 내부 일관성을 확인하기 위한 독립 reference calculation이다. 실제 Phase 1 코드의 unit test를 대체하지 않는다.

## File manifest

`P0A_MANIFEST_AND_QA.md`와 machine-readable QA 파일은 자기 참조 hash 문제를 피하기 위해 아래 hash 표에서 제외했다. ZIP에는 둘 다 포함된다.

| 파일 | lines | bytes | SHA-256 prefix |
|---|---:|---:|---|
| `P0A_IMMEDIATE_TEST_SPEC.md` | 511 | 26550 | `6d7890415c12c8af…` |
| `P0A_PHASE_0A_SYNTHESIS.md` | 209 | 12755 | `8501e3a3d5792242…` |
| `P0A_REFERENCE_REGISTER.md` | 61 | 4969 | `12fa7065c2a1e377…` |
| `P0_00_DEPRECATED_ASSUMPTIONS.md` | 56 | 5546 | `674c85f973a4eab4…` |
| `P0_00_EVIDENCE_REGISTER.md` | 86 | 7399 | `8ab17f7c585ddc65…` |
| `P0_01_DECISION_LEDGER.md` | 60 | 7500 | `a08b564f83a7e19e…` |
| `P0_02_TRUTH_SENSOR_ESTIMATOR_BOUNDARY.md` | 139 | 10097 | `793bd100a519b7f6…` |
| `P0_03_TRUTH_MODEL_SPEC.md` | 271 | 13215 | `366e08be41040106…` |
| `P0_04_SENSOR_ERROR_CATALOG.md` | 150 | 9709 | `e93b1855c1019e52…` |
| `P0_04_SENSOR_ROLE_AND_MODEL_SPEC.md` | 357 | 16664 | `d863a01ec61c20e0…` |
| `P0_05_MEKF_CONVENTION_TEST_VECTORS.md` | 400 | 7644 | `55b29674eb8b4979…` |
| `P0_05_MEKF_MATH_CONTRACT.md` | 612 | 16034 | `4c20cd530d1aa277…` |
| `P0_06_NEURAL_INSERTION_OPTIONS.md` | 324 | 12284 | `0153090af8483e92…` |
| `P0_07_CONTEXT_CONTRACT.md` | 298 | 12665 | `498e89d9517e52ae…` |
| `README.md` | 42 | 1464 | `2f89c54f5c624d5e…` |

전체 machine-readable 결과는 `P0A_QA_RESULTS.json`에 있다.
