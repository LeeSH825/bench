# Decision and Status Ledger

## Status vocabulary

- `current`: 현재도 claim/contract/decision의 source of truth다.
- `historical`: 당시 유효했던 결정이나 handoff이며 이력 확인에 사용한다.
- `superseded`: 같은 domain의 더 최신 결정/정정이 있어 현재 판정에 사용하지 않는다.

구현 단계가 끝났다는 사실과 Phase 2 진입 승인은 다른 결정이다. 현재 Phase 1
Exit가 `CONDITIONAL_GO`이므로 Phase 2는 승인되지 않았다.

Master summary는 current navigation/handoff지만 새로운 research decision이
아니다. Phase 2 Design Review는 별도의 명시적 사용자 요청으로 시작할 수 있고,
Phase 2 implementation은 그와 별개의 미착수·미승인 상태다.

## Decision ledger

| ID | domain | 결정 | canonical source / locator | status | replacement 관계 |
|---|---|---|---|---|---|
| D-P0-ESTIMATOR-FIRST | P0 | estimator quality primary; closed-loop final validation | `P0_01_DECISION_LEDGER.md::D01` | current | none |
| D-P0-MEKF-CONVENTION | P0 | kinematic right-error MEKF, active scalar-first q_NB | 같은 ledger D03-D08 | current | none |
| D-P0-BOUNDARY | P0 | deployable estimator에서 truth/oracle 금지 | `P0_02_TRUTH_SENSOR_ESTIMATOR_BOUNDARY.md::§4-5` | current | none |
| D-P1A-GATE-A | Gate A | GO | `P1A_GATE_A_FINAL_APPROVAL.md::§1` | current | none |
| D-P1A-GATE-B1 | Gate B1 | GO after A1 | `P1A_GATE_B1_AMENDMENT_A1_REPORT.md::Final decision` | current | original B1 approval을 보완, schema 유지 |
| D-P1A-QNB-ERRATUM | Gate B1/B2 | q_NB의 실제 의미는 active B→N | `P1A_EVENT_SCHEMA_CONTRACT.md::Gate B2 convention erratum` | current | old passive prose superseded; code/schema/hash unchanged |
| D-P1A-GATE-B2 | Gate B2 | GO | `P1A_GATE_B2_FINAL_APPROVAL.md::Final decision` | current | failed/retry attempt가 아닌 final proof 우선 |
| D-P1A-GATE-C | Gate C | GO | `P1A_GATE_C_FINAL_APPROVAL.md::Final decision` | current | none |
| D-P1A-CP4 | CP4 integration | GO | `P1A_CP4_VALIDATION_REPORT.md::Final decision` | current | none |
| D-P1B-STEP1 | Phase 1B Step 1 | PASS; F-BASE primary, F-TUNED frozen sensitivity comparator | `P1B_STEP1_FINAL_APPROVAL_AND_STEP2_HANDOFF.md::§1-2` | current | none |
| D-P1B-STEP2 | Phase 1B Step 2 | PASS sensor fusion/C4 implementation and paired studies | `P1_EXIT_CONDITIONAL_GO_AND_CLOSURE_HANDOFF.md` header/§1 | current | implementation completion, Phase 2 approval 아님 |
| D-P1-EXIT-ORIGINAL | P1 Exit | CONDITIONAL_GO pending covariance closure | `P1_EXIT_REVIEW.md::§7` | historical | D-P1-EXIT-UPDATED가 supersede |
| D-P1-FCAL-FREEZE | P1 closure | F-CALIBRATED-v1 = P0_att 2, P0_bias 4, Qg 2, Qb 8; R fixed | `P1_EXIT_COVARIANCE_CALIBRATION_REPORT.md::Frozen candidate` | current | new comparator; F-BASE/F-TUNED overwrite 아님 |
| D-P1-EXIT-UPDATED | P1 Exit | CONDITIONAL_GO; stationary passed, C4 failed | `P1_EXIT_REVIEW_UPDATED.md::Decision/§8` | **current canonical** | supersedes D-P1-EXIT-ORIGINAL |
| D-PHASE2-NOT-AUTHORIZED | Phase 2 entry | Phase 2 implementation not authorized | current updated review final decision | **current** | explicit separate future approval required; master는 변경하지 않음 |

## Current policy matrix

| policy | locked parameters | 역할 | current status |
|---|---|---|---|
| F-BASE | original baseline P0/Q/R | primary classical baseline/reference | frozen current |
| F-TUNED | `s_Qg=.125`, `s_Qb=.125`, `s_R_ST=8` | sensitivity comparator only | frozen current; primary 승격 금지 |
| F-CALIBRATED-v1 | `s_P0_att=2`, `s_P0_bias=4`, `s_Qg=2`, `s_Qb=8`, all sensor R=1 | independently selected covariance comparator | frozen current; stationary pass/C4 fail |
| ORACLE-PROCESS | current-event process sidecar | simulation-only upper-bound/causal ablation | frozen evidence, deployable 아님 |
| ORACLE-MEASUREMENT | current-event measurement sidecar | simulation-only upper-bound/causal ablation | frozen evidence, deployable 아님 |
| ORACLE-FULL | current-event process+measurement sidecar | simulation-only upper bound | frozen evidence, deployable 아님 |
| WRONG-PROCESS / WRONG-MEASUREMENT | deliberate wrong-side action | causal comparison | frozen evidence |

## Historical-to-current exit timeline

```text
Step 1 PASS
  → Step 2 PASS
  → original P1 Exit CONDITIONAL_GO
       condition: original MAIN-FUSION settled NEES/DOF = 1.8730178719854724
  → independent covariance closure
       stationary: PASS, F-CALIBRATED-v1 NEES/DOF = 1.0206761630935368
       C4: FAIL, calibrated bias degradation = 0.5807472544511563
  → updated P1 Exit CONDITIONAL_GO (current)
  → Phase 2: not authorized
```

## Superseded wording and values

1. `q_NB`를 passive라고 부른 B1 prose는 superseded다. executable proof가 잠근
   current 의미는 scalar-first Hamilton active body-to-navigation이다. 코드,
   NPZ schema, hash domain은 변경되지 않았다.
2. `P1_EXIT_REVIEW.md`는 historical decision evidence다. 현재 판정에는
   `P1_EXIT_REVIEW_UPDATED.md`를 먼저 사용한다.
3. original 1.873을 closure 후 stationary 값 1.021로 “정정”했다고 말하면 안
   된다. dataset/split/policy가 다른 별도 결과다.
4. original C4 full-oracle 28.56%/32.57%와 independent closure C4
   32.09%/41.32%는 별도 experiment scope다.

## Unresolved decisions/ambiguities

| ID | 내용 | 상태 | 영향 |
|---|---|---|---|
| A-PHASE2-ENTRY-CONDITION | updated review는 C4 limitation을 남기며 Phase 2 implementation을 승인하지 않음 | unresolved | Design Review는 explicit user request로만 시작; implementation은 별도 승인 필요 |

## Resolved ambiguity history

| ID | resolution | 근거 |
|---|---|---|
| A-MISSING-MASTER-SUMMARY | resolved 2026-08-02 | expected digest와 일치하는 artifact를 exact path로 복원하고 full claim audit·index·validator를 완료 |
