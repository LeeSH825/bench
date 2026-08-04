# Phase 0–1 Master Summary Audit Report

- 실행일: 2026-08-02
- disposition: `PASS_MASTER_SUMMARY_PRESENT_AND_AUDITED`
- master corrections: 5 logical documentation-only correction groups
- current P1 Exit: `CONDITIONAL_GO` (unchanged)
- Phase 2 implementation: `not_started_not_authorized`
- Phase 2 Design Review: separate explicit user request required

## 1. Provenance

Contract path:

```text
docs/research/phase1b/AI_ADCS_PHASE0_1_MASTER_SUMMARY_AND_PHASE2_HANDOFF.md
```

The supplied artifact was initially visible with a filename suffix:

```text
docs/research/phase1b/AI_ADCS_PHASE0_1_MASTER_SUMMARY_AND_PHASE2_HANDOFF (1).md
```

Its SHA-256 exactly matched the amendment contract's expected source digest. The
artifact was copied byte-for-byte to the exact contract path using the repository
patch mechanism; the suffixed user artifact was left untouched.

| item | SHA-256 |
|---|---|
| expected/input artifact | `657b956362457472c25ba03177e521114d1d92082cc30e10cf5f4170f52b96a2` |
| exact-path byte copy before correction | `657b956362457472c25ba03177e521114d1d92082cc30e10cf5f4170f52b96a2` |
| final evidence-corrected master | `8827f4a3996b0e1f6b736de13a33a738bab06bee30e789bc1e85876e8dd40526` |

This is an explained `same content with intentional evidence-based correction`,
not an unexplained provenance change.

## 2. Canonical source families checked

| family | canonical evidence checked |
|---|---|
| Phase 0 objective/state/frame | `P0_01_DECISION_LEDGER.md` D01–D08; `P0A_PHASE_0A_SYNTHESIS.md` §1/§3 |
| truth/context boundary | `P0_02_TRUTH_SENSOR_ESTIMATOR_BOUNDARY.md`; `P0_07_CONTEXT_CONTRACT.md` |
| Gate A | `P1A_GATE_A_FINAL_APPROVAL.md`; `P1A_MATH_VALIDATION_REPORT.md`; MEKF source/tests |
| Gate B1/A1 | event schema contract, B1/A1 approvals/reports, event source/tests |
| Gate B2 | B2 final approval, frame-convention proof, Basilisk source/test |
| Gate C | Gate C final approval/report, metric source/test |
| D1/CP4 | D1/CP4 reports, bridge/runner source, adapter/runner tests |
| Step 1 | final handoff, tuning/pilot/long-horizon JSON, three specialized reports, source/config/tests |
| Step 2 | Step 2 handoff/validation, settled/pilot JSON, MAIN/STRESS/C4 reports, source/config/tests |
| Closure | transient/calibration/validation reports, diagnosis/search/confirmation/updated-review JSON, source/config/test |
| Current decision | `P1_EXIT_REVIEW_UPDATED.md` and `updated_exit_review.json` |

Machine-frozen JSON and updated review were used ahead of the master. No master
number was promoted to canonical numeric authority.

## 3. Required claim audit matrix

| claim family | master claim | canonical locator/result | audit |
|---|---|---|---|
| research objective | time-varying gyro/sensor reliability under a structured adaptive estimator study | Phase 0 synthesis §1 and decision ledger D01-D03 | PASS |
| research title | Korean/English title described as fixed | exact title appears only in the supplied master; no separate Phase 0–1 title-lock decision was found | CORRECTED to handoff-recorded title |
| MEKF state | `[q_NB,b_g]`, right-local 6D error | ledger D04-D08; math contract; `MEKFState` | PASS |
| quaternion/frame | scalar-first Hamilton active B→N; `C_BN=R_NB.T` | ledger D05-D07; convention tests | PASS |
| truth boundary | truth→sensor output→estimator; listed truth/oracle inputs excluded | truth boundary §1/§4-5; event-sidecar tests | PASS |
| context/oracle | simulation-only oracle sidecar | context contract; Step 1/2 source/tests | PASS |
| Gate A count | 55 passed | Gate A final approval §4 | PASS |
| Gate B1 count | 55 passed after A1 | B1 A1 report regression | PASS |
| Gate B2 count | 67 passed | B2 final approval | PASS |
| Gate C count | 43 passed | Gate C final approval §3 | PASS |
| Basilisk frame | `q_NB=normalize(MRP2EP(sigma_BN))`, `R_NB=MRP2C(sigma_BN).T` | B2 final approval §2-3; executable proof | PASS |
| project-owned ST | built-in Basilisk ST not used | B2 final approval §4; generator source | PASS |
| adapter/runner IDs | `mekf_unit_st_v1`, `mekf_event_replay_v1` | CP4 validation; runner constants | PASS |
| replay equality | direct=bridge=runner q/b/P/r/S; fresh=cache | D1/CP4 reports and tests | PASS |
| F-BASE role | primary classical baseline | Step 1 final approval §2 | PASS |
| F-TUNED | scales `.125/.125/8`, sensitivity-only | `tuning.json:/fixed_tuning`; final handoff | PASS |
| C1/C2/C3 | matched baseline, modest process effect, strong ST covariance effect | UNIT-ST and problem-existence reports; pilot JSON | PASS |
| C5 | scalar innovation RMS insufficient only for constructed pair; raw gyro useful | C5 report and pilot/tuning JSON | PASS |
| long horizon | T=600 s, N=10; F-TUNED penalty | `long_horizon.json`; baseline report | PASS |
| fusion order | gyro→mag→sun→ST | Step 2 handoff §4; fusion schema/tests | PASS |
| MAIN settled | mag/sun/ST NIS 1.023/1.000/1.092, full NEES 1.873 (rounded) | `settled_consistency.json:/main_fusion_stationary_F_BASE` | PASS as navigation copies |
| STRESS-MAG | weak direction 0.195676 rad; plane 0.001331 rad | STRESS report decomposition | PASS |
| original C4 oracle | slow/fast/mag-NIS/NEES improvements 28.56/32.57/47.20/96.32% | C4 report and Step 2 pilot JSON | PASS, original Step 2 scope |
| closure split | train 30, validation 20, stationary 50, C4 50 | diagnosis and confirmation JSON | PASS |
| diagnosed cause | initial transient; settled bias marginal ranks ahead of attitude; not sensor R | transient report; updated review | PASS |
| validation diagnostics | initial 15.558045; settled full 1.906245; attitude 1.434813; bias 2.744853; cross norm 0.559550 | `diagnosis.json:/groups/validation/aggregate/partitions` | CORRECTED scope/precision |
| F-CALIBRATED-v1 | P0 attitude/bias 2/4, Qg/Qb 2/8, R all one | updated JSON `/F_CALIBRATED_status` | PASS |
| stationary confirmation | F-BASE 1.418027; calibrated 1.020676; marginals/NIS near declared values | updated JSON confirmation fields | PASS |
| C4 confirmation | bias degradation 58.075%; NIS 1.733/1.921/4.006 | updated JSON remaining limitation | PASS |
| fixed calibration conclusion | text generalized from one frozen search to all global fixed calibration | closure acceptance only establishes failure of declared F-CALIBRATED-v1 | CORRECTED unsupported generalization |
| updated P1 Exit | `CONDITIONAL_GO` | updated review decision and JSON `/decision` | PASS, unchanged |
| Phase 2 boundary | separate stage, implementation not begun | updated review future scope | CORRECTED to explicit Design Review/request vs implementation approval |

## 4. Corrections made

1. Clarified the header as completed classical-evidence execution while retaining
   current `CONDITIONAL_GO`; added the navigation-only authority note.
2. Changed “fixed title” to a handoff-recorded title because no separate title-lock
   decision exists in the Phase 0–1 canonical ledger.
3. Scoped closure diagnosis to independent validation F-BASE and copied the
   machine values at unambiguous precision, including cross-relative norm.
4. Replaced the universal fixed-calibration impossibility wording with the supported
   statement about the frozen F-CALIBRATED-v1 declared search.
5. Split Phase 2 Design Review from implementation: Design Review requires an
   explicit user request; implementation remains not started/not authorized and
   requires separate approval.

No frozen Phase 0–1 source, test, config, report, result, or manifest was changed.
Only the contract-allowlisted index documents, index validator/test/report artifacts,
and the exact-path master documentation were amended.

## 5. Line-level source-to-final diff

```diff
@@ header and authority
-- Phase 0–1 상태: **완료**
+- Phase 0–1 classical evidence 실행 상태: **완료**
+> Authority note: 이 문서는 navigation/handoff summary다. 정확한 수치에는
+> machine-frozen result JSON과 specialized final report가, 현재 판정에는
+> `P1_EXIT_REVIEW_UPDATED.md`가 항상 우선한다.
-확정 제목:
+Handoff에 기록된 연구 제목(Phase 0–1 decision ledger의 별도 title-lock 항목은 아님):

@@ independent validation diagnosis
-Diagnosis:
-initial full NEES/DOF=15.558
-settled full=1.906
-settled attitude marginal=1.435
-settled bias marginal=2.745
-attitude-bias P cross norm=0.560
+Diagnosis on the independent validation split, F-BASE:
+initial full NEES/DOF=15.558045
+settled full NEES/DOF=1.906245
+settled attitude marginal NEES/DOF=1.434813
+settled bias marginal NEES/DOF=2.744853
+settled attitude-bias P cross relative norm=0.559550

@@ scientific conclusion
-7. A global fixed covariance calibration cannot satisfy stationary and C4 simultaneously.
+7. The frozen F-CALIBRATED-v1 selected under the declared P0/Q search does not
+   satisfy stationary and C4 acceptance simultaneously; this is not a universal
+   impossibility claim for every fixed calibration.

@@ Phase 2 boundary
-## 9. Phase 2 Entry Conditions
-Phase 2 is allowed only as a separately approved design stage.
+## 9. Phase 2 Design-Review and Implementation Boundary
+이 master summary는 Phase 2 Design Review나 implementation을 자동 승인하지 않는다.
+Phase 2 Design Review는 별도의 명시적 사용자 요청으로만 시작할 수 있다.
+Phase 2 implementation은 미착수 상태이며 Design Review와 별개의 명시적 승인이 필요하다.
```

## 6. Unsupported claims and numeric scope

Before correction, two unsupported/overbroad formulations existed: a separately
locked project title and a universal statement about every global fixed covariance
calibration. Both are resolved in the final master. No unsupported claim remains
active.

The master retains rounded navigation values for original MAIN-FUSION, STRESS-MAG,
original C4, and confirmation results. Their canonical exact sources remain the
machine JSON/specialized reports. The index's earlier closure-validation value
`1.905894` was independently found to be erroneous and was corrected to canonical
`1.9062451467732702` from `diagnosis.json`; this is an index correction, not a master
number promoted to authority.

## 7. Ambiguities and boundary

- `A-MISSING-MASTER-SUMMARY`: **resolved** after exact-path restoration, audit,
  correction, index update, and validation.
- `A-PHASE2-ENTRY-CONDITION`: remains active and refined. The current P1 Exit is
  `CONDITIONAL_GO`; Phase 2 implementation is not authorized. A Phase 2 Design
  Review may start only under a separate explicit user request.

No Phase 2 design or implementation was performed.
