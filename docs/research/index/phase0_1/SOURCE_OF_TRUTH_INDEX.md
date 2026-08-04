# Source of Truth Index

## 현재 최상위 상태

| 주장 | canonical source | exact locator | supporting evidence | 상태 |
|---|---|---|---|---|
| Phase 0–1 전체 navigation/handoff | `docs/research/phase1b/AI_ADCS_PHASE0_1_MASTER_SUMMARY_AND_PHASE2_HANDOFF.md` | authority note와 §1–9 | master audit report; 각 claim은 아래 canonical source까지 추적 | current navigation; exact-number authority 아님 |
| 현재 P1 Exit는 `CONDITIONAL_GO` | `experiments/phase1b/reports/P1_EXIT_REVIEW_UPDATED.md` | `Decision`, `## 8. Updated P1 Exit decision` | `.../updated_exit_review.json:/decision` | current |
| stationary covariance closure는 통과 | 같은 updated review | stationary acceptance table | JSON `/acceptance/stationary_passed` | current |
| C4 covariance closure는 실패 | 같은 updated review | C4 acceptance와 remaining limitation | JSON `/acceptance/c4_passed`, `/remaining_classical_limitation` | current |
| Phase 2 implementation은 미승인·미구현 | 같은 updated review | header/final decision | JSON `/phase2_implemented=false` | current |
| Phase 2 Design Review는 explicit request로만 시작 | master summary §9 | Design-Review and Implementation Boundary | updated review의 separate-approval 문구 | current navigation boundary; implementation 승인 아님 |
| 최초 P1 Exit도 `CONDITIONAL_GO`였음 | `experiments/phase1b/reports/P1_EXIT_REVIEW.md` | `## 7. P1 Exit decision` | original `exit_review.json` | historical; updated review가 대체 |

## Topic별 canonical chain

| ID | 질문/주장 | 먼저 열 source와 locator | supporting evidence | implementation / test / config | 상태 |
|---|---|---|---|---|---|
| NAV-MASTER-SUMMARY | Phase 0–1 전체 요약과 authority boundary | master summary authority note, §1–9 | master audit report; topic별 canonical chain | navigation only | current audited |
| P0-OBJECTIVE | estimator-first 연구 순서와 scope | `P0_01_DECISION_LEDGER.md::D01-D03`; `P0A_PHASE_0A_SYNTHESIS.md::§1, §10` | decision table | 문서 결정 | current |
| P0-STATE-CONVENTION | `[q_NB,b_g]`, 6D right error, scalar-first active B→N | `P0_01_DECISION_LEDGER.md::D04-D08` | `P0_05_MEKF_MATH_CONTRACT.md::§2-3` | `bench/estimators/mekf.py`; core/convention tests | current |
| P0-TRUTH-BOUNDARY | truth/sensor/estimator/oracle 분리 | `P0_02_TRUTH_SENSOR_ESTIMATOR_BOUNDARY.md::§1, §4-5` | leakage prohibitions | event and sidecar sources/tests | current |
| P0-SENSOR-ROLES | gyro/mag/sun/ST 역할과 비행제품 비주장 | `P0_04_SENSOR_ROLE_AND_MODEL_SPEC.md` sensor sections | Phase 1 generators | Basilisk/fusion generator tests | current |
| P0-CONTEXT-CONTRACT | physical/oracle/estimated/latent context | `P0_07_CONTEXT_CONTRACT.md::§1-2, §8-9` | identifiability limits | Step 1/2 experiments and tests | current |
| P1A-GATE-A-CORE | locked MEKF math/core | `P1A_GATE_A_FINAL_APPROVAL.md::§1, §4` | math validation report | `bench/estimators/mekf.py`; `test_mekf_core.py` | current GO |
| P1A-EXACT-PI-IMMUTABILITY | q/-q exact-pi invariance와 array immutability | Gate A approval `§2-3` | 1000 ordinary, 256 near-pi sweep | `align_quaternion`, `quat_log`, `MEKFState`; core tests | current |
| P1A-TYPED-EVENTS | versioned gyro/ST schema, hash, split, replay | `P1A_EVENT_SCHEMA_CONTRACT.md` schema/order/erratum | B1 final + A1 report | `mekf_events.py`, synthetic generator; three tests | current |
| P1A-B1-GENERATOR-IDENTITY | schema와 generator identity 분리 | `P1A_GATE_B1_AMENDMENT_A1_CONTRACT.md` identity invariants | A1 validation report | `mekf_events.py`; identity tests | current |
| P1A-BASILISK-FRAME | sigma_BN→active q_NB와 omega proof | `P1A_GATE_B2_FINAL_APPROVAL.md::§2-4` | `P1A_BASILISK_FRAME_CONVENTION_PROOF.md` | `basilisk_unit_st.py`; generator tests | current GO |
| P1A-CANONICAL-METRICS | geodesic/bias/NIS/NEES/SPD | `P1A_GATE_C_FINAL_APPROVAL.md::§2-3` | Gate C report | `bench/metrics/mekf.py`; metric tests | current GO |
| P1A-ADAPTER-RUNNER | direct=D1=runner artifact와 dataset identity | `P1A_CP4_VALIDATION_REPORT.md::§5` | fresh/cache hashes | model bridge, task prep, registry, runner; adapter/runner tests | current GO |
| P1B-STEP1-FROZEN-BASELINES | F-BASE primary, F-TUNED sensitivity-only | `P1B_STEP1_FINAL_APPROVAL_AND_STEP2_HANDOFF.md::§2` | `tuning.json:/fixed_tuning` | Step 1 experiment/config/tests | current |
| P1B-C1-STATIONARY | stationary matched baseline | `P1B_UNIT_ST_BASELINE_REPORT.md` C1 table | `pilot_summary.json` C1 group | Step 1 experiment | current |
| P1B-C2-PROCESS | gyro process mismatch 존재 | `P1B_PROBLEM_EXISTENCE_REPORT.md` C2 | pilot summary | regimes experiment/test | current preliminary support |
| P1B-C3-MEASUREMENT | ST inlier covariance mismatch 존재 | 같은 report C3 | pilot summary | regimes experiment/test | current strong classical evidence |
| P1B-C5-IDENTIFIABILITY | matched innovation RMS A/B | `P1B_IDENTIFIABILITY_PILOT_REPORT.md` matching/limits | `tuning.json`, `pilot_summary.json` | Step 1 experiment/test | current, pair-specific only |
| P1B-LONG-HORIZON | F-BASE 안정, F-TUNED penalty | `P1B_UNIT_ST_BASELINE_REPORT.md` long-horizon | `long_horizon.json` | Step 1 experiment/config | current |
| P1B-STEP2-FUSION-SCHEMA | 별도 typed schema와 gyro→mag→sun→ST | Step 1→2 handoff `§4`; Step 2 validation | fusion pilot | `mekf_fusion_events.py`; event tests | current |
| P1B-FUSION-SENSORS | deterministic benchmark mag/sun layer | Step 1→2 handoff `§4` | sensor fusion validation | generator, fusion metrics, estimator tangent functions | current; flight environment claim 아님 |
| P1B-MAIN-FUSION | stationary four-sensor baseline | `P1B_SENSOR_FUSION_BASELINE_REPORT.md` | `settled_consistency.json` | Step 2 experiment/config/test | current original Step 2 evidence |
| P1B-STRESS-MAG | single-vector observability stress | `P1B_STRESS_MAG_REPORT.md` decomposition | fusion pilot summary | Step 2 experiment | current limitation evidence |
| P1B-C4-COMBINED | slow Qb + fast Rmag event와 oracle/wrong-side | `P1B_C4_COMBINED_EVENT_REPORT.md` | fusion pilot summary | Step 2 experiment/config/tests | current original C4 study |
| P1B-INITIAL-EXIT | NEES 1.873 condition | `P1_EXIT_REVIEW.md::§7` | original exit JSON | Step 2 result | historical; updated review가 대체 |
| P1B-CLOSURE-DIAGNOSTICS | transient/marginal/whitened/cross 분해 | `P1_EXIT_TRANSIENT_DIAGNOSTIC_REPORT.md` | `diagnosis.json` | closure experiment/test/config | current |
| P1B-F-CALIBRATED | P0/Q only frozen calibration | `P1_EXIT_COVARIANCE_CALIBRATION_REPORT.md` frozen candidate | updated JSON `/F_CALIBRATED_status` | closure source/test/config | current frozen comparator |
| P1B-CLOSURE-CONFIRMATION | independent stationary+C4 N=50 | `P1_EXIT_CLOSURE_VALIDATION_REPORT.md` | confirmation and updated JSON | closure source/test/config | current |
| P1B-CURRENT-EXIT | updated conditional decision | `P1_EXIT_REVIEW_UPDATED.md::§8` | updated JSON | closure test/config | current canonical |
| PROVENANCE-TESTS | 전 regression suite | `regression_evidence.json` | closure validation report | named test files | current |
| PROVENANCE-COMMANDS | 재현 명령 | CP4 validation `§5`; Step 1/2/closure command logs | reports/results | four suite YAMLs | current |
| PROVENANCE-DIRTY-TREE | 기존 dirty 변경 보존 | closure `FINAL_INTEGRITY.md`; indexing preflight README | status/patch/hash snapshots | provenance only | current |

## 충돌처럼 보이는 수치의 판독 규칙

`1.873`, `1.906245`, `1.418`, `1.021`은 같은 sample set을 네 번 측정한 값이
아니다. 각각 original Step 2 MAIN-FUSION settled F-BASE, closure validation
settled F-BASE, independent closure confirmation settled F-BASE, 같은 confirmation의
F-CALIBRATED-v1이다. policy·dataset·split·window가 다르므로 scope를 생략한
“최종 NEES” 표현은 금지한다.

original C4 full-oracle 개선 28.56%/32.57%와 closure independent C4 confirmation
32.09%/41.32%도 서로 다른 dataset과 실행 단계의 값이다. 전자는 Phase 1B
Step 2 pilot, 후자는 closure confirmation이다.

## Master amendment status

mandatory master summary는 expected input SHA와 일치하는 artifact에서 exact
path로 복원되었고, 모든 required claim family가 audit되었다. title-lock 표현,
numeric scope/precision, fixed-calibration 일반화, Phase 2 boundary를
documentation-only로 교정했다. master의 navigation copy가 topic별 canonical
source나 numeric catalog를 대체하지 않는다.
