# Question-to-File Map

각 행의 `first open`을 먼저 읽고, 수치는 locator의 experiment scope까지 함께
인용한다. `next evidence / implementation`은 주장을 source 또는 test로 확인할
때만 연다. 아래 103개 질문은 machine index의 150개 question pattern 중 자주
필요한 subset이다.

## 1–10: Phase 0 decisions and boundaries

| # | 질문 | first open | exact locator | next evidence / implementation | status |
|---:|---|---|---|---|---|
| 1 | Phase 0의 최우선 연구 목적은? | `docs/research/phase0a/decision_lock/P0_01_DECISION_LEDGER.md` | D01 | `P0A_PHASE_0A_SYNTHESIS.md::§1` | current |
| 2 | closed-loop 검증은 언제 하는가? | 같은 ledger | D01 | synthesis exit criteria | current |
| 3 | 어떤 estimator class를 잠갔는가? | 같은 ledger | D03 | `P0_05_MEKF_MATH_CONTRACT.md` | current |
| 4 | nominal state는 무엇인가? | 같은 ledger | D04 | `bench/estimators/mekf.py::MEKFState` | current |
| 5 | error state 차원과 구성은? | 같은 ledger | D04 | math contract §3; core tests | current |
| 6 | quaternion 순서와 방향은? | 같은 ledger | D05-D07 | convention tests | current |
| 7 | truth/sensor/estimator boundary는? | `P0_02_TRUTH_SENSOR_ESTIMATOR_BOUNDARY.md` | §1-2 | event generator source | current |
| 8 | deployable estimator 금지 입력은? | 같은 boundary | §4-5 | Step 1/2 leakage tests | current |
| 9 | oracle의 허용 역할은? | 같은 boundary | oracle section | `P0_07_CONTEXT_CONTRACT.md` | current |
| 10 | Phase 0 metric decision은? | `P0_01_DECISION_LEDGER.md` | D21 | Gate C approval/source | current |

## 11–20: Sensor and context contracts

| # | 질문 | first open | exact locator | next evidence / implementation | status |
|---:|---|---|---|---|---|
| 11 | gyro의 estimator 역할은? | `P0_04_SENSOR_ROLE_AND_MODEL_SPEC.md` | Gyroscope | `mekf.py::propagate_state` | current |
| 12 | magnetometer model은? | 같은 sensor spec | Magnetometer | fusion generator/vector update | current |
| 13 | sun sensor invalid 처리법은? | 같은 sensor spec | Sun sensor validity | generator and fusion tests | current |
| 14 | star tracker가 제공하는 자세량은? | 같은 sensor spec | Star tracker | event contract/ST update | current |
| 15 | sensor model이 flight product claim인가? | 같은 sensor spec | scope limitations | Step 2 handoff truth section | current: no |
| 16 | physical context 정의는? | `P0_07_CONTEXT_CONTRACT.md` | §1 | Step 1/2 configs | current |
| 17 | oracle context 정의는? | 같은 context contract | §1-2 | oracle sidecar source/tests | current |
| 18 | estimated context 허용 입력은? | 같은 context contract | estimated context | estimator replay source | current |
| 19 | latent context는 true label인가? | 같은 context contract | latent context | §8 forbidden inputs | current: no |
| 20 | identifiability claim 한계는? | 같은 context contract | §9 | C5 report limitations | current |

## 21–30: Gate A and B1

| # | 질문 | first open | exact locator | next evidence / implementation | status |
|---:|---|---|---|---|---|
| 21 | Gate A 최종 판정은? | `docs/research/phase1a/P1A_GATE_A_FINAL_APPROVAL.md` | §1 | math validation report | current GO |
| 22 | MEKF propagation source는? | 같은 approval | implementation list | `bench/estimators/mekf.py::propagate_state` | current |
| 23 | ST residual/update source는? | 같은 approval | exact-pi section | `star_tracker_residual`, `star_tracker_update` | current |
| 24 | exact 180° q/-q 처리법은? | 같은 approval | §2 | `align_quaternion`, `quat_log`; core test | current |
| 25 | MEKFState가 실제 immutable인가? | 같은 approval | §3 | `MEKFState`; mutation tests | current |
| 26 | Gate A test counts는? | 같은 approval | §4 | regression JSON | current, 55 + legacy 18/5 |
| 27 | B1 schema version은? | `docs/research/phase1a/P1A_EVENT_SCHEMA_CONTRACT.md` | versioned schema | `mekf_events.py::SCHEMA_VERSION` | current |
| 28 | Gate B1 latency는? | 같은 event contract | zero-latency rule | event tests | current, zero |
| 29 | 같은 timestamp의 gyro/ST 순서는? | 같은 event contract | deterministic ordering | replay tests | current |
| 30 | whole-trajectory split은 어디에? | 같은 event contract | split section | `mekf_events.py::split_trajectory_ids` | current |

## 31–40: Gate B1 identity, B2, and C

| # | 질문 | first open | exact locator | next evidence / implementation | status |
|---:|---|---|---|---|---|
| 31 | schema ID와 generator ID 차이는? | `P1A_GATE_B1_AMENDMENT_A1_CONTRACT.md` | identity separation | A1 report/source tests | current |
| 32 | generator ID가 바꾸는 hash는? | 같은 A1 contract | hash invariants | manifest identity tests | current: manifest only |
| 33 | q_NB passive 문구는 유효한가? | `P1A_EVENT_SCHEMA_CONTRACT.md` | Gate B2 convention erratum | frame proof | superseded wording; active meaning current |
| 34 | sigma_BN→q_NB 식은? | `P1A_GATE_B2_FINAL_APPROVAL.md` | §2 locked relation | `basilisk_sigma_BN_to_q_NB` | current |
| 35 | R_NB와 MRP2C 관계는? | 같은 B2 approval | §2 | executable frame proof | current |
| 36 | omega_BN_B frame/sign/unit은? | 같은 B2 approval | §3 dynamic proof | generator source/test | current |
| 37 | MRP shadow set이 검증됐는가? | 같은 B2 approval | §3 | `run_static_frame_proof` | current |
| 38 | built-in Basilisk ST를 썼는가? | 같은 B2 approval | §4 sensor layer | generator source | current: no |
| 39 | canonical NIS/NEES source는? | `P1A_GATE_C_FINAL_APPROVAL.md` | §2 | `bench/metrics/mekf.py` | current |
| 40 | SPD consistency가 repair를 쓰는가? | 같은 Gate C approval | safety contract | metric source/tests | current: strict Cholesky, no repair |

## 41–50: Adapter, runner, and Step 1 freeze

| # | 질문 | first open | exact locator | next evidence / implementation | status |
|---:|---|---|---|---|---|
| 41 | D1 bridge artifact는 무엇인가? | `P1A_CP4_STEP1_FINAL_APPROVAL.md` | D1 approval | `bench/models/mekf.py::MEKFReplayArtifact` | current |
| 42 | direct replay와 bridge가 같은가? | 같은 approval | equality evidence | adapter tests | current, array-equal |
| 43 | task/model 고정 ID는? | `P1A_CP4_VALIDATION_REPORT.md` | task/model routing | runner constants | current |
| 44 | dense float32 path 이전 분기는? | 같은 report | runner routing | `run_suite.py::_is_p1a_mekf_event_replay_pair` | current |
| 45 | fresh/cache 검증 명령은? | 같은 report | §5 | CLI logs and smoke YAML | current |
| 46 | runner truth join key는? | 같은 report | metric integration | `_p1a_exact_truth_join` | current: trajectory_id/time |
| 47 | primary classical baseline은? | `P1B_STEP1_FINAL_APPROVAL_AND_STEP2_HANDOFF.md` | §2 Primary | Step 1 baseline report | current: F-BASE |
| 48 | F-TUNED scales는? | `tuning.json` | `/fixed_tuning/selected_policy` | handoff §2 | current frozen 0.125/0.125/8 |
| 49 | F-TUNED이 primary인가? | Step 1 final approval | §2 F-TUNED status | long-horizon report | current: sensitivity-only |
| 50 | Step 1 tuning이 test를 보았는가? | `tuning.json` | `/test_split_accessed` | Step 1 validation | current: false |

## 51–60: C1, C2, and C3

| # | 질문 | first open | exact locator | next evidence / implementation | status |
|---:|---|---|---|---|---|
| 51 | C1 F-BASE attitude RMSE는? | `P1B_UNIT_ST_BASELINE_REPORT.md` | stationary row F-BASE | pilot JSON C1 group | current, 1.6615e-3 rad |
| 52 | C1 F-BASE NIS/NEES는? | 같은 baseline report | stationary row F-BASE | pilot JSON | current, 0.924/0.967 |
| 53 | C1 paired N은? | `pilot_summary.json` | `/completed_paired_N_per_condition` | Step 1 validation | current, 50 |
| 54 | fixed Q/R mismatch 결과는? | baseline report | mismatch table | pilot JSON | current sensitivity evidence |
| 55 | C2 event가 바꾸는 것은? | `P1B_PROBLEM_EXISTENCE_REPORT.md` | C2 definition | Step 1 regimes source | current: gyro process uncertainty |
| 56 | C2 alpha 8 결과는? | 같은 report | C2 alpha 8 row | pilot JSON | current, RMSE 1.7520e-3; NEES 1.187 |
| 57 | C2 oracle이 attitude를 개선했나? | 같은 report | C2 oracle interpretation | pilot JSON | current: resolved benefit 없음 |
| 58 | C3 event가 outlier인가? | 같은 report | C3 definition | regimes source/config | current: inlier covariance, not outlier |
| 59 | C3 alpha 8 result는? | 같은 report | C3 alpha 8 row | pilot JSON | current, RMSE 3.6859e-3; NIS 2.245 |
| 60 | C3 correct-side oracle benefit은? | 같은 report | C3 oracle comparison | pilot JSON | current, strong in scoped case |

## 61–70: C5, long horizon, fusion schema

| # | 질문 | first open | exact locator | next evidence / implementation | status |
|---:|---|---|---|---|---|
| 61 | C5 alpha_R_ST는? | `tuning.json` | `/frozen_c5_B_alpha_R` | C5 report | current, 1.08 |
| 62 | C5 validation RMS match는? | `tuning.json` | `/c5_matching/selected/...` | C5 report | current, 0.3962% |
| 63 | C5 independent test match는? | `pilot_summary.json` | `/c5_AB_independent_test/F-BASE/...` | C5 report | current, 1.4851% |
| 64 | C5가 일반 불식별성을 증명했나? | `P1B_IDENTIFIABILITY_PILOT_REPORT.md` | limitations | context contract | current: no, pair-specific only |
| 65 | long-horizon duration/N은? | `long_horizon.json` | `/duration_s`, `/num_trajectories` | baseline report | current, 600 s/N=10 |
| 66 | long-horizon F-BASE consistency는? | baseline report | long-horizon F-BASE row | long JSON records | current, NIS .988/NEES .854 |
| 67 | long-horizon F-TUNED penalty는? | baseline report | long-horizon F-TUNED row | long JSON records | current, NEES 2.252 and ~85% attitude penalty |
| 68 | fusion schema는 B1 schema를 수정했나? | Step 1→2 handoff | separate-schema decision | `mekf_fusion_events.py` | current: separate v1 family |
| 69 | four-sensor same-time order는? | 같은 handoff | Same-time order | fusion event source/tests | current: gyro→mag→sun→ST |
| 70 | invalid sun을 zero로 넣나? | 같은 handoff | Sun sensor | generator/replay tests | current: update skip |

## 71–80: Fusion sensors, MAIN-FUSION, STRESS-MAG

| # | 질문 | first open | exact locator | next evidence / implementation | status |
|---:|---|---|---|---|---|
| 71 | magnetic/sun reference가 flight environment인가? | Step 1→2 handoff | Truth section | generator source | current: deterministic benchmark only |
| 72 | mag/sun NIS source는? | `P1B_STEP2_VALIDATION_REPORT.md` | metric evidence | `bench/metrics/mekf_fusion.py` | current |
| 73 | MAIN-FUSION N은? | `P1B_SENSOR_FUSION_BASELINE_REPORT.md` | scenario scope | pilot JSON | current, 50 |
| 74 | original settled mag NIS는? | `settled_consistency.json` | `/.../mag_nis/normalized_mean` | baseline report | current, 1.02289827 |
| 75 | original settled sun/ST NIS는? | 같은 JSON | sun/star_tracker pointers | baseline report | current, 1.00048662/1.09188891 |
| 76 | original settled posterior NEES는? | 같은 JSON | `/.../nees/normalized_mean` | original exit review | current original study, 1.87301787 |
| 77 | F-TUNED이 MAIN에서 개선했나? | baseline report | F-TUNED comparison | pilot JSON | current: 여러 지표 악화 |
| 78 | STRESS-MAG attitude RMSE는? | `P1B_STRESS_MAG_REPORT.md` | F-BASE aggregate | pilot JSON | current, .197681 rad |
| 79 | STRESS-MAG weak/plane 분해는? | 같은 stress report | observability decomposition | Step 2 source | current, .195676/.00133072 rad |
| 80 | 한 magnetic vector로 full attitude 관측 가능한가? | 같은 report | limitation | unsupported claims JSON | current: no |

## 81–90: C4 and original exit

| # | 질문 | first open | exact locator | next evidence / implementation | status |
|---:|---|---|---|---|---|
| 81 | C4 slow event는? | `P1B_C4_COMBINED_EVENT_REPORT.md` | primary event | Step 2 source/config | current: alpha_b(t) |
| 82 | C4 fast event는? | 같은 report | primary event | Step 2 source/config | current: inlier alpha_R_mag(t) |
| 83 | C4 event scales/windows는? | 같은 report | event definition | suite YAML | current, 100000/16 with documented windows |
| 84 | original C4 process-only effect는? | 같은 report | oracle comparison | pilot JSON paired differences | current original study |
| 85 | original C4 measurement-only effect는? | 같은 report | oracle comparison | pilot JSON | current original study |
| 86 | original C4 full-oracle slow/fast improvement는? | 같은 report | full-oracle row | pilot JSON | current original study, 28.56%/32.57% |
| 87 | wrong measurement는 무엇을 악화했나? | 같은 report | wrong-side comparison | pilot JSON | current original study |
| 88 | oracle이 future event window를 보는가? | 같은 report | information boundary | sidecar/replay test | current: current-event forward-only |
| 89 | original P1 Exit decision은? | `P1_EXIT_REVIEW.md` | §7 | original exit JSON | historical CONDITIONAL_GO |
| 90 | original named condition은? | `P1_EXIT_CONDITIONAL_GO_AND_CLOSURE_HANDOFF.md` | §2 | settled JSON | historical trigger: NEES 1.873 |

## 91–100: Closure and current exit

| # | 질문 | first open | exact locator | next evidence / implementation | status |
|---:|---|---|---|---|---|
| 91 | closure split이 기존 test와 독립인가? | `P1_EXIT_TRANSIENT_DIAGNOSTIC_REPORT.md` | split integrity | `diagnosis.json:/split` | current: yes, train30/val20 |
| 92 | closure가 무엇을 원인으로 진단했나? | 같은 diagnostic report | marginal ranking | updated JSON `/diagnosed_cause` | current: bias then attitude; transient large |
| 93 | validation initial/settled 수치는? | updated JSON | `/diagnosed_cause` | calibration report | current validation scope: 15.558/1.435/2.745 |
| 94 | F-CALIBRATED-v1 scales는? | updated JSON | `/F_CALIBRATED_status/scales` | calibration report/freeze manifest | current frozen: 2/4/2/8 |
| 95 | sensor R를 바꿨나? | updated JSON | `/F_CALIBRATED_status/sensor_R_scales` | config/test | current: all 1.0 |
| 96 | stationary confirmation F-BASE/F-CAL NEES는? | updated JSON | two confirmation full-NEES pointers | closure report | current independent N=50: 1.418/1.021 |
| 97 | F-CAL stationary sensor NIS는? | updated JSON | calibrated confirmation NIS fields | closure report | current: mag .9853/sun .9704/ST .9899 |
| 98 | closure C4는 왜 실패했나? | updated JSON | `/remaining_classical_limitation` | C4 confirmation report | current: bias +58.07%, settled NIS outside bounds |
| 99 | 현재 P1 Exit decision은? | `P1_EXIT_REVIEW_UPDATED.md` | §8 | updated JSON `/decision` | current CONDITIONAL_GO |
| 100 | Phase 2가 승인되었나? | 같은 updated review | final decision | updated JSON `/phase2_implemented` | current: no |

## 101–103: Master navigation and Phase 2 boundary

| # | 질문 | first open | exact locator | next evidence / implementation | status |
|---:|---|---|---|---|---|
| 101 | Phase 0–1 전체 요약을 먼저 어디서 읽는가? | `docs/research/phase1b/AI_ADCS_PHASE0_1_MASTER_SUMMARY_AND_PHASE2_HANDOFF.md` | authority note, §1–9 | topic별 source-of-truth index | current audited navigation |
| 102 | master summary와 exact numeric source가 다르면 무엇이 우선하는가? | 같은 master | authority note | machine result JSON → specialized report → numeric catalog | canonical result 우선 |
| 103 | Phase 2 Design Review와 implementation authorization은 어떻게 다른가? | 같은 master | §9 Design-Review and Implementation Boundary | updated P1 Exit review final scope | Design Review는 explicit request 필요; implementation은 별도 미승인 |

Master summary는 expected digest와 일치하는 제공 artifact에서 exact path로
복원된 뒤 audit되었다. 현재 P1 Exit는 계속 `CONDITIONAL_GO`이며 master 자체는
새 research decision이 아니다.
