# Phase 0A 즉시 실행 최소 검증 패키지

> 작성일: 2026-07-30
> 표지 규칙: **[확인]** 실험·실측, **[문헌]** 논문·공식 자료, **[분석]** 수식·구조 해석, **[가설]** 검증 대상, **[결정]** 설계 선택, **[보류]** 근거 부족·후속 범위

| 항목 | 내용 |
|---|---|
| 목적 | 코드를 크게 확장하거나 context estimator를 학습하기 전에 MEKF 수학, UNIT-ST, time-varying uncertainty 문제와 oracle context의 필요성을 검증한다. |
| 입력 근거 | S0–S4 및 P0_02–P0_07 계약 문서 |
| 결정 상태 | LOCK — 실험 순서·비교군·로그·해석; PROVISIONAL — 일부 tolerance와 실용 효과량 threshold |
| 남은 TBD | 실측 기반 base noise, final Monte Carlo sample size/power, practical-improvement threshold |
| 다음 Gate | Package B 전부 통과 + Package C에서 문제 존재성과 oracle usefulness가 확인되어야 Phase 1 이후 ANN/SNN 단계로 진행 |


## 0. 실행 원칙

1. **AI/context estimator를 학습하지 않는다.** Oracle context는 simulator가 지정한 scale/gate를 직접 사용한다.
2. 모든 estimator는 동일 truth trajectory, 동일 pre-generated sensor packet, 동일 initial state를 사용한다.
3. randomization은 trajectory-level이고 train/validation/test 개념을 지금부터 유지한다.
4. 각 실험은 `no-noise analytic`, `single-seed debug`, `Monte Carlo` 순서로 확장한다.
5. Package B 실패 시 Package C 결과를 해석하지 않는다.
6. sensor numeric base profile이 없으면 dimensionless normalized profile로 시험하고, 결과를 실제 제품 성능으로 주장하지 않는다.

## 1. 공통 실험 계약

### 1.1 Common baseline

```yaml
state: [q_NB, b_g]
error_state: [delta_theta, delta_b_g]
filter: 6D Kinematic MEKF
quaternion: scalar-first Hamilton, B_to_N active, right error
scenario_unit: UNIT-ST
scenario_main: MAIN-FUSION
rates_representative:
  gyro: 100 Hz
  magnetometer: 10 Hz
  sun_vector: 5 Hz
  star_tracker: 1 Hz
  temperature: 1 Hz
numeric_reference: float64
```

### 1.2 Estimator set

| ID | estimator | knowledge |
|---|---|---|
| `F-MIS` | fixed mismatched MEKF | baseline `Q0/R0`, time variation unknown |
| `F-TUNED` | tuned MEKF | scenario-wide best fixed values or matched stationary values; tuning budget disclosed |
| `ORACLE-QR` | oracle Q/R-scaled MEKF | true assigned `α_g,α_b,α_R,j`, no future measurement realization |
| `ROBUST` | robust/gated MEKF | packet validity and declared residual gate; optional where outlier exists |
| `WRONG-SIDE` | diagnostic only | process event treated as R change or measurement event treated as Q change |

`F-TUNED` is not allowed to use event time at inference. `ORACLE-QR` may use current true context but is labeled non-deployable.

### 1.3 Common logs

- truth: `q_NB`, `ω`, `b_g`, orbit/environment, hidden event state
- sensor: measurement/arrival time, raw/calibrated values, validity/quality
- estimator: prior/posterior `q,b,P`, `Φ,Q_d,H,R,S,K`, innovation, NIS, gate, reset Jacobian
- oracle: true scale/gate labels
- numerical: quaternion norm, symmetry error, minimum eigenvalues, jitter, exceptions
- experiment: config hash, code commit, seeds, runtime, platform

### 1.4 Common metrics

| metric | definition/use |
|---|---|
| attitude geodesic RMSE/P95/peak | sign-invariant SO(3) angle |
| bias RMSE/P95 | `b_true-b_hat`, rad/s |
| convergence time | first time rolling error enters and remains in configured band |
| recovery time | after event end, rolling error returns to `≤max(1.2×pre-event baseline, absolute floor)` and stays for window `W` |
| divergence rate | nonfinite, SPD failure, or configured sustained large attitude error; threshold versioned |
| NIS/NEES | only explicit covariance models and valid inlier assumptions |
| innovation whiteness | autocorrelation/Ljung–Box with caveats for gating/multirate |
| stationary penalty | adaptive model degradation in matched stationary condition |

### 1.5 Statistical policy

- pilot: **[결정: provisional]** at least 50 independent trajectories per condition;
- final `N`: selected from paired effect/power analysis after pilot;
- comparisons are paired by truth/sensor seed;
- report median, mean, P95, confidence interval and divergence count;
- primary improvement requires a paired 95% bootstrap confidence interval excluding zero **and** a practical threshold set after the pilot;
- multiple hypotheses use a declared correction or hierarchical primary/secondary metric order.

## 2. Package B — 최소 수학 검증

---

## B1. `[q,b_g]` Kinematic MEKF 수식·discretization 검증

| 항목 | specification |
|---|---|
| 목적 | quaternion propagation, error `F/G`, `Φ/Q_d`, state/covariance units와 analytic cases 검증 |
| truth condition | zero motion, constant body rate, constant bias; no environment/sensor noise in first pass |
| sensor configuration | gyro only for propagation; exact synthetic samples |
| estimator configuration | common 6D MEKF predict only |
| 입력 파라미터 | convention test vectors; multiple `dt`, `ω`, bias values; SPD `Q_c` cases |
| 변경 변수 | `dt`, rate axis/magnitude, bias, random-walk PSD |
| 고정 변수 | quaternion convention, right error, float64 |
| 기록 로그 | `q`, norm, `F,G,Φ,Q_d,P`, analytic/reference state |
| 평가 지표 | absolute/relative error, symmetry, eigenvalues |
| expected behavior | zero motion unchanged; constant rate equals quaternion exponential; `Q_d→0` as `dt→0`; covariance grows monotonically for positive process noise |
| 합격 기준 | `P0_05_MEKF_CONVENTION_TEST_VECTORS.md`; `Φ/Q_d` agrees with high-accuracy numerical integration within configured tolerance |
| 실패 시 해석 | multiplication order/frame sign, gyro-bias sign, Van Loan block extraction or unit error |

Additional checks:

- finite-difference linearization of one-step error map vs `Φ`;
- `Q_d` convergence under substepping;
- optional GM bias profile only after RW baseline passes.

---

## B2. `UNIT-ST` gyro + star-tracker unit scenario

| 항목 | specification |
|---|---|
| 목적 | attitude+bias observability, ST log residual, multirate/asynchronous update와 covariance consistency 검증 |
| truth condition | stationary and constant-rate cases; constant gyro bias, then bias RW; no other sensors |
| sensor configuration | gyro 100 Hz, ST 1 Hz; first zero latency, then representative delayed packet |
| estimator configuration | `F-TUNED` only initially; no neural/context |
| 입력 파라미터 | base gyro/ST noise symbolic profile; initial attitude error strata; constant bias vectors |
| 변경 변수 | initial attitude error, bias magnitude/direction, ST rate/latency, noise seed |
| 고정 변수 | same truth/sensor stream, exact mounting, no outlier |
| 기록 로그 | full common logs + ST relative quaternion/residual and buffer/repropagation trace |
| 평가 지표 | geodesic/bias error, convergence, NIS/NEES, norm/SPD, delay handling error |
| expected behavior | no-noise exact case converges to numerical tolerance; noisy matched case remains bounded and statistically consistent; delay-aware output matches replay reference |
| 합격 기준 | no divergence; bias estimate enters posterior uncertainty band; ensemble NIS/NEES within computed confidence interval or any deviation is explained by initialization/nonlinearity |
| 실패 시 해석 | ST residual order/sign, bias coupling, initial covariance, delayed-update implementation, large-error local linearization |

Required subcases:

1. identity/no bias/no noise;
2. constant rate/no bias;
3. stationary known bias;
4. moderate random initial attitude;
5. large initial error regression;
6. zero-latency vs delayed-packet replay.

---

## B3. Magnetometer analytic Jacobian vs finite difference

| 항목 | specification |
|---|---|
| 목적 | `H_m=[[h_m]_×,0]`의 sign/frame 검증 |
| truth condition | arbitrary normalized quaternion and nonzero inertial magnetic vectors; no noise |
| sensor configuration | ideal calibrated body vector |
| estimator configuration | measurement function only |
| 입력 파라미터 | at least 100 random `q,r_N`; perturbation `ε∈{1e-7,3e-7,1e-6,3e-6,1e-5}` rad |
| 변경 변수 | attitude, reference-vector direction/magnitude, perturbation axis/step |
| 고정 변수 | right error and residual `y-h` |
| 기록 로그 | analytic H, central-difference H, absolute/relative error |
| 평가 지표 | Frobenius error and column-wise sign |
| expected behavior | convergence to analytic H over stable epsilon range; rank 2 for nonzero vector |
| 합격 기준 | relative `≤1e-6` and absolute `≤1e-9` on normalized test scale; convention vector TV-M01 exact direction |
| 실패 시 해석 | `q_NB/q_BN`, active/passive, right/left error or residual sign mismatch |

Finite-difference definition: hold the nominal prediction `h_0=h(q)` fixed, generate perturbed **true measurements**, and form

\[
\nu_i^+=h(q\otimes Exp(+\epsilon e_i))-h_0,\qquad
\nu_i^-=h(q\otimes Exp(-\epsilon e_i))-h_0,
\]

\[
H_{:,i}^{FD}=\frac{\nu_i^+-\nu_i^-}{2\epsilon}.
\]

This differentiates the error-state residual with respect to the injected true right error. Perturbing the nominal estimate instead would introduce the opposite sign and is a different derivative.

---

## B4. Sun-vector analytic Jacobian vs finite difference

| 항목 | specification |
|---|---|
| 목적 | deterministic tangent basis `U`와 `H_s=U^T[h]_×` 검증 |
| truth condition | random quaternion/sun vectors away from numerical basis singularity; no eclipse/no noise |
| sensor configuration | ideal unit body sun vector; WLS excluded from this math test |
| estimator configuration | 2D tangent residual only |
| 입력 파라미터 | random cases and epsilon sweep identical to B3 |
| 변경 변수 | attitude, sun direction, basis branch selection |
| 고정 변수 | deterministic `U(h_hat)` at linearization point |
| 기록 로그 | `U`, orthogonality, analytic/FD H, residual |
| 평가 지표 | `||U^TU-I||`, `||U^Th||`, Jacobian error |
| expected behavior | rank 2; rotation about `h` unobservable; no discontinuous basis flip in local perturbations |
| 합격 기준 | orthogonality `≤1e-12`; Jacobian criteria as B3; TV-S01 sign passes |
| 실패 시 해석 | tangent-basis algorithm, re-evaluating `U` inconsistently during FD, normalization covariance omission |

A second integration test runs CSS raw→WLS and empirically estimates tangent `R_s`; it is not used to validate the analytic attitude Jacobian itself.

---

## B5. Quaternion injection/reset consistency

| 항목 | specification |
|---|---|
| 목적 | correction injection, new local error and covariance transport `J_r` 검증 |
| truth condition | arbitrary `q_hat`, injected rotations from `1e-8` to `0.5` rad |
| sensor configuration | none; synthetic correction |
| estimator configuration | injection/reset function only |
| 입력 파라미터 | random axes, covariance matrices, TV-INJ01/TV-RST01 |
| 변경 변수 | correction magnitude/direction, cross-covariance |
| 고정 변수 | right injection |
| 기록 로그 | pre/post relative error, `J_r`, finite-difference reset Jacobian, `P_c/P_plus` |
| 평가 지표 | log error, Jacobian error, symmetry/eigenvalues |
| expected behavior | if correction equals true local error, posterior error≈0; exact `J_r` matches FD; SPD preserved |
| 합격 기준 | convention-vector tolerance; reset FD relative error `≤1e-7` in stable epsilon range |
| 실패 시 해석 | wrong injection side, wrong sign in reset, quaternion log hemisphere, covariance mapped in wrong tangent |

---

## B6. Quaternion sign invariance

| 항목 | specification |
|---|---|
| 목적 | `q`/`-q`가 identical rotation/update/loss를 만드는지 확인 |
| truth condition | arbitrary attitudes including near-π; no noise and noisy ST packets |
| sensor configuration | duplicate ST packet pairs `{q_z,-q_z}` |
| estimator configuration | identical prior cloned into two runs |
| 입력 파라미터 | random q pairs, TV-SIGN01 |
| 변경 변수 | sign only |
| 고정 변수 | all floating inputs except sign |
| 기록 로그 | residual, K, correction, posterior rotation, metric |
| 평가 지표 | residual/posterior/metric difference |
| expected behavior | all physical outputs equal within roundoff |
| 합격 기준 | geodesic difference and correction difference `≤1e-12` reference |
| 실패 시 해석 | missing hemisphere alignment or component-MSE use |

---

## B7. Long-horizon quaternion norm/stability

| 항목 | specification |
|---|---|
| 목적 | repeated propagation/update에서 quaternion norm·drift·sign continuity 검증 |
| truth condition | bounded deterministic gyro sequence + zero/noisy sensor cases; duration at least multiple orbits or equivalent long replay |
| sensor configuration | gyro-only analytic composition, then UNIT-ST |
| estimator configuration | tuned MEKF |
| 입력 파라미터 | multiple `dt`, long seed set |
| 변경 변수 | duration, rate profile, update frequency |
| 고정 변수 | exact exponential and normalization policy |
| 기록 로그 | norm, dot with previous quaternion, geodesic error, hidden numerical events |
| 평가 지표 | max norm error, drift vs high-accuracy reference, sign flip count after continuity rule |
| expected behavior | norm bounded; no numerical divergence; sign changes do not affect rotation |
| 합격 기준 | norm TV-LONG01; attitude drift consistent with sensor/process model, not integration-order bug |
| 실패 시 해석 | missing normalization, incorrect delta quaternion, time unit, accumulated timestamp error |

---

## B8. Covariance symmetry/SPD

| 항목 | specification |
|---|---|
| 목적 | propagation, Joseph update, reset 후 `P/Q_d/R/S`의 numerical validity 검증 |
| truth condition | synthetic SPD matrices + UNIT-ST replay |
| sensor configuration | ST, then mag/sun stacked dimensions |
| estimator configuration | classical MEKF explicit covariance |
| 입력 파라미터 | random SPD `P,Q,R`, ill-conditioned but valid cases |
| 변경 변수 | condition number, dt, residual dimension, correction size |
| 고정 변수 | float64 reference, Joseph form |
| 기록 로그 | asymmetry, eigenvalues, Cholesky status, jitter |
| 평가 지표 | relative asymmetry, `λ_min`, solve residual |
| expected behavior | SPD maintained for SPD inputs; no unlogged clipping |
| 합격 기준 | asymmetry `≤1e-12`; no eigenvalue below negative roundoff threshold; all required Cholesky solves pass or failure is intentionally triggered test |
| 실패 시 해석 | wrong `Q_d`, simple covariance update cancellation, reset transport error, unit scaling/conditioning |

## 3. Package B exit gate

All B1–B8 must pass. Large-initial-error statistical convergence may remain a characterized limitation, but algebra, sign, reset, norm and SPD failures are blocking.

## 4. Package C — 최소 problem test

### Common event layout

Let trajectory duration be `T` (pilot default `T=600 s` unless sensor convergence requires longer).

- pre-event: `[0,0.4T)`
- event: `[0.4T,0.6T)`
- recovery: `[0.6T,T]`

Ramp events declare rise/fall time. Event timing is randomized around this template in Monte Carlo while remaining hidden from deployable estimators.

---

## C1. Matched fixed-noise baseline

| 항목 | specification |
|---|---|
| 목적 | 정상 조건에서 MEKF가 안정적이고 oracle context가 all-ones일 때 부당한 이득/차이가 없는지 확인 |
| truth condition | Tier-0 stationary noise: `α_g=α_b=α_R,j=1`, all inlier/valid |
| sensor configuration | first UNIT-ST, then MAIN-FUSION |
| estimator configuration | `F-TUNED`, `ORACLE-QR` with all-one context; optional `F-MIS` over/under variants |
| 입력 파라미터 | base symbolic/representative sensor profile |
| 변경 변수 | seeds, initial state, sensor scenario |
| 고정 변수 | context all ones, no event |
| 기록 로그 | common logs |
| 평가 지표 | state difference between tuned/oracle, stationary RMSE, NIS/NEES, divergence |
| expected behavior | `F-TUNED` and all-one `ORACLE-QR` are numerically identical; no adaptive oscillation |
| 합격 기준 | state/covariance replay difference within numerical tolerance; matched filter stable/consistent |
| 실패 시 해석 | oracle mapping changes baseline, implementation path mismatch, incorrect base matrices |

This test is a regression oracle: any future context implementation must reduce exactly to the classical baseline when its output is zero log-scale and full reliability.

---

## C2. Gyro process-uncertainty step

| 항목 | specification |
|---|---|
| 목적 | time-varying gyro process uncertainty가 fixed covariance를 악화시키고 correct-side oracle Q adaptation이 유용한지 확인 |
| truth condition | `α_g` step or ramp `>1` during event; `α_b=1`; measurement noise/validity fixed |
| sensor configuration | UNIT-ST first; repeat MAIN-FUSION |
| estimator configuration | `F-MIS`, `F-TUNED` fixed, `ORACLE-QR`, optional `WRONG-SIDE` inflating `R_ST` instead of `Q_g` |
| 입력 파라미터 | scale sweep `{mild,medium,severe}` chosen dimensionlessly; event duration/timing |
| 변경 변수 | `α_g`, rise time, seed |
| 고정 변수 | ST/mag/sun inlier `R`, bias process, motion profile |
| 기록 로그 | true/used scales, P blocks, innovations, NIS/NEES, correction/recovery |
| 평가 지표 | event peak/P95, recovery, bias/attitude RMSE, consistency |
| expected behavior | fixed underestimated Q becomes overconfident; oracle Q increases prior uncertainty and improves transition/recovery; wrong-side response differs |
| 합격 기준 | fixed degradation reproducible; oracle paired improvement on at least one primary transition metric without material stationary penalty |
| 실패 시 해석 | event too weak/short, ST rate dominates, base Q already overconservative, process scale maps to wrong block |

If all fixed filters remain unaffected, H1 “adaptation is needed” is not yet supported; do not solve by merely increasing neural complexity.

---

## C3. Measurement reliability step

| 항목 | specification |
|---|---|
| 목적 | inlier measurement noise degradation와 gross invalidity를 분리하고 `R` scale/gate가 필요한지 확인 |
| truth condition | primary subcase: `α_R,j>1` inlier step with gyro process fixed; secondary: outage/outlier requiring gate |
| sensor configuration | UNIT-ST with ST degradation first; then MAIN-FUSION mag or ST event |
| estimator configuration | `F-MIS`, `F-TUNED`, `ORACLE-QR`; `ROBUST` only for gross subcase; `WRONG-SIDE` Q inflation diagnostic |
| 입력 파라미터 | sensor ID, inlier scale sweep, outage/outlier interval |
| 변경 변수 | `α_R,j` or `ρ_j`, event timing/magnitude |
| 고정 변수 | gyro `α_g=α_b=1`, other sensors normal |
| 기록 로그 | sensor-specific innovation/NIS/gate/R, correction spikes |
| 평가 지표 | peak/P95, recovery, rejected inlier/outlier rates, NIS |
| expected behavior | inlier degradation: oracle R down-weights noisy measurement; gross invalid: gate prevents catastrophic correction; robust gate should not replace ordinary R scaling |
| 합격 기준 | correct mechanism improves event metric and does not reject excessive normal measurements; cause-specific logs match injection |
| 실패 시 해석 | event not observable, gate threshold incorrect, R units/frame wrong, sensor redundancy masks effect |

---

## C4. Slow drift + fast event simultaneous

| 항목 | specification |
|---|---|
| 목적 | slow bias/process change와 fast measurement/reliability event를 동시에 처리할 필요성 및 context interaction 확인 |
| truth condition | slow `α_b` ramp and temperature-calibration residual over trajectory + fast mag interference or ST outage/noise burst |
| sensor configuration | MAIN-FUSION |
| estimator configuration | fixed, oracle process-only, oracle measurement-only, oracle full 4-value, optional robust gate |
| 입력 파라미터 | slow ramp amplitude/time constant, fast event sensor/magnitude/duration, overlap fraction |
| 변경 변수 | overlap timing, event order, sensor ID |
| 고정 변수 | trajectory and unaffected sensor profiles |
| 기록 로그 | both context labels, bias/P blocks, sensor-specific R/gate, errors |
| 평가 지표 | pre-event bias error, fast peak, recovery, post-event steady error, false adaptation of unaffected sensors |
| expected behavior | process-only fails fast sensor issue; measurement-only fails slow bias uncertainty; full oracle handles both without global overinflation |
| 합격 기준 | full oracle is Pareto-improving or gives a clear tradeoff advantage over single-side oracle across paired trajectories |
| 실패 시 해석 | one event dominates, labels duplicate same effect, insufficient bias observability, dual context unnecessary |

This experiment tests the **need** for multiple physical targets, not yet the need for a dual-timescale neural architecture.

---

## C5. Matched innovation-RMS A/B identifiability pair

### C5.1 Goal

Create two different causes with nearly the same scalar innovation magnitude under the fixed baseline:

- **Case A:** gyro process uncertainty only increases; measurement noise fixed.
- **Case B:** ST measurement noise only increases; gyro process uncertainty fixed.

Use `UNIT-ST` first to avoid magnetometer observability confounds.

### C5.2 Specification

| 항목 | Case A | Case B |
|---|---|---|
| truth condition | `α_g=A>1`, `α_R,ST=1` | `α_g=1`, `α_R,ST=B>1` |
| bias process | fixed `α_b=1` | fixed `α_b=1` |
| ST validity | inlier, valid | inlier, valid |
| motion/trajectory | identical paired truth profile | identical paired truth profile |
| changed noise source | gyro samples | ST quaternion samples |
| correct oracle action | increase process `Q_g` | increase measurement `R_ST` |
| wrong-side action | inflate `R_ST` | inflate `Q_g` |

### C5.3 RMS matching procedure

1. choose a process scale `A` from the C2 medium condition;
2. run a pilot grid/bisection over `B`;
3. compute event-window

\[
r_{\nu}=\sqrt{\frac1N\sum_k\|\nu_{ST,k}\|^2}
\]

under the same fixed mismatched MEKF;
4. select `B` such that aggregate RMS differs from Case A by at most **5%**;
5. confirm overlap of per-event innovation-norm distributions and report autocorrelation/cross-feature differences;
6. freeze `A,B` before final Monte Carlo.

### C5.4 Estimators

- `F-MIS`
- case-specific `F-TUNED` or matched oracle reference
- `ORACLE-QR` correct-side
- `WRONG-SIDE`
- optional scalar/shared inflation baseline

No learned classifier/context estimator is trained.

### C5.5 Logs/metrics

- innovation vector/norm/RMS and temporal autocorrelation
- gyro increment statistics
- correction sequence and P decomposition
- attitude/bias peak/RMSE/recovery
- NIS/NEES
- correct-side vs wrong-side performance difference

### C5.6 Expected behavior

- scalar innovation RMS alone is nearly indistinguishable by construction;
- process event may show temporal propagation/cross-update effects and gyro-side evidence;
- measurement event directly changes ST scatter;
- correct-side oracle should outperform wrong-side adaptation in at least one consistency or state metric if separation matters.

### C5.7 Pass/interpretation

| finding | interpretation |
|---|---|
| RMS matched and correct-side≫wrong-side | supports separate process/measurement context and limits of innovation-magnitude-only estimator |
| RMS matched but correct/wrong side equivalent | separation may be unnecessary for this operating regime; simpler scalar adaptation favored |
| cannot match RMS | redesign scale/window; do not claim identifiability result |
| extra onboard features separate A/B | supports context estimator using temporal/gyro/sensor metadata, not innovation magnitude alone |
| even full oracle gives no state benefit | rejects practical need for this context mechanism in the tested regime |

This is structural evidence, not a proof that no innovation-history method can ever distinguish the causes.

## 5. Hypothesis decision table

| hypothesis | supported when | rejected/not supported when | next action |
|---|---|---|---|
| H1 time-varying adaptation is needed | fixed mismatched error/consistency degrades repeatedly under C2/C3/C4 | fixed/tuned classical remains adequate | reduce adaptation scope |
| H2 oracle context is useful | oracle improves paired transition/recovery/consistency with no material stationary penalty | oracle gives no repeated benefit | stop learned context/SNN |
| H3 process/measurement context separation is needed | full/correct-side oracle beats shared/wrong-side in C4/C5 | shared scalar performs equivalently | keep simpler context |
| H4 innovation-only context has a structural limit | matched RMS A/B require different optimal actions and magnitude-only evidence overlaps | A/B easily separated by magnitude or same action suffices | revise claim; use simpler feature |
| H5 robust gate is needed | gross outlier causes failure that R scale alone cannot prevent | inlier scaling handles all tested cases | gate only for validity |

## 6. Phase-1 entry/stop gates

### Mandatory to enter Phase 1A implementation

- math/convention documents approved;
- UNIT-ST truth/sensor packet schema approved;
- no hardware numeric value is presented as measured without provenance.

### Mandatory before neural baseline/context training

- all Package B tests pass;
- C1 classical matched baseline is stable;
- at least one C2/C3/C4 condition demonstrates a real adaptation gap;
- oracle context improves a declared primary metric;
- A/B test defines what can and cannot be identified from allowed features.

### Stop conditions

- algebra/reset/SPD failures;
- problem disappears under properly tuned classical filter;
- oracle context provides no benefit;
- context improvement relies on truth/event leakage;
- performance gain occurs only from unequal sensor streams or estimator knowledge.

## 7. Result report template

```yaml
experiment_id:
contract_version:
scenario_id:
truth_config_hash:
sensor_stream_hash:
estimator_id:
estimator_knowledge:
seed_manifest:
changed_variables:
fixed_variables:
metrics:
confidence_intervals:
pass_fail:
failure_interpretation:
artifacts:
```

## 8. Immediate execution order

1. B1, B3, B4, B5, B6
2. B7, B8
3. B2 UNIT-ST zero-noise then noisy
4. C1
5. C2
6. C3
7. C5 A/B pair
8. C4 combined event
9. only then decide context estimator architecture
