# Phase 1A Gate A MEKF Implementation Contract

- 작성일: 2026-07-31
- 기준 정책: `CURRENT_TREE_ACCEPTED_WITHOUT_HEAD_REVIEW`
- 구현 source of truth: `bench/estimators/mekf.py`
- 결정 상태: Gate A pure math/core 및 Amendment A1 구현·검증 완료
- 다음 Gate: 별도 승인 후 Gate B의 typed event schema와 UNIT-ST generator

## 1. 목적과 입력 근거

이 계약은 runner, sensor generator, model adapter와 독립적인 float64 6D kinematic MEKF reference core의 수학·배열·수치 경계를 고정한다.

입력 source of truth:

1. `docs/research/phase0a/decision_lock/P0A_PHASE_0A_SYNTHESIS.md`
2. `docs/research/phase0a/decision_lock/P0_01_DECISION_LEDGER.md`
3. `docs/research/phase0a/decision_lock/P0_02_TRUTH_SENSOR_ESTIMATOR_BOUNDARY.md`
4. `docs/research/phase0a/decision_lock/P0_05_MEKF_MATH_CONTRACT.md`
5. `docs/research/phase0a/decision_lock/P0_05_MEKF_CONVENTION_TEST_VECTORS.md`
6. `docs/research/phase0a/decision_lock/P0A_IMMEDIATE_TEST_SPEC.md`
7. Phase 1A audit 문서 3개
8. `docs/research/phase1a/prompts/02_CODE_AGENT_MATH_CORE_CURRENT_TREE_ACCEPTED_PROMPT.md`
9. `docs/research/phase1a/P1A_GATE_A_CHAT_REVIEW.md`
10. `docs/research/phase1a/prompts/02A_CODE_AGENT_GATE_A_HARDENING_EXACT_PI_PROMPT.md`

현재 working tree 전체는 사용자 승인 기준선으로 사용했다. 과거 HEAD, commit history, commit delta는 승인 판단에 사용하지 않았다.

## 2. Locked convention

| 항목 | 고정 정의 |
|---|---|
| Filter | 6D kinematic MEKF |
| Nominal state | `[q_NB, b_g]`, 저장 차원 4+3 |
| Local error | `[delta_theta, delta_b_g] in R^6` |
| Quaternion | scalar-first Hamilton `[w,x,y,z]` |
| Attitude | active body-to-navigation `q_NB` |
| DCM | `R_NB(q)` maps B coordinates to N; `C_BN=R_NB^T` |
| Multiplicative error | right error, `q_true=q_hat⊗Exp_q(delta_theta)` |
| Gyro | `omega_m=omega_true+b_g+n_g`, propagation uses `omega_m-b_hat` |
| Bias error | `delta_b_g=b_true-b_hat` |
| Covariance | 6x6, same right/local tangent as the error state |
| Update | star-tracker three-dimensional tangent residual |
| Covariance update | Joseph form followed by exact right-reset transport |
| Numeric reference | NumPy/SciPy float64 |

No function may locally reinterpret the quaternion as scalar-last, passive, `q_BN`, or left-multiplicative.

## 3. 수식과 함수의 1:1 mapping

| 수학 기능 | Locked expression | 실제 함수/class |
|---|---|---|
| skew | `[a]_x b=a×b` | `bench.estimators.mekf.skew` |
| quaternion normalization | `q/||q||` | `quat_normalize` |
| Hamilton product | scalar/vector Hamilton product | `quat_multiply` |
| conjugate/inverse | `q*`, `q*/||q||^2` | `quat_conjugate`, `quat_inverse` |
| SO(3) exponential | `[cos(theta/2), sin(theta/2) phi/theta]` | `quat_exp` |
| shortest-arc logarithm | ordinary hemisphere selection; exact-pi tie uses deterministic vector-axis sign; then `2 atan2(||q_v||,q_0) q_v/||q_v||` | `quat_log` |
| hemisphere alignment | flip measurement when `q_z^T q_hat<0` | `align_quaternion` |
| active DCM | `(q0^2-qv^Tqv)I+2qvqv^T+2q0[qv]_x` | `quat_to_dcm` |
| DCM inverse conversion | proper `R_NB` to deterministic scalar-first quaternion | `dcm_to_quat` |
| physical quaternion distance helper | `2 acos(|q^T p|)` | `quat_geodesic_angle` |
| right Jacobian | `J_r=I-A[phi]_x+B[phi]_x^2` | `right_jacobian_so3` |
| body-vector prediction | `h=C_BN(q) r_N=R_NB(q)^T r_N` | `body_vector_prediction` |
| body-vector Jacobian | `H=[[h]_x,0]` | `body_vector_jacobian` |
| sun tangent basis | deterministic `U`, `U^TU=I`, `U^Th=0` | `sun_tangent_basis` |
| sun tangent Jacobian | `H_s=[U^T[h]_x,0]` | `sun_tangent_jacobian` |
| ST residual | `Log_q(q_hat^-1⊗q_z_aligned)` | `star_tracker_residual` |
| continuous error matrices | `F=[-[omega]_x,-I;0,0]`, `G=[-I,0;0,I]` | `continuous_error_matrices` |
| continuous PSD | `Q_c=blkdiag(S_g,S_b)` | `continuous_noise_covariance` |
| transition | `Phi=exp(F dt)` | `discretize_van_loan` |
| discrete covariance | locked Van Loan `Q_d=E12 Phi^T` | `discretize_van_loan` |
| SPD solve | Cholesky plus triangular solves | `cholesky_solve_spd` |
| innovation/gain | `S=HPH^T+R`, solve `K S=P H^T` | `kalman_gain` |
| Joseph covariance | `(I-KH)P(I-KH)^T+KRK^T` | `joseph_covariance_update` |
| injection | `q+=normalize(q-⊗Exp(delta_theta_hat))`, `b+=b-+delta_b_hat` | `inject_error_state` |
| reset | `G_reset=blkdiag(J_r(delta_theta_hat),I)` | `reset_covariance` |
| nominal/covariance propagation | gyro correction, quaternion propagation, `Phi P Phi^T+Q_d` | `propagate_state` |
| complete ST update | residual, gain, Joseph, injection, reset | `star_tracker_update` |
| covariance diagnostics | asymmetry, eigenvalue, Cholesky status | `covariance_diagnostics`, `assert_positive_definite`, `assert_positive_semidefinite` |

## 4. State, array, dtype, unit, frame contract

### 4.1 Nominal and covariance

| 값 | Shape | dtype | Unit | Frame/meaning |
|---|---:|---|---|---|
| `MEKFState.q_NB` | `(4,)` | float64 | unitless | active B-to-N unit quaternion |
| `MEKFState.b_g` | `(3,)` | float64 | rad/s | body frame |
| `MEKFState.P` | `(6,6)` | float64 | mixed | covariance of `[rad, rad/s]` right-local error |
| `omega_m` | `(3,)` | float64 | rad/s | body frame gyro measurement |
| `delta_x` | `(6,)` | float64 | `[rad, rad/s]` | right/local body tangent and body bias |

`MEKFState` initialization normalizes `q_NB`, independently copies all arrays, requires `P` to be finite, symmetric, and strictly SPD, and stores `q_NB`, `b_g`, and `P` as non-writeable NumPy arrays. Caller arrays do not alias the state.

### 4.2 Process model

| 값 | Shape | Unit/meaning |
|---|---:|---|
| `F`, `G`, `Phi` | `(6,6)` | local error dynamics/transition |
| `S_g` | `(3,3)` | gyro white-noise PSD, rad^2/s |
| `S_b` | `(3,3)` | bias random-walk PSD, rad^2/s^3 |
| `Q_c`, `Q_d` | `(6,6)` | continuous/discrete local process covariance |
| `dt` | scalar | seconds, finite and nonnegative |

`Q_c`와 `Q_d`는 PSD일 수 있다. `P`는 strictly SPD여야 한다.

### 4.3 Measurement model

| 값 | Shape | Unit/meaning |
|---|---:|---|
| body reference/prediction | `(3,)` | caller-declared vector unit, N input/B output |
| body-vector `H` | `(3,6)` | residual derivative with respect to right error |
| sun `U` | `(3,2)` | dimensionless deterministic tangent basis |
| sun `H_s` | `(2,6)` | 2D tangent residual derivative |
| ST residual | `(3,)` | rad, local attitude tangent |
| `R_ST`, `S` | `(3,3)` | rad^2, strictly SPD |
| `K_ST` | `(6,3)` | mixed state-error units per rad |

## 5. Prior/posterior notation and operation order

1. `MEKFState` entering propagation is posterior `(q_k^+, b_k^+, P_k^+)`.
2. `propagate_state` computes `omega_hat=omega_m-b_k^+`.
3. It returns prior `(q_{k+1}^-, b_{k+1}^-, P_{k+1}^-)` with `b^- = b^+`.
4. `star_tracker_update` receives a prior state.
5. It computes local residual, `S`, `K`, and `delta_x_hat`.
6. Joseph covariance `P_c` remains in the pre-injection tangent.
7. The correction is right-injected into `q^-`; bias correction is additive.
8. `P_c` is transported by exact `J_r` to posterior tangent `P^+`.
9. The local error mean is reset to zero; it is not stored as a nominal state component.

## 6. Normalization and sign policy

Quaternion normalization occurs only at these explicit boundaries:

- `MEKFState` initialization
- `quat_exp` result, to remove series/roundoff norm error
- `quat_to_dcm`, `quat_log`, and hemisphere helper inputs
- propagation result after Hamilton composition
- injection result after Hamilton composition
- ST measurement before estimate-relative hemisphere alignment

Time-series quaternions are not globally forced to `q0>=0`. `dcm_to_quat` chooses a deterministic nonnegative-scalar representative only because a single DCM has two valid quaternion representatives. Physical comparisons and ST updates remain antipodal-invariant.

For `quat_log`, the exact-pi scalar tie tolerance is

```text
EXACT_PI_TIE_TOL = 8 * eps(float64)
                   = 1.7763568394002505e-15
```

When `abs(q0)` is no larger than this tolerance, the first vector component whose absolute value exceeds the same tolerance is made positive by selecting `q` or `-q`; exact zero components are canonicalized to positive zero. Outside this machine-roundoff-scale scalar tie region, the existing nonnegative-scalar shortest-arc rule is unchanged.

This rule removes `q/-q` representation dependence at exactly pi. The SO(3) logarithm axis at exactly 180 degrees remains mathematically non-unique, and this software convention does not claim that a local MEKF always converges from an exact 180-degree initial attitude error. The large-initial-error convergence threshold remains an experimental TBD.

## 7. Numerical safety and correction ledger

### 7.1 Fail-loud rules

- All public numeric arrays must have the declared shape and finite float64 values.
- `P`, `R`, `S`, `P_c`, `P_plus` require successful Cholesky factorization.
- `Q_c` and `Q_d` require symmetry and no eigenvalue below the negative PSD tolerance.
- No pseudo-inverse, explicit matrix inverse, eigenvalue clipping, silent diagonal perturbation, or non-SPD repair exists.
- Invalid input raises `ValueError` or `NumericalSafetyError` before the caller's state object is mutated.
- `MEKFState` owns defensive copies whose writeable flags are false; predict/update return new state objects and do not mutate the prior.

### 7.2 Allowed corrections

| Correction | Location | Limit/evidence |
|---|---|---|
| Quaternion normalization | initialization, Exp, propagation, injection, conversion/log boundaries | unit norm test tolerance `1e-12` |
| Covariance symmetrization | `Q_d`, propagated `P`, `S`, Joseph `P_c`, reset `P_plus` | input relative asymmetry must be `<=1e-12`; relative correction is returned in result objects |
| Small-angle series | `quat_exp`, `right_jacobian_so3`, `quat_log` zero limit | near-zero round-trip tests |

No jitter or covariance eigenvalue modification was used.

## 8. Truth/sensor/estimator information boundary

The core accepts only:

- nominal estimate and local covariance
- gyro measurement, elapsed time, and caller-supplied nominal `Q_c`
- star-tracker quaternion measurement and caller-supplied nominal `R_ST`
- reference vectors for pure measurement-function/Jacobian tests

The core API does not accept true attitude, true bias, injected event label, oracle Q/R scale, future measurement, sensor validity cause, or evaluation metric inputs. Tests may create synthetic truth outside the core and compare outputs, but that truth is never passed into estimator functions.

## 9. 이번 Gate A에서 구현하지 않은 기능

- Basilisk truth or sensor interface
- gyro/ST event packet schema, timestamp ordering, latency, OOSM buffer
- UNIT-ST scenario generation or replay
- magnetometer/sun hardware model or CSS WLS
- gating, robust outlier logic, sensor validity handling
- canonical metric module, NIS/NEES aggregation
- model adapter, registry, runner, cache, YAML
- learned context, ANN, SNN, FPGA, Package C
- optional Gauss-Markov bias profile
- built-wheel packaging changes

These remain separate gates. Passing Gate A authorizes no runner or sensor integration.

## 10. Gate A Amendment A1

Chat independent review found two boundary defects in the original 42-test Gate A result:

1. an explicitly zero-scalar exact-pi measurement could return opposite residual/correction signs for `q_z` and `-q_z`;
2. frozen `MEKFState` fields still contained writeable NumPy arrays.

Amendment A1 adds only the exact-pi tie convention described in §6 and defensive-copy/read-only state storage described in §4.1. Exact-pi x/y/z/arbitrary-axis measurement antipodes, nominal quaternion antipodes, both sides of near pi outside the tie, caller aliasing, direct writes, and failed/successful predict/update immutability are covered by the amended test suite. No Gate B API or integration was added.

## 11. 남은 TBD와 다음 Gate

Gate A에 남은 수학 TBD는 없다. Phase 0A에서 이미 보류한 actual `Q_c/R` numerical profile, large-error initialization threshold, delayed-measurement buffer policy는 이 core에서 결정하지 않았다.

다음 작업은 Chat 검토와 별도 프롬프트 승인 후 Gate B로 진행한다. Gate B는 typed event schema, simulator-to-`q_NB` basis-vector proof, deterministic UNIT-ST sensor stream, trajectory split/cache identity를 다뤄야 한다.
