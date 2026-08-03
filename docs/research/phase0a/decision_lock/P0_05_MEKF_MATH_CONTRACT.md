# P0_05 Kinematic MEKF 수학 계약

> 작성일: 2026-07-30  
> 표지 규칙: **[확인]** 실험·실측, **[문헌]** 논문·공식 자료, **[분석]** 수식·구조 해석, **[가설]** 검증 대상, **[결정]** 설계 선택, **[보류]** 근거 부족·후속 범위

| 항목 | 내용 |
|---|---|
| 목적 | nominal quaternion+gyro-bias state, 6D local error, propagation, sensor update, injection/reset, loss와 consistency 지표를 한 convention으로 완전히 고정한다. |
| 입력 근거 | S0–S4, E2; sensor 식은 `P0_04_SENSOR_ROLE_AND_MODEL_SPEC.md` |
| 결정 상태 | LOCK — quaternion/frame/error convention과 모든 1차 수식 |
| 남은 TBD | 실제 `Q_c/R_j` 수치, delayed-measurement buffer 구현 방식, large-error initialization policy의 수치 threshold |
| 다음 Gate | 독립 구현이 `P0_05_MEKF_CONVENTION_TEST_VECTORS.md`와 Package B를 모두 통과 |


## 1. Contract summary

| 항목 | 고정 정의 |
|---|---|
| quaternion algebra | Hamilton product, scalar-first `[q_0,q_x,q_y,q_z]` |
| attitude | `q_NB`: body `B` → inertial `N` active rotation |
| DCM | `R_NB(q)` maps B coordinates to N; `C_BN=R_NB^T` maps N to B |
| angular rate | `ω_BN^B`: body relative inertial, expressed in body, rad/s |
| local attitude error | right multiplicative: `q_true=q_hat⊗δq(δθ)` |
| nominal state | `x̄=(q_NB,b_g)`, dimensions `4+3` but covariance dimension `6` |
| error state | `δx=[δθ^T,δb_g^T]^T∈R^6` |
| gyro role | propagation input |
| vector updates | mag raw 3D; sun 2D tangent residual |
| ST update | quaternion relative log residual, 3D |
| injection | `q_hat+=q_hat-⊗Exp_q(δθ_hat)` |
| reset | local error mean→0, covariance transported by right Jacobian |

## 2. 기본 연산과 표기

### 2.1 Skew matrix

\[
[a]_\times=\begin{bmatrix}
0&-a_3&a_2\\
a_3&0&-a_1\\
-a_2&a_1&0
\end{bmatrix},\qquad [a]_\times b=a\times b.
\]

`a,b∈R^3`; 회전벡터이면 단위 rad, 방향벡터이면 해당 물리 단위다.

### 2.2 Hamilton quaternion product

\[
q\otimes p=
\begin{bmatrix}
q_0p_0-q_v^Tp_v\\
q_0p_v+p_0q_v+q_v\times p_v
\end{bmatrix}.
\]

\[
q^{-1}=q^*=[q_0,-q_v^T]^T\quad\text{for }\|q\|=1.
\]

### 2.3 SO(3) exponential/logarithm

For `φ∈R^3`, `θ=||φ||`:

\[
\operatorname{Exp}_q(\phi)=
\begin{bmatrix}
\cos(\theta/2)\\
\frac{\sin(\theta/2)}{\theta}\phi
\end{bmatrix},
\]

with the continuous zero-angle limit. The shortest-arc logarithm first chooses the hemisphere with nonnegative scalar part:

\[
\operatorname{Log}_q(q)=
2\,\frac{\operatorname{atan2}(\|q_v\|,q_0)}{\|q_v\|}q_v.
\]

### 2.4 Active rotation matrix

\[
R_{NB}(q)=
(q_0^2-q_v^Tq_v)I+2q_vq_v^T+2q_0[q_v]_\times.
\]

Thus

\[
v^N=R_{NB}(q)v^B,
\qquad
v^B=C_{BN}(q)v^N=R_{NB}(q)^Tv^N.
\]

### 2.5 Quaternion normalization/sign

- propagation과 injection 후 `q←q/||q||`.
- time series에서는 `q_k^Tq_{k-1}<0`이면 `q_k←-q_k`로 연속 hemisphere를 유지한다.
- measurement update 전에 `q_z^Tq_hat<0`이면 `q_z←-q_z`.
- metric과 loss는 `|q_hat^Tq_true|`를 사용한다.
- **[결정]** 매 시각 무조건 `q_0≥0`로 강제하지 않는다. `180°` 부근의 불연속을 피하기 위해 temporal/estimate-relative hemisphere를 쓴다.

## 3. State와 units

### 3.1 Nominal state

\[
\bar x_k=(\hat q_{NB,k},\hat b_{g,k}^B).
\]

| 변수 | 차원 | 단위 | frame | 시각 표기 |
|---|---:|---|---|---|
| `q_NB` | 4 | unitless | B→N | `k`, `-`, `+` |
| `b_g^B` | 3 | rad/s | B | `k`, `-`, `+` |

### 3.2 Error state

\[
\delta x_k=\begin{bmatrix}\delta\theta_k^B\\\delta b_{g,k}^B\end{bmatrix}
\in\mathbb R^6,
\quad
q_{true}=\hat q\otimes\operatorname{Exp}_q(\delta\theta).
\]

| block | 차원 | 단위 | 의미 |
|---|---:|---|---|
| `δθ` | 3 | rad | estimate의 right/local body tangent에서 true attitude까지의 rotation vector |
| `δb_g` | 3 | rad/s | `b_true-b_hat` |
| `P=E[δxδx^T]` | 6×6 | mixed | 위 local error covariance |

## 4. Gyro measurement와 nominal propagation

### 4.1 Baseline gyro model

\[
\omega_m^B=\omega_{true}^B+b_g^B+n_g^B.
\]

- `ω_m,ω_true,b_g,n_g∈R^3`, rad/s, body frame.
- `n_g`는 continuous white-noise equivalent이며 sampled sensor model과 bandwidth 연결을 별도 기록한다.

Corrected rate:

\[
\hat\omega_k^B=\omega_{m,k}^B-\hat b_{g,k}^{B,+}.
\]

### 4.2 Quaternion kinematics

\[
\dot q_{NB}=\tfrac12 q_{NB}\otimes[0,(\omega_{BN}^{B})^T]^T.
\]

Piecewise-constant discrete propagation over `[t_k,t_{k+1})`:

\[
\hat q_{k+1}^{-}=\operatorname{normalize}\left(
\hat q_k^+\otimes\operatorname{Exp}_q(\hat\omega_k\Delta t_k)
\right).
\]

### 4.3 Gyro-bias process

Phase-1 baseline random walk:

\[
\dot b_g=w_b,
\qquad
\hat b_{g,k+1}^{-}=\hat b_{g,k}^{+}.
\]

Optional Gauss–Markov profile:

\[
\dot b_g=-\tau_b^{-1}b_g+w_b,
\]

is activated only when a source supports `τ_b`; its nominal-state propagation and `F` bottom-right block then change accordingly.

## 5. Continuous-time error dynamics

With

\[
\delta b_g=b_{g,true}-\hat b_g,
\]

first-order right-error dynamics are

\[
\delta\dot\theta
=-[\hat\omega]_\times\delta\theta-\delta b_g-n_g,
\]

\[
\delta\dot b_g=w_b.
\]

Therefore

\[
\delta\dot x=F\delta x+Gw,
\]

\[
F=
\begin{bmatrix}
-[\hat\omega]_\times&-I_3\\
0_3&0_3
\end{bmatrix}\in\mathbb R^{6\times6},
\quad
G=
\begin{bmatrix}
-I_3&0_3\\
0_3&I_3
\end{bmatrix}\in\mathbb R^{6\times6},
\]

\[
w=\begin{bmatrix}n_g\\w_b\end{bmatrix},
\qquad
Q_c=E[w(t)w(t')^T]/\delta(t-t')
=\operatorname{blkdiag}(S_g,S_b).
\]

| quantity | 차원 | 단위 |
|---|---:|---|
| `F` | 6×6 | block-dependent; maps `[rad,rad/s]` to derivatives |
| `G` | 6×6 | sign/selection matrix |
| `S_g` | 3×3 | rad²/s |
| `S_b` | 3×3 | rad²/s³ |
| `Q_c` | 6×6 | corresponding continuous PSD blocks |

For the optional GM bias, `F_bb=-τ_b^{-1}I`.

## 6. Discretization

Assume `F,G,Q_c` constant over `Δt`:

\[
\Phi_k=\exp(F_k\Delta t_k)\in\mathbb R^{6\times6},
\]

\[
Q_{d,k}=\int_0^{\Delta t_k}
\exp(F_k\tau)GQ_cG^T\exp(F_k^T\tau)d\tau
\in\mathbb R^{6\times6}.
\]

### 6.1 Van Loan contract

Let `L=GQ_cG^T` and

\[
\mathcal M=\begin{bmatrix}F&L\\0&-F^T\end{bmatrix}\Delta t,
\qquad
\exp(\mathcal M)=\begin{bmatrix}E_{11}&E_{12}\\0&E_{22}\end{bmatrix}.
\]

Then

\[
\Phi=E_{11},\qquad Q_d=E_{12}\Phi^T.
\]

Numerical contract:

1. `Q_d←(Q_d+Q_d^T)/2` only for roundoff symmetry.
2. negative eigenvalue below the configured roundoff tolerance is a test failure, not silently clipped.
3. any diagonal jitter used for Cholesky is logged with magnitude and reason.

### 6.2 Covariance propagation

\[
P_{k+1}^{-}=\Phi_k P_k^+\Phi_k^T+Q_{d,k}.
\]

`P∈R^{6×6}` is in the local tangent of the propagated nominal attitude.

## 7. Generic measurement update

At measurement event `j` with residual dimension `m_j`:

\[
\nu_j=y_j-h_j(\hat x^- )\in\mathbb R^{m_j},
\]

\[
H_j=\frac{\partial\nu_j}{\partial\delta x}\bigg|_0
\in\mathbb R^{m_j\times6},
\quad
R_j\in\mathbb R^{m_j\times m_j},
\]

\[
S_j=H_jP^-H_j^T+R_j\in\mathbb R^{m_j\times m_j},
\]

\[
K_j=P^-H_j^TS_j^{-1}\in\mathbb R^{6\times m_j},
\]

\[
\widehat{\delta x}=K_j\nu_j.
\]

Joseph covariance before reset:

\[
P_c=(I-KH)P^-(I-KH)^T+KRK^T.
\]

**[결정]** validity/gross-outlier gate is evaluated before this update. A hard invalid measurement produces no correction and no covariance reduction.

## 8. Magnetometer vector update

Reference field `m^N∈R^3` and calibrated body measurement `y_m^B∈R^3`:

\[
h_m(\hat q)=C_{BN}(\hat q)m_{model}^N\in\mathbb R^3.
\]

Residual:

\[
\nu_m=y_m^B-h_m(\hat q)\in\mathbb R^3.
\]

For right error,

\[
H_m=
\begin{bmatrix}
[h_m]_\times&0_{3\times3}
\end{bmatrix}
\in\mathbb R^{3\times6},
\qquad R_m\in\mathbb R^{3\times3}.
\]

**[분석]** `rank([h_m]_×)=2`; a single instantaneous vector does not independently observe rotation about that vector. `S_m` can still be SPD because `R_m` is SPD. Full trajectory observability depends on changing field direction, motion, bias coupling and other sensors.

## 9. Sun-vector update

Let calibrated/WLS unit measurement and prediction be

\[
y_s^B\in S^2,\qquad h_s=C_{BN}(\hat q)s_{model}^N\in S^2.
\]

Choose `U(h_s)∈R^{3×2}` with

\[
U^TU=I_2,\qquad U^Th_s=0.
\]

Tangent residual:

\[
\nu_s=U^T(y_s^B-h_s)=U^Ty_s^B\in\mathbb R^2.
\]

Jacobian:

\[
H_s=
\begin{bmatrix}
U^T[h_s]_\times&0_{2\times3}
\end{bmatrix}
\in\mathbb R^{2\times6}.
\]

If the 3D pre-normalization covariance is `R_{s,3}`, then

\[
R_s=U^TR_{s,3}U\in\mathbb R^{2\times2}
\]

or it is obtained directly from CSS→WLS Monte Carlo tangent errors. Eclipse/FOV invalidity skips the update.

**[결정]** `U` is generated deterministically: choose the Cartesian basis least aligned with `h_s`, cross to make the first tangent axis, then complete a right-handed orthonormal pair. This avoids numerical singularity and implementation-dependent basis flips.

## 10. Star-tracker quaternion update

After mounting/convention conversion, measurement `q_{z,NB}` represents B→N attitude.

1. Normalize `q_z`.
2. If `q_z^T\hat q^-<0`, set `q_z←-q_z`.
3. Relative quaternion:

\[
\delta q_z=(\hat q^-)^{-1}\otimes q_z.
\]

4. Tangent residual:

\[
\nu_{ST}=\operatorname{Log}_q(\delta q_z)\in\mathbb R^3.
\]

5. Linearization:

\[
H_{ST}=\begin{bmatrix}I_3&0_3\end{bmatrix}
\in\mathbb R^{3\times6},
\qquad R_{ST}\in\mathbb R^{3\times3}\;[\mathrm{rad}^2].
\]

This residual is valid as a local Gaussian measurement. A false solution is not absorbed by simply enlarging `R_ST`; it is handled by robust/gating logic.

## 11. Multiplicative injection and local reset

Partition

\[
\widehat{\delta x}=
\begin{bmatrix}\widehat{\delta\theta}\\\widehat{\delta b_g}\end{bmatrix}.
\]

Inject:

\[
\hat q^+=\operatorname{normalize}\left(
\hat q^-\otimes\operatorname{Exp}_q(\widehat{\delta\theta})
\right),
\]

\[
\hat b_g^+=\hat b_g^-+\widehat{\delta b_g}.
\]

Reset the local error mean to zero. The exact first-order covariance transport for finite injected rotation uses the SO(3) right Jacobian:

\[
G_{reset}=\operatorname{blkdiag}
\left(J_r(\widehat{\delta\theta}),I_3\right),
\]

\[
J_r(\phi)=I-
\frac{1-\cos\theta}{\theta^2}[\phi]_\times+
\frac{\theta-\sin\theta}{\theta^3}[\phi]_\times^2,
\quad\theta=\|\phi\|.
\]

Small-angle form:

\[
J_r(\phi)\simeq I-\tfrac12[\phi]_\times+\tfrac16[\phi]_\times^2.
\]

Posterior covariance:

\[
P^+=G_{reset}P_cG_{reset}^T,
\qquad P^+\leftarrow\tfrac12(P^++P^{+T}).
\]

**[결정]** code uses exact `J_r` with a small-angle series near zero; the first-order matrix is used as an analytic regression check.

## 12. Asynchronous/multirate ordering

### 12.1 Event order

For each gyro interval `[t_k,t_{k+1})`:

1. use the gyro sample assigned to that interval;
2. propagate nominal state and covariance to the next measurement time;
3. process all valid measurements whose **measurement timestamp** equals that time;
4. inject/reset once per stacked independent update or after each sequential relinearized update;
5. log posterior.

### 12.2 Same-time measurements

Default: after sensor-specific validity/gate, stack independent residuals:

\[
\nu=\operatorname{col}(\nu_{j_1},\ldots),\quad
H=\operatorname{col}(H_{j_1},\ldots),\quad
R=\operatorname{blkdiag}(R_{j_1},\ldots).
\]

If cross-sensor correlation is modeled, the full `R` is required. Sequential updates are allowed only with a fixed order and relinearization; the order becomes part of the contract.

### 12.3 Delayed measurement

- measurement timestamp and arrival time are distinct.
- Tier 1 uses a state/covariance/gyro buffer and repropagates after delayed ST update, or a documented fixed-lag equivalent.
- stale data is never applied as though measured at current time.
- if the delay exceeds buffer horizon, skip and log the reason.

## 13. Sensor-specific dimensions

| sensor | residual `m` | `H` | `R` | `K` |
|---|---:|---:|---:|---:|
| magnetometer raw vector | 3 | 3×6 | 3×3 | 6×3 |
| sun tangent vector | 2 | 2×6 | 2×2 | 6×2 |
| star tracker tangent attitude | 3 | 3×6 | 3×3 | 6×3 |
| stacked mag+sun+ST | 8 | 8×6 | 8×8 | 6×8 |

## 14. Loss and evaluation metrics

### 14.1 SO(3) geodesic error/loss

\[
q_e=\hat q^{-1}\otimes q_{true},
\]

\[
e_R=\operatorname{Log}_q(q_e),
\quad
\theta_e=\|e_R\|=
2\arccos\left(\operatorname{clip}(|\hat q^Tq_{true}|,0,1)\right).
\]

Primary attitude loss:

\[
\mathcal L_R=\theta_e^2
\]

or robust `Huber(θ_e)` for training ablation. Quaternion component MSE is not the primary metric.

### 14.2 Bias metric

\[
e_b=b_{g,true}-\hat b_g,
\quad RMSE_b=\sqrt{E[\|e_b\|^2/3]}.
\]

Report SI `rad/s` and optionally converted `deg/h` with explicit conversion.

### 14.3 NIS

For an accepted inlier update:

\[
\epsilon_{NIS}=\nu^TS^{-1}\nu.
\]

Use only when `S` is explicit SPD and residual/model assumptions are defined. Degrees of freedom are residual dimension (`3` mag raw, `2` sun tangent, `3` ST). Normalization, gating and mixture outliers alter the theoretical distribution; report accepted-set and pre-gate statistics separately.

### 14.4 NEES

Posterior true local error:

\[
e_x^+=\begin{bmatrix}
\operatorname{Log}_q((\hat q^+)^{-1}\otimes q_{true})\\
b_{g,true}-\hat b_g^+
\end{bmatrix},
\]

\[
\epsilon_{NEES}=e_x^{+T}(P^+)^{-1}e_x^+.
\]

Conditions:

1. truth available only for evaluation;
2. error and `P` use the same right/local tangent;
3. `P` includes reset transport;
4. ensemble confidence intervals use the correct sample count and 6 DOF;
5. pure direct-gain Split-KalmanNet without a validated explicit `P` cannot claim NEES consistency.

## 15. Opposite convention을 사용할 때 바뀌는 것

Switching to `q_BN` or left-multiplicative error requires a complete migration, not isolated sign edits:

- quaternion propagation multiplication order
- reference vector prediction (`R` vs `R^T`)
- vector Jacobian sign/frame
- ST relative quaternion order
- injection side
- error dynamics `F`
- reset Jacobian and covariance tangent
- test vectors, adapters, training features

**[결정]** no module may redefine these independently. Basilisk/external-library conventions are converted at one boundary adapter and verified by basis-vector tests.

## 16. Numerical safety contract

- use float64 for reference implementation and unit tests;
- normalize quaternion after every propagation/injection;
- solve `Sx=b` by Cholesky/linear solve, never explicit inverse in implementation;
- Joseph form for covariance update;
- symmetrize only roundoff, log any jitter;
- reject NaN/Inf and invalid timestamp before state mutation;
- assert monotonic measurement time inside each replay sequence;
- `P,Q,R` unit/frame metadata are immutable with the config.

## 17. Required unit-test behavior

| test | expected behavior |
|---|---|
| zero motion | exact no-noise case preserves identity attitude and bias; no drift |
| constant angular rate | quaternion matches analytic exponential and norm=1 |
| known constant gyro bias | with ST, bias converges; with exact initialized bias, no attitude drift |
| small-angle update | correction direction equals analytic residual/Jacobian |
| large initial attitude error | finite, sign-consistent behavior; convergence is tested separately and not guaranteed by local linearization |
| `q` and `-q` | identical residual, update, loss and metric |
| update then reset | injected correction is removed from local error and covariance transported by `J_r` |
| covariance symmetry | Joseph+reset result symmetric within roundoff |
| covariance positive definiteness | SPD under SPD initialization/Q/R; no hidden clipping |
| long horizon quaternion norm | norm remains within configured float64 tolerance with normalization |

## 18. Gate

- [ ] all convention test vectors pass in two independent implementations or reference vs implementation.
- [ ] mag/sun analytic Jacobians match finite differences.
- [ ] ST residual returns injected right error for small angles.
- [ ] exact `J_r` and finite-difference reset map agree.
- [ ] asynchronous event replay is deterministic.
- [ ] NIS/NEES are disabled for models lacking valid explicit covariance.
