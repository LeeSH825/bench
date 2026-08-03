# P0_05 MEKF Convention Test Vectors

> 작성일: 2026-07-30  
> 표지 규칙: **[확인]** 실험·실측, **[문헌]** 논문·공식 자료, **[분석]** 수식·구조 해석, **[가설]** 검증 대상, **[결정]** 설계 선택, **[보류]** 근거 부족·후속 범위

| 항목 | 내용 |
|---|---|
| 목적 | quaternion ordering, multiplication, frame direction, vector Jacobian, propagation, injection/reset과 sign invariance를 숫자로 고정한다. |
| 입력 근거 | E2 및 `P0_05_MEKF_MATH_CONTRACT.md` |
| 결정 상태 | LOCK — 모든 숫자와 expected direction |
| 남은 TBD | target language/library별 floating-point tolerance 조정 |
| 다음 Gate | reference implementation과 production implementation이 동일 vector에서 허용오차 내 일치 |


## 1. Global test settings

```yaml
quaternion_order: [w, x, y, z]
product: Hamilton
attitude: q_NB, active B_to_N
error: right_multiplicative
angle_unit: rad
floating_point_reference: float64
```

Recommended provisional tolerances:

| 항목 | tolerance |
|---|---:|
| scalar/vector algebra | absolute `1e-12` |
| quaternion norm after normalization | `|norm-1| ≤ 1e-12` |
| analytic vs central finite-difference Jacobian | relative Frobenius `≤1e-6` and absolute `≤1e-9` where nonzero |
| covariance relative asymmetry | `≤1e-12` |
| reset finite-difference Jacobian | relative `≤1e-7` for `ε∈[1e-8,1e-6]` |

These are software-test settings, not sensor-performance claims.

## 2. TV-Q01 — Identity

Input:

```text
q = [1, 0, 0, 0]
v_B = [1, 2, 3]
```

Expected:

```text
R_NB = I3
C_BN = I3
v_N = [1, 2, 3]
v_B_recovered = [1, 2, 3]
```

## 3. TV-Q02 — +90° active rotation about body/inertial x at identity

```text
q_x90 = [0.7071067811865476,
         0.7071067811865476, 0, 0]
```

Expected:

\[
R_{NB}=\begin{bmatrix}
1&0&0\\
0&0&-1\\
0&1&0
\end{bmatrix},
\qquad
C_{BN}=R_{NB}^T.
\]

Basis behavior:

```text
R_NB * e_y^B = e_z^N
R_NB * e_z^B = -e_y^N
C_BN * e_y^N = -e_z^B
```

This last line is a critical sensor-prediction sign test.

## 4. TV-Q03 — +90° about z

```text
q_z90 = [0.7071067811865476,
         0, 0, 0.7071067811865476]
```

\[
R_{NB}=\begin{bmatrix}
0&-1&0\\
1&0&0\\
0&0&1
\end{bmatrix}.
\]

Expected:

```text
R_NB * e_x^B = e_y^N
C_BN * e_x^N = -e_y^B
```

## 5. TV-Q04 — Hamilton product/order

```text
q_x90 = [√2/2, √2/2, 0, 0]
q_y90 = [√2/2, 0, √2/2, 0]
```

Expected:

```text
q_x90 ⊗ q_y90 = [0.5, 0.5, 0.5, 0.5]
R(q_x90 ⊗ q_y90) = R(q_x90) R(q_y90)
```

Thus the right operand rotation is applied first to a vector.

## 6. TV-Q05 — Constant-rate propagation

Input:

```text
q0 = [1,0,0,0]
omega_B = [0,0,1] rad/s
dt = 0.1 s
bias_hat = [0,0,0]
```

Expected:

```text
q1 = Exp_q([0,0,0.1])
   = [0.9987502603949663,
      0, 0, 0.04997916927067833]
```

After `1000` steps with exact exponential and normalization, expected total orientation is equivalent to `Exp_q([0,0,100])`; metric uses sign invariance.

## 7. TV-Q06 — Gyro bias cancellation

True stationary body:

```text
omega_true = [0,0,0]
bias_true  = [0.01,-0.02,0.005] rad/s
noise      = 0
omega_m    = bias_true
```

Case A:

```text
bias_hat = bias_true
```

Expected corrected rate and propagation:

```text
omega_hat = [0,0,0]
q remains unchanged
```

Case B:

```text
bias_hat = [0,0,0]
```

Expected initial nominal rotation increment is `Exp_q(bias_true*dt)`, demonstrating why bias is a state.

## 8. TV-M01 — Body-vector prediction and Jacobian sign

Input:

```text
q_hat = identity
reference r_N = [1,0,0]
h = C_BN(q_hat) r_N = [1,0,0]
```

Expected attitude Jacobian:

\[
H_\theta=[h]_\times=
\begin{bmatrix}
0&0&0\\
0&0&-1\\
0&1&0
\end{bmatrix}.
\]

For right perturbation `δθ=[0,0,ε]`:

```text
q_true = q_hat ⊗ Exp_q([0,0,ε])
y_true = C_BN(q_true) r_N ≈ [1,-ε,0]
residual y_true-h ≈ [0,-ε,0]
H_theta*δθ = [0,-ε,0]
```

A result `[0,+ε,0]` indicates a convention/sign error.

## 9. TV-S01 — Sun tangent basis/Jacobian

For `h=e_x`, deterministic tangent basis:

```text
U = [e_y, e_z]  # 3x2 columns
```

Expected:

\[
H_{s,\theta}=U^T[h]_\times=
\begin{bmatrix}
0&0&-1\\
0&1&0
\end{bmatrix}.
\]

- rotation about `x` is instantaneously unobservable;
- small `+z` right error gives first residual component `-ε`.

## 10. TV-ST01 — Star-tracker log residual

Input:

```text
q_hat = identity
delta_theta = [0.01,-0.02,0.03] rad
q_z = Exp_q(delta_theta)
```

Expected quaternion:

```text
q_z = [ 0.9998250051041071,
        0.0049997083384377,
       -0.0099994166768754,
        0.0149991250153131 ]
```

Expected residual:

```text
Log_q(q_hat^-1 ⊗ q_z) = [0.01,-0.02,0.03]
```

Within numerical tolerance.

## 11. TV-INJ01 — Right injection

Input:

```text
q_hat_minus = q_z90
injected_delta = [0.1,0,0] rad
```

Expected:

```text
q_hat_plus = q_z90 ⊗ Exp_q([0.1,0,0])
           = [0.7062230818371108,
              0.03534060950936696,
              0.03534060950936696,
              0.7062230818371108]
```

If truth is set equal to this `q_hat_plus`, the post-injection relative log error is zero.

## 12. TV-RST01 — Reset Jacobian

Injected correction:

```text
a = [0.1,-0.2,0.05] rad
```

First-order expected:

\[
I-\tfrac12[a]_\times=
\begin{bmatrix}
1&0.025&0.1\\
-0.025&1&0.05\\
-0.1&-0.05&1
\end{bmatrix}.
\]

Exact right Jacobian expected:

\[
J_r(a)\approx
\begin{bmatrix}
0.99293524&0.02156622&0.10039441\\
-0.02821541&0.99792213&0.04811934\\
-0.09873212&-0.05144393&0.99168851
\end{bmatrix}.
\]

Finite-difference definition to verify:

\[
f(x)=\operatorname{Log}_q(\operatorname{Exp}_q(-a)\otimes\operatorname{Exp}_q(x)),
\]

and evaluate `∂f/∂x` at `x=a`; expected `J_r(a)`.

## 13. TV-SIGN01 — Antipodal invariance

For any unit `q`:

```text
distance(q, -q) = 0
ST_residual(q_hat=q, q_z=-q) = [0,0,0] after hemisphere alignment
update(q_z=q) == update(q_z=-q)
```

The internal quaternion values may differ by sign, but rotation, residual, correction and metrics must match.

## 14. TV-KF01 — Simple ST gain dimension/value

Input:

```text
P_minus = diag([0.04,0.04,0.04, 0.01,0.01,0.01])
H_ST = [I3, 0]
R_ST = 0.01 I3
```

Expected:

```text
S = 0.05 I3
K = [0.8 I3;
     0.0 I3]     # 6x3
```

For zero residual and zero injected correction, Joseph posterior before/after reset:

```text
P_plus = diag([0.008,0.008,0.008, 0.01,0.01,0.01])
```

## 15. TV-COV01 — Symmetry/SPD

For every covariance operation:

```text
relative_asymmetry = ||P-P^T||_F / max(||P||_F, eps)
```

Expected:

- relative asymmetry within configured roundoff tolerance;
- Cholesky succeeds for strictly SPD test inputs;
- any eigenvalue less than the negative roundoff tolerance is a failure;
- no silent eigenvalue clipping.

## 16. TV-LONG01 — Long-horizon norm

No-noise, arbitrary bounded gyro sequence, exact exponential propagation, normalize after each step:

```text
max_k | ||q_k||_2 - 1 | <= 1e-12  # float64 reference
```

The attitude result is additionally compared against a high-accuracy batch composition or analytic case; norm preservation alone does not prove correct multiplication order.

## 17. Adapter test for Basilisk/external libraries

The adapter must pass all three basis vectors:

1. obtain simulator orientation message;
2. convert to internal `q_NB`;
3. verify `R_NB e_i^B` against simulator body axes in inertial coordinates;
4. verify `C_BN r^N` against simulator sensor truth.

**[결정]** no inference from variable name such as `sigma_BN` is accepted without this test.

## 18. Pass report schema

```yaml
test_id:
implementation_commit:
platform:
float_type:
input:
expected:
actual:
abs_error:
rel_error:
pass:
```
