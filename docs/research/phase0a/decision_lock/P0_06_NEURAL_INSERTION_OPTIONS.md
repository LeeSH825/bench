# P0_06 Neural Insertion Options

> 작성일: 2026-07-30
> 표지 규칙: **[확인]** 실험·실측, **[문헌]** 논문·공식 자료, **[분석]** 수식·구조 해석, **[가설]** 검증 대상, **[결정]** 설계 선택, **[보류]** 근거 부족·후속 범위

| 항목 | 내용 |
|---|---|
| 목적 | Direct Split-KalmanNet과 structured Q/R/reliability-adaptive MEKF를 동일 tangent-space shell에서 분리 정의하고 공정 비교 조건을 고정한다. |
| 입력 근거 | S0–S4, E3; `P0_05_MEKF_MATH_CONTRACT.md`, `P0_07_CONTEXT_CONTRACT.md` |
| 결정 상태 | LOCK — 두 경로의 연구상 지위와 interface; TBD-BY-EXPERIMENT — 최종 proposed 선택 |
| 남은 TBD | Split baseline의 exact repository adaptation, feature normalization, context ANN/SNN architecture와 parameter budget |
| 다음 Gate | oracle intervention에서 direct-gain과 structured 경로를 동일 데이터·state·metric으로 비교하고 최종 지위를 결정 |


## 1. 공통 shell

두 경로는 다음을 공유한다.

- nominal state `x̄=(q_NB,b_g)`
- local error `δx=[δθ,δb_g]∈R^6`
- 동일 gyro propagation, sensor residual, `H_j`, quaternion injection/reset
- 동일 truth trajectory, pre-generated sensor packet, initial estimate
- 동일 valid/invalid packet semantics, timestamp/latency treatment
- 동일 loss/metric, trajectory-level split, adaptation budget disclosure

```text
predict(q,b,P_or_latent,gyro,dt)
form_residual(sensor_packet, q_minus, b_minus)
obtain_gain_or_covariance(sensor_id, features/context)
local_update(delta_x)
inject_and_reset(q,b,local_objects)
log_internal_state()
```

**[결정]** proposed만 MEKF이고 baseline은 additive quaternion EKF인 비교를 금지한다. Legacy MRP/EKF 결과는 참고군으로만 별도 표지한다.

## 2. 경로 A — Direct Split-KalmanNet

### 2.1 Local-gain contract

At a sensor event with residual dimension `m_j`:

\[
K_t=G_{1,t}H_t^TG_{2,t},
\]

with the conceptual dimensions

\[
G_{1,t}\in\mathbb R^{6\times6},\quad
H_t^T\in\mathbb R^{6\times m_j},\quad
G_{2,t}\in\mathbb R^{m_j\times m_j},\quad
K_t\in\mathbb R^{6\times m_j}.
\]

Update:

\[
\widehat{\delta x}_t=K_t\nu_t,
\]

followed by the same multiplicative injection and bias correction as the classical MEKF.

### 2.2 Allowed interpretation

- **[문헌]** Split-KalmanNet separates recurrent computation related to prior-state uncertainty and innovation uncertainty and combines them with the model Jacobian.
- **[결정]** `G1`: `prior-side latent factor`
- **[결정]** `G2`: `innovation-side latent factor`
- **[결정]** final `K`: local tangent-space direct gain

### 2.3 Prohibited interpretation without extra constraints

- `G1=P^-` or `Q`
- `G2=R^{-1}` or `S^{-1}`
- either factor is an SPD covariance
- branch output uniquely identifies process vs measurement noise

Scale ambiguity:

\[
(cG_1)H^T(c^{-1}G_2)=G_1H^TG_2,
\]

and

\[
S=HP^-H^T+R
\]

mixes prior and measurement uncertainty on the innovation side. Therefore branch labels are architectural names, not physical proof.

### 2.4 MEKF tangent/reset contract

The direct-gain network must consume and output quantities in the current local error convention.

- input innovations use the sensor-specific residual defined in the Math Contract;
- previous attitude correction is a 3D local rotation vector, not quaternion component difference;
- output attitude correction is `δθ` in the current right/body tangent;
- injection/reset occurs after every accepted update;
- hidden state is reset at sequence boundaries, not after every MEKF update.

**Reset caveat:** an unconstrained RNN hidden state has no known geometric transport. Therefore:

1. hidden state is called latent, not error/covariance state;
2. explicitly geometric vector/matrix slots, if introduced, must be transported with `J_r`/`G_reset`;
3. default features should favor reset-invariant or clearly local quantities (norms, normalized innovation, current local correction with explicit timestamp);
4. branch activation before/after reset is logged for diagnosis.

### 2.5 Covariance/consistency limitation

- Direct gain does not by itself provide a valid `P` or `S`.
- NIS/NEES are not primary consistency claims unless a separate explicit covariance recursion remains valid and is not contradicted by the learned gain.
- A `covariance-like` neural factor must not be labeled covariance solely from shape.

### 2.6 Baseline reproduction policy

- reproduce the source architecture/feature definition as faithfully as possible inside the common MEKF adapter;
- record every required deviation caused by quaternion/tangent state;
- do not add special robust gates only to one model;
- include parameter count, recurrent state size, update rate and training supervision in comparison metadata;
- branch output, norm, singular value, correlation and collapse diagnostics are logged.

## 3. 경로 B — Structured adaptive MEKF

### 3.1 Data flow

```text
onboard feature or physical oracle context
  → positive process/measurement scales + reliability
  → explicit Q_c(t), R_j(t), gate_j(t)
  → classical MEKF covariance recursion
  → K_t
  → multiplicative local update/reset
```

### 3.2 Covariance mapping

For the event-local context in `P0_07_CONTEXT_CONTRACT.md`:

\[
Q_c(t)=\operatorname{blkdiag}
\left(e^{c_g(t)}S_{g,0},e^{c_b(t)}S_{b,0}\right),
\]

\[
R_j(t)=e^{c_{R,j}(t)}R_{j,0}.
\]

- baseline matrices `S_g0,S_b0,R_j0` are SPD/PSD with declared units;
- scalar exponentials preserve positive scaling;
- anisotropic matrices, if later needed, use a Cholesky-factor parameterization and are a separate ablation;
- context bounds are implemented in log space and disclosed, not hidden clipping.

### 3.3 Reliability/gate mapping

Hard validity:

```text
if sensor_packet.validity == false:
    skip update
```

Hidden outlier reliability `ρ_j∈[0,1]` can drive a robust weight or reject threshold. One permitted soft form is

\[
R_{j,eff}=R_j/\max(\rho_j,\rho_{min}),
\]

but it must be labeled a reliability approximation, not a second independent physical noise scale. A hard false-solution decision skips or robustly down-weights the update and is logged separately from inlier `R_j` scaling.

### 3.4 Classical recursion

\[
P^-=\Phi P^+\Phi^T+Q_d(c_g,c_b),
\]

\[
S_j=H_jP^-H_j^T+R_j(c_{R,j}),
\quad K_j=P^-H_j^TS_j^{-1}.
\]

The common Joseph update and `G_reset` transport are mandatory.

### 3.5 Advantages and risks

| 항목 | 장점 | 위험/한계 |
|---|---|---|
| physical interpretation | context target maps to defined `Q/R/gate` | wrong context can be confidently wrong |
| SPD/consistency | explicit recursion enables NIS/NEES | model assumptions still approximate |
| modularity | sensor-specific scaling/gate | context dimension can grow |
| failure analysis | wrong-side adaptation diagnosable | pure model error may not be covariance-scale problem |
| fallback | works with oracle/classical/ANN/SNN context | may underperform unconstrained direct gain if model mismatch is complex |

## 4. Path A/B interface comparison

| 항목 | Path A: Direct Split | Path B: Structured adaptive |
|---|---|---|
| neural output | latent factors / local gain | context or `Q/R/gate` parameters |
| gain source | direct learned factorization | classical covariance recursion |
| covariance | not guaranteed | explicit `P,Q,R,S` |
| SPD guarantee | not inherent | positive parameterization + Joseph/reset |
| NIS/NEES | restricted | permitted under assumptions |
| branch meaning | latent prior/innovation side | physical process/sensor target |
| main strength | flexible model-mismatch compensation | interpretation, consistency, controlled interventions |
| main risk | non-identifiability, hidden-state/reset ambiguity | model-form restriction, context estimation error |
| Phase 0A status | backbone/baseline | proposed/fallback candidate |

## 5. Common model/event API

Pseudocode only:

```python
if path == "structured":
    c_process = oracle_or_estimator(process_features, sensor_id=None)
    Qc = map_process_context(c_process)
else:
    Qc = None

prior = common_predict(previous, gyro_packet, dt, Qc=Qc)
residual, H, R0, meta = common_measurement_model(prior, sensor_packet)

if path == "direct_split":
    G1, G2, latent = split_model(features, H, sensor_id)
    K = G1 @ H.T @ G2
    delta = K @ residual
    covariance_for_claims = None

elif path == "structured":
    c_measurement = oracle_or_estimator(measurement_features, sensor_id)
    R, gate = map_measurement_context(c_measurement, R0, sensor_id)
    if gate.reject:
        return prior
    K, S = classical_gain(prior.P, H, R)
    delta = K @ residual

posterior = common_inject_reset(prior, delta)
```

Process context is evaluated before covariance propagation; measurement context is evaluated at the sensor event. The implementation must not propagate `P` twice.

## 6. Inputs and supervision budget disclosure

Every model row in a result table reports:

- raw sensor channels used
- reference-model inputs
- quality/telemetry inputs
- history length/recurrent state
- true state labels used in training
- oracle labels used at inference
- online gradient update or adaptation
- number of trainable parameters
- update frequency and floating-point precision

**[결정]** an oracle context model is not compared as if deployable. It is an upper-bound mechanism test.

## 7. Required diagnostics

### Direct Split

- `||G1||`, `||G2||`, singular values, condition numbers
- scale drift under `G1×c,G2/c` ambiguity
- branch output correlation and branch swap sensitivity
- one-branch ablation and hidden-state reset test
- correction norm, NaN/divergence, response by intervention

### Structured

- estimated/oracle log scales and bounds
- `λ_min(Q_d),λ_min(R),λ_min(P),λ_min(S)`
- sensor-specific pre/post-gate NIS
- wrong-side adaptation test
- context delay/noise sensitivity

## 8. Fair-comparison matrix

| condition | must be identical? |
|---|---:|
| truth trajectory/orbit/maneuver | yes |
| sensor raw packets/noise realization | yes |
| nominal state and initial error | yes |
| quaternion/frame/error convention | yes |
| sensor measurement function/Jacobian | yes |
| validity/latency handling | yes, unless method under test |
| train/validation/test split | yes |
| metric/evaluation windows | yes |
| oracle information | disclosed; deployable models must not receive it |
| parameter/operation budget | matched where claimed, otherwise reported |

## 9. Final proposed-selection rule

The final main method is not locked by preference. Use this order:

1. verify fixed/tuned MEKF;
2. verify problem exists under time-varying uncertainty;
3. compare oracle structured vs fixed/tuned/robust;
4. reproduce direct Split baseline in the same shell;
5. compare accuracy, peak/recovery, divergence, consistency, interpretation and complexity;
6. choose the simplest method that gives repeated practical improvement.

- If structured is comparable in accuracy and better in consistency/failure behavior, prefer it as final proposed.
- If direct gain has a robust, repeatable advantage not reproducible by structured scaling, retain it but limit physical branch claims.
- If classical adaptive/robust baselines close the gap, narrow the neural contribution.
- If oracle structured context gives no improvement, do not train ANN/SNN context.

## 10. SNN insertion policy

**[결정: PROVISIONAL]** first SNN candidate is one of:

- fast change detector
- outlier/inlier reliability estimator
- event-triggered context updater
- fast `α_g` or `ρ_j` estimator

It is not locked as a full gain generator. Layer count, neuron model, surrogate gradient and hardware mapping are outside Phase 0A. Claims are limited to measured algorithmic quantities until hardware evidence exists.

## 11. Gate

- [ ] both paths accept identical local residual/H and produce a 6D correction.
- [ ] direct branch outputs are never labeled physical covariance in logs/paper.
- [ ] structured path maintains explicit SPD matrices and valid reset transport.
- [ ] same sensor stream and initial condition can be replayed through both paths.
- [ ] oracle/deployable information budgets are visibly different.
- [ ] final proposed status remains TBD until oracle/problem tests.
