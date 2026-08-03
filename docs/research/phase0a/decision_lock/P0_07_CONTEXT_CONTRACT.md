# P0_07 SoW/Context 계약

> 작성일: 2026-07-30  
> 표지 규칙: **[확인]** 실험·실측, **[문헌]** 논문·공식 자료, **[분석]** 수식·구조 해석, **[가설]** 검증 대상, **[결정]** 설계 선택, **[보류]** 근거 부족·후속 범위

| 항목 | 내용 |
|---|---|
| 목적 | physical context, latent context, oracle label, onboard feature, process/measurement side, slow/fast time scale와 최소 차원을 정의한다. |
| 입력 근거 | S0–S4, E3; `P0_02_TRUTH_SENSOR_ESTIMATOR_BOUNDARY.md`, `P0_06_NEURAL_INSERTION_OPTIONS.md` |
| 결정 상태 | LOCK — 의미와 최소 oracle interface; TBD-BY-EXPERIMENT — scalar/vector, shared/specific, single/dual-timescale 최종 선택 |
| 남은 TBD | oracle usefulness, identifiability, actual telemetry/quality flags, ANN/SNN architecture |
| 다음 Gate | Package C에서 oracle context가 유용하고 process/measurement separation 필요 범위가 확인됨 |


## 1. 용어 계약

| 용어 | 정의 | 사용 조건 |
|---|---|---|
| physical context | simulator/physics에서 명시적으로 정의되고 filter target에 대응하는 값 | label generation과 units/mapping이 문서화됨 |
| oracle context | truth generator가 알고 있는 physical context를 inference에 직접 제공 | upper-bound mechanism test only |
| estimated context | onboard-available feature에서 추정한 physical context | deployable candidate |
| latent context | end-to-end loss로 학습되며 physical dimension과 1:1 의미가 보장되지 않는 vector | “SoW” 또는 physical `Q/R`라고 부르지 않음 |
| context target | network가 예측해야 할 output/label | true state와 구분 |
| context-estimator input | 실제 onboard에서 가능한 sensor/residual/telemetry feature | oracle label과 파일/API 분리 |
| reliability | inlier/valid update에 대한 신뢰도 | inlier variance scale과 gross outlier를 구분 |

## 2. 최소 event-local oracle context

At time `t`, for active measurement sensor `j∈{mag,sun,ST}`:

\[
c_t^{(j)}=
\begin{bmatrix}
c_g(t)\\c_b(t)\\c_{R,j}(t)\\\rho_j(t)\end{bmatrix}
=
\begin{bmatrix}
\log\alpha_g(t)\\
\log\alpha_b(t)\\
\log\alpha_{R,j}(t)\\
\rho_j(t)
\end{bmatrix}
\in\mathbb R^4.
\]

At gyro-only ticks, only the first two persistent process values are active; measurement entries are null/masked, not zero-valued physical measurements. Sensor identity and validity metadata are separate categorical/boolean inputs.

### 2.1 Mapping

\[
S_g(t)=\alpha_g(t)S_{g,0},\qquad
S_b(t)=\alpha_b(t)S_{b,0},
\]

\[
R_j(t)=\alpha_{R,j}(t)R_{j,0}.
\]

`ρ_j` means inlier reliability after explicit hardware validity:

- packet invalid/outage/eclipsed: deterministic skip from packet metadata;
- hidden false solution/outlier: oracle `ρ_j≈0`;
- valid inlier with increased noise: `ρ_j=1`, `α_R,j>1`.

This prevents the same degradation from being encoded in both `R` scale and gate.

## 3. Why these four and not all initial candidates

| original candidate | Phase-0A decision | reason |
|---|---|---|
| `log α_bias-RW` | keep as `c_b` | distinct bias-state process block |
| `log α_thermal/model-uncertainty` | do not keep separately initially | temperature mean is calibration; residual often maps to `c_b/c_g`; avoid duplicate target |
| `log α_gyro-noise-floor` | keep as `c_g` | distinct attitude process noise block |
| separate `log α_gyro-burst` | merge into time-varying `c_g` initially | same `S_g` target; fast/slow dynamics can differ without duplicate output |
| `log α_mag` | active-sensor `c_R,j` | sensor-specific `R` target |
| `log α_sun` | active-sensor `c_R,j` | same interface, distinct sensor ID |
| `log α_ST` | active-sensor `c_R,j` | same interface, distinct sensor ID |
| `p_outlier/reliability` | keep as `ρ_j` | gross error cannot be represented safely by Gaussian scale alone |

**[결정]** output dimension is minimized by reusing `c_R,j,ρ_j` with sensor identity, not by forcing all sensors to share the same physical value.

## 4. Dense representation for batch/ablation

For `N_s=3` external sensors:

\[
c_{dense}=[c_g,c_b,
 c_{R,mag},c_{R,sun},c_{R,ST},
 \rho_{mag},\rho_{sun},\rho_{ST}]^T\in\mathbb R^8.
\]

Use only when the architecture requires persistent simultaneous outputs. A mask indicates unavailable sensors. The event-local 4-value form remains the primary interface because it scales to sensor additions and avoids inventing values at absent update events.

## 5. Scalar baseline

For Adaptive-KalmanNet comparability, include a scalar oracle baseline such as

\[
c_{scalar}^{(j)}=\log\frac{\alpha_Q}{\alpha_{R,j}},
\]

where a declared aggregate `α_Q` is used. This baseline has known limitations:

- cannot separately scale gyro noise and bias RW;
- loses absolute scale when Q and R change together;
- cannot represent multiple sensors with different reliability at the same time;
- cannot distinguish inlier noise from outlier validity.

Its purpose is controlled comparison, not default truth representation.

## 6. Context dimension contract

| dimension | physical meaning | side | expected time scale | filter target | oracle label generation | onboard estimation possibility |
|---|---|---|---|---|---|---|
| `c_g` | gyro-driven attitude process PSD multiplier | process | fast burst + possible slow floor | `S_g/Q_θ` block | simulator sets `S_g=exp(c_g)S_g0` | gyro increments, high-pass energy, correction history, vibration proxy, command/wheel telemetry |
| `c_b` | gyro-bias driving PSD multiplier | process | slow | `S_b/Q_b` block | simulator sets `S_b=exp(c_b)S_b0` | temperature, bias-estimate evolution, long-window residual statistics |
| `c_R,j` | active sensor inlier covariance multiplier | measurement | event/slow depending sensor | `R_j` | sensor generator sets `R_j=exp(c_R,j)R_j0` | innovation history, quality, norm/geometry mismatch, age |
| `ρ_j` | active sensor inlier/valid reliability | measurement/gate | fast event | robust weight/skip | true inlier/outlier state after explicit validity | quality/validity, residual shape, norm, cross-sensor disagreement |

### Axis anisotropy

The minimum context is scalar per block. If Tier-2 data shows independent axis changes, expand to diagonal/full SPD scales only after a scalar residual analysis. A scalar that cannot represent anisotropy must not be interpreted as true full `Q/R`.

## 7. Mean correction vs uncertainty scale

### Mean/calibration path

```text
temperature measurement
  → calibrated mean-bias prediction b_T_hat(T)
  → subtract from raw gyro
  → residual gyro bias estimated as b_g state
```

### Uncertainty path

```text
residual variation / event regime
  → c_g or c_b
  → S_g or S_b scale
```

**[결정]** physical context does not output the gyro bias mean in the minimum design. True bias is an evaluation/calibration label, not an uncertainty context input.

## 8. Process-side vs measurement-side inputs

### 8.1 Process-side onboard features

- gyro sample/increment and finite differences
- estimated bias and its innovation-driven correction history
- measured temperature and temperature derivative
- commanded torque/slew flag when truly available
- reaction-wheel speed or magnetorquer state when truly available
- optional accelerometer vibration energy
- previous attitude correction norm

### 8.2 Measurement-side onboard features

- sensor-specific innovation vector and history
- normalized innovation energy using the **current nominal** `S`
- mag norm/model mismatch
- CSS illuminated-head count, WLS condition/residual, eclipse/FOV validity
- ST quality, age, tracking status
- cross-sensor disagreement at same/near time

### 8.3 Forbidden features

- true attitude/rate/bias
- injected `α` values or event label
- future measurement/innovation/validity
- truth magnetic/sun vector without onboard-model mismatch
- test-set normalization statistics

## 9. Identifiability limits

**[analysis]** a large innovation can arise from:

- propagation uncertainty or bias error;
- measurement noise/interference;
- reference-model error;
- initial attitude error;
- delayed packet;
- genuine maneuver.

Therefore instantaneous innovation magnitude alone generally cannot uniquely identify `Q` vs `R`. The contract requires:

1. temporal information;
2. sensor identity and multiple sensors where available;
3. gyro/temperature/telemetry or geometry features;
4. controlled A/B interventions;
5. explicit acknowledgment of unidentifiable cases.

An estimated context is called physical only if its dimensions correlate with intervention labels and changing one label produces the intended filter-side response. Otherwise it is latent context.

## 10. Scalar/vector, shared/specific, single/dual sequence

The validation order is locked:

1. fixed no-context MEKF;
2. oracle scalar context;
3. oracle event-local 4-value context;
4. if needed, dense 8D sensor-specific context;
5. shared temporal estimator;
6. process/measurement-specific heads;
7. single-timescale ANN;
8. dual-timescale ANN;
9. only the functionally justified fast channel as SNN.

**Stop rules**

- scalar≈vector: keep scalar/smaller context;
- shared≈specific: keep shared;
- single≈dual: keep single;
- oracle gives no benefit: stop learned context;
- ANN does not beat classical detector/adaptation: do not proceed to SNN;
- SNN lacks accuracy/latency/sparsity advantage: remove SNN claim.

## 11. Slow vs fast contract

The output target and estimator memory are separated concepts.

- `c_b`: explicitly slow target; penalize implausible rapid variation in learned stages.
- `c_g`: can change fast for vibration/noise burst; slow floor drift is allowed but no second output until evidence requires it.
- `c_R,j`: sensor-event target; may be slow degradation or fast inlier-noise change.
- `ρ_j`: fast gate/outlier target.

A dual-timescale model may maintain slow and fast hidden states while emitting the same four physical targets. Hidden states remain latent unless separately supervised.

## 12. Oracle label generation

Each simulation intervention writes target values from configuration, not from noisy post-hoc sample estimates.

```yaml
process_context:
  log_alpha_g: log(Sg_scale_assigned_by_generator)
  log_alpha_b: log(Sb_scale_assigned_by_generator)
measurement_event_context:
  sensor_id: mag|sun|star_tracker
  log_alpha_R: log(R_scale_assigned_to_inlier_generator)
  rho: true_inlier_reliability_after_explicit_validity
```

- scalar labels require base matrices and units in the same config.
- transition ramps/steps are logged at measurement time.
- interpolation between context update times is specified.
- outlier samples do not also change inlier `R` label unless the scenario explicitly combines both.

## 13. Estimated-context output constraints

- log scales are bounded to a disclosed training/operating interval;
- `ρ=sigmoid(z)` or equivalent lies in `[0,1]`;
- no future smoothing in online inference;
- missing sensor event uses mask, not imputation/pseudo-measurement;
- context timestamp is logged separately from sensor timestamp;
- context delay/noise/quantization sensitivity is mandatory.

## 14. Loss policy for later stages

Not an architecture decision, but supervision modes are defined:

1. direct physical-context loss;
2. attitude/bias state loss through the filter;
3. multi-task physical context + state loss;
4. latent end-to-end context.

Physical and latent models must be reported separately. A latent model that improves state loss does not thereby estimate `Q/R` correctly.

## 15. Context metrics

- log-scale MAE/RMSE by dimension
- inlier/outlier AUROC/precision-recall where class balance is reported
- event detection delay and false adaptation rate
- transition overshoot/settling
- context calibration/reliability curve
- downstream attitude peak/RMSE/recovery and consistency
- oracle-to-estimated performance gap

## 16. Versioned context schema

```yaml
context_contract_version: P0A-v1
event_local:
  process: [log_alpha_g, log_alpha_b]
  measurement: [sensor_id, log_alpha_R, rho]
null_policy: masked
units: dimensionless_log_scale_and_probability
base_profile_ids: [Sg0, Sb0, Rmag0, Rsun0, Rst0]
```

Any dimension change creates a new version and regenerates labels/normalization.

## 17. Gate

- [ ] each dimension maps to exactly one defined filter target.
- [ ] oracle labels exist independently of noisy measurements.
- [ ] true state/event labels are inaccessible to deployable runner.
- [ ] A/B interventions test process-vs-measurement identifiability.
- [ ] scalar/vector and shared/specific are ablations, not assumptions.
- [ ] ANN usefulness is proven before SNN design.
