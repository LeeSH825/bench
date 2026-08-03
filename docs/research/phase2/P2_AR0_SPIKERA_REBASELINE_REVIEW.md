# P2-AR0 SpikeRA Architecture Re-baseline Review

- review ID: `P2-AR0-SPIKERA-REBASELINE-REVIEW-v1`
- primary track: `IAA SpaceAI 2026 SpikeRA-KalmanNet`
- scope: documentation-only claim, architecture, sensor, baseline, and phase-plan review
- reviewed date: 2026-08-03 (Asia/Seoul)

> **Formal review status:** `BLOCKED_SPIKERA_EVIDENCE_OR_ARCHITECTURE_INSUFFICIENT`  
> **Reason:** the live repository `/home/dss-pc-05/bench` was not mounted in this execution environment, so the required entry/exit repository fingerprints and direct writes to the exact repository paths could not be performed. The scientific/architectural review is complete and is packaged with the exact relative paths for later integration.  
> **Architecture decision:** `NOT FROZEN`  
> **P2-A0 / model implementation:** `NOT AUTHORIZED`

## 1. Authority and scope

[PROMPT] is the controlling instruction artifact. This review does not implement P2-A0, a classical adaptive filter, KalmanNet/Split-KalmanNet, an SNN, training, a pilot, a dataset, a manifest, a test, or a YAML configuration. It does not edit Phase 0–1, P2-DR0, or P2-A frozen artifacts.

The only research center is the IAA SpikeRA-KalmanNet track. The low-cost gyro compensation + KalmanNet study is a separate Track 2. It is not merged here; only the need for its own future Phase 0 is retained.

## 2. Current official project state retained

```text
P1 Exit                         CONDITIONAL_GO
Stationary covariance closure  PASS
Independent C4 closure         FAIL
P2-DR0 Design Freeze           PASS
P2-A Implementation Review     PASS
P2-A0/A1/A2/A3                 NOT AUTHORIZED
P2-B and neural implementation NOT AUTHORIZED
```

The earlier `READY_FOR_EXPLICIT_P2_A0_FOUNDATION_IMPLEMENTATION` recommendation is not annulled as an infrastructure review result. It is placed on hold because SpikeRA changes the research-facing target/sensor/model contracts that P2-A0 would otherwise encode.

## 3. Abstract-derived research problem

The abstract states a single-path concept:

```text
innovation-derived features
  -> compact spiking adapter
  -> time-varying measurement reliability signal
  -> measurement gate
  -> suppress measurement update
```

It attributes transient degradation to maneuvers, actuator-induced disturbances, and vibration-related events; names transient gyro bias jumps and event-window noise amplification as benchmark interventions; and claims improved event-window robustness, stable nominal performance, and sparse spike activity. It does not provide model equations, sensor identity, backbone variant, dataset count, split, baseline list, numerical result, parameter count, operation count, or hardware measurement [ABS].

### Core problem exposed by re-baselining

A large innovation can arise from either side of the residual:

\[
\nu_t = z_t - h(\hat x_t^-).
\]

A degraded external sensor increases the measurement-error part of \(z_t\). A gyro/process degradation increases propagation error in \(\hat x_t^-\), which is then projected into the same innovation. A scalar innovation magnitude can therefore be similar under different causes. Phase 1 C5 provides pair-specific evidence that matched scalar innovation RMS does not identify process versus measurement cause; it is not a universal impossibility proof [P1, P1-NUM].

The correct actions can be opposite:

- **prediction/process degradation:** increase prior-side responsiveness or process uncertainty and allow a trustworthy external attitude sensor to correct the propagated state;
- **measurement degradation:** reduce the influence of the affected external measurement.

A single measurement gate can therefore suppress the correction that is most needed when the gyro/process side is degraded.

## 4. Terminology decision

Recommended paper terms:

```text
P-SoW = prediction-side SoW
        (expanded prose: prediction/process-side reliability context)
M-SoW = measurement-side SoW
        (expanded prose: measurement/update-side reliability context)
```

`gyro(state) SoW` is rejected. Gyro output is a propagation input. The nominal bias \(b_g\) is part of the MEKF state. `innovation-side reliability context` may be used for a feature stream, but not as the final M-SoW name because innovation includes both prediction and measurement error.

## 5. Recommended common MEKF shell

Retain the Phase 0 convention:

```text
nominal state: (q_NB, b_g)
local error: delta_x = [delta_theta, delta_b_g] in R^6
scalar-first Hamilton quaternion
active body-to-navigation q_NB
right-multiplicative injection/reset
```

Gyro propagation:

\[
\hat\omega_t = \omega_{m,t}^{B}-\hat b_{g,t-1}^{+},
\qquad
\hat q_{NB,t}^{-}=\hat q_{NB,t-1}^{+}\otimes
\operatorname{Exp}_q(\hat\omega_t\Delta t),
\qquad
\hat b_{g,t}^{-}=\hat b_{g,t-1}^{+}.
\]

For an external sensor \(j\), form a right-local residual \(\nu_{j,t}\). For ST:

\[
\nu_{ST,t}=\operatorname{Log}_q\!\left((\hat q_{NB,t}^{-})^{-1}\otimes q_{ST,t}\right)\in\mathbb R^3.
\]

The direct local gain produces:

\[
K_t\in\mathbb R^{6\times m},\qquad
\delta x_t=K_t\nu_t=
\begin{bmatrix}\delta\theta_t\\\delta b_{g,t}\end{bmatrix}.
\]

Injection and reset:

\[
\hat q_{NB,t}^{+}=\hat q_{NB,t}^{-}\otimes\operatorname{Exp}_q(\delta\theta_t),
\qquad
\hat b_{g,t}^{+}=\hat b_{g,t}^{-}+\delta b_{g,t},
\]

followed by the frozen right-local reset. Typed event order, validity behavior, exact replay, and truth/oracle separation remain unchanged in principle.

## 6. Recommended architecture

### Target architecture: dual-SoW Split-MEKF-KalmanNet

```text
causal gyro/process features
  -> P-SoW SNN population/memory
  -> continuous decoded c_P(t)
  -> prior-side Split latent modulation G1

causal sensor-specific innovation features
  -> M-SoW SNN population/memory
  -> continuous decoded c_M,j(t)
  -> measurement-side Split latent modulation G2

modulated G1/G2 + H_t
  -> local gain K_t
  -> delta_x_t = K_t nu_t
  -> right-local MEKF injection/reset
```

Interpretation restrictions:

```text
G1 = prior-side latent factor
G2 = innovation-side latent factor
G1 is not automatically P or Q
G2 is not automatically R^-1 or S^-1
direct-model NEES/covariance claims are unavailable by default
```

Dual SoW is not merely two scalar outputs. It requires source-separated causal inputs, separate temporal state or populations, action-aligned modulation sites, separately supervised targets, and swapped/wrong-side diagnostics. A shared innovation-only encoder that emits two numbers but controls one gate does not establish dual-cause separation.

### SNN responsibility

The SNN estimates SoW only. It does not compute \(K_t\), replace the MEKF shell, or acquire truth/event-window labels at inference.

Preferred form:

```text
one logical SpikeRA adapter
  - process membrane population/state
  - measurement membrane population/state
  - optional shared low-level encoder only after source-separation ablation
spikes and/or membrane state
  -> small causal decoder
  -> bounded log-domain continuous contexts
```

A binary spike gate is retained only as an ablation. Continuous contexts support graded transitions and branch modulation. They must remain bounded and causal.

## 7. Initial target and loss recommendation

IAA core targets:

```text
P-SoW target = c_b(t)    = log(alpha_b(t))
M-SoW target = c_R,ST(t) = log(alpha_R,ST(t))
```

Rationale:

- `c_b` maps directly to the Phase 1 slow bias-random-walk mechanism and keeps the first problem identifiable enough to test.
- `c_g` is deferred until a distinct gyro-white-noise scenario is validated.
- `c_R,ST` gives a clean full-attitude measurement-side target.
- gross outlier/false-solution reliability `rho` remains outside the core; it needs a new versioned dataset and robust-update contract.
- IMU+Mag generalization uses `c_R,mag` in a separate scenario family.

Training sidecars may expose the oracle targets only to authorized training/supervision. Inference receives causal onboard features, not alpha values, event identity, window, future packets, or truth.

Recommended loss family:

\[
\mathcal L = \lambda_x\mathcal L_{state}
+\lambda_P\mathcal L_{P\text{-SoW}}
+\lambda_M\mathcal L_{M\text{-SoW}}
+\lambda_s\mathcal L_{activity}
+\lambda_t\mathcal L_{transition}.
\]

- `state`: necessary for the end-task claim.
- P/M supervision: necessary for a cause-separation claim; without it, the contexts remain latent.
- activity regularization: necessary only for sparse-activity evidence, not energy/power claims.
- transition loss: useful for delay/overshoot control; it must be branch-aware so a slow P target does not over-smooth a fast M event.

## 8. Sensor configuration decision recommendation

```text
IAA primary:   Gyro + Star Tracker
Generalization: Gyro + Magnetometer, secondary and separately reported
```

Reasons:

1. ST supplies a full-attitude measurement, making M-SoW interpretation clearer.
2. It avoids the single-vector weak-direction confound demonstrated by the Phase 1 Mag-only stress case.
3. UNIT-ST artifacts and right-local ST residuals already exist, reducing schedule risk.
4. Mag provides useful later evidence that the method is not tied to a full-attitude sensor, but should not be allowed to obscure the core causal claim.

`IMU+Mag primary` is not recommended because reliability change and geometry-dependent observability would be entangled.

## 9. Backbone and fallback strategy

Recommended development order, not implementation authorization:

```text
1. MEKF-KalmanNet local-gain smoke
2. Split-MEKF-KalmanNet stable backbone
3. oracle dual-SoW branch modulation
4. EWMA/CUSUM and GRU SoW baselines
5. SpikeRA dual-SoW adapter
6. optional joint fine-tuning
```

Split remains the target because its prior/innovation branch structure is the closest match to the hypothesis. MEKF-KalmanNet remains the IAA fallback.

Fallback is triggered at a preregistered P2-D checkpoint if Split fails any of the following after the authorized stability budget is exhausted:

- deterministic sequence reset and right-local injection;
- zero divergence and finite gain/latent diagnostics;
- repeated-seed training stability;
- no improvement over monolithic MEKF-KalmanNet in the cause-specific swapped/wrong-side test beyond the frozen equivalence margin;
- inability to preserve a defensible latent-only interpretation of G1/G2.

The exact calendar deadline, seed count, and stability budget require user approval and a later P2-D numeric freeze.

## 10. Baseline and falsification plan

| ID | comparator | hypothesis it can falsify |
|---|---|---|
| B0 | Classical F-BASE MEKF | adaptation is unnecessary |
| B1 | MEKF-KNet or Split-MEKF without adapter | any reliability adaptation adds value |
| B2 | Backbone + oracle dual SoW | the proposed action mechanism has no attainable benefit |
| B3 | Backbone + EWMA/CUSUM dual-side SoW | learned recurrence is unnecessary |
| B4 | Backbone + small GRU dual-side SoW | spiking dynamics add no incremental value over matched recurrent ANN |
| P | Backbone + SpikeRA dual SoW | proposed model |
| A-single | single-gate SpikeRA | dual separation is unnecessary |
| A-swap | swapped/wrong-side SoW action | contexts are not causally/action aligned |
| A-MLP | memoryless small MLP | temporal state is unnecessary |
| A-backbone | monolithic vs Split backbone | branch separation is unnecessary |

SNN superiority is not assumed. A valid outcome is GRU/CUSUM equivalence or superiority, in which case the SpikeRA claim must narrow or stop.

## 11. Scenario matrix

| scenario | physical intervention | generator target | estimator-visible evidence | forbidden label | expected correct action | wrong-side action | primary metric |
|---|---|---|---|---|---|---|---|
| S0 stationary matched | all scales 1 | P=0, M=0 | nominal gyro and ST streams | stationary/scenario ID | no adaptation | false P/M activation | stationary attitude/bias penalty, false adaptation |
| S1 process only | time-varying `alpha_b` or separately versioned bias-jump intervention | `c_b`; M masked/0 by contract | raw gyro dynamics, estimated bias/correction history, causal ST residual history | alpha/event window | prior-side modulation; retain correction from trustworthy ST | suppress ST/update as if M degraded | slow-window bias RMSE, P/M confusion, delay |
| S2 ST measurement only | `alpha_R,ST(t)` inlier covariance amplification | `c_R,ST`; P normal | ST residual, whitened residual, NIS history, sensor ID/validity | alpha/event window | downweight ST-side update | inflate prior-side/process response | event attitude peak/RMSE, M delay |
| S3 overlap | S1 slow + S2 fast overlap | both targets | both causal feature families | both labels/windows | independent branch modulation | shared/global or swapped action | both co-primary endpoints, recovery |
| S4 matched-innovation pair | one process arm and one measurement arm selected to match a prespecified scalar innovation summary | arm-specific P or M | raw gyro + sensor-specific residual history | arm identity | classify/action by source | identical single gate | confusion rate, endpoint sign, matched-summary gap |
| S5 long horizon | stationary and one complete event/recovery schedule | same targets | complete causal stream | event timing | return contexts to nominal without drift | persistent adaptation/saturation | recovery, false adaptation, divergence, membrane saturation |

Abstract mapping:

- `transient gyro bias jumps` is **not identical** to the current Phase 1 `alpha_b` random-walk-intensity change. Either add a separately versioned bias-jump scenario later or reword the abstract to “transient gyro-bias/process uncertainty change.”
- `event-window noise amplification` is ambiguous. In the IAA core it should mean ST inlier covariance amplification for M-SoW. A separate gyro white-noise amplification family would belong to P-SoW and must not share the same label.

## 12. Metrics and claim boundary

Required additions to existing state metrics:

- P-SoW and M-SoW MAE/RMSE;
- process/measurement confusion and swapped-side action error;
- transition detection delay and false adaptation rate;
- context overshoot, settling, and recovery;
- stationary penalty and event-window attitude/bias improvement;
- spike rate, active-neuron ratio, membrane saturation;
- parameter count and operation-event count.

Without hardware measurement, the following claims remain prohibited:

```text
lower energy
lower power
neuromorphic efficiency
flight real-time suitability
```

Sparse activity is algorithmic activity evidence only.

## 13. Prior-art and novelty conclusion

Established or close prior art includes direct learned Kalman gains [L1], split prior/innovation covariance-style recurrent branches [L2], SoW/context modulation [L3], neural change-point reliability indicators [L4], SNN-based Kalman gain computation [L5, L6], SNN-based covariance adaptation for low-cost IMU fusion [L7], neural process-noise adaptation in a Lie-group filter [L8], and spacecraft neural/adaptive filtering [L9–L11]. [L12-WITHDRAWN] is not used as stable authority.

Therefore the following claims are prohibited without a substantially broader search and stronger evidence:

```text
first SNN Kalman filter
first reliability-aware KalmanNet
first adaptive neural Kalman filter
first low-cost-IMU SNN Kalman method
first neural spacecraft attitude filter
```

A defensible, narrow contribution statement is:

> A right-local MEKF-compatible neural-gain architecture in which a compact spiking adapter decodes two causal, source-separated continuous contexts—a prediction-side gyro-bias process scale and a measurement-side star-tracker inlier covariance scale—and modulates corresponding latent prior/innovation branches, evaluated against fixed, oracle, CUSUM/EWMA, GRU, single-gate, swapped-side, and monolithic-backbone baselines under paired transient spacecraft-attitude scenarios.

This remains a proposed contribution until implementation and evidence exist.

## 14. Existing plan impact summary

- Phase 0 MEKF math, Phase 1 typed replay/frame proof/canonical metrics: retained.
- C2/C3/C4/C5 evidence: retained as scoped motivation, not SpikeRA result.
- P2-A artifact/access, seed/split, normalization, sealed-evaluation infrastructure: retained.
- P2-A dataset/target/scenario and model-role assumptions: major amendment.
- CA-DT remains mandatory; it becomes the dual-side classical detector/action baseline.
- N-SPLIT remains the target direct-gain baseline; N-CTX is replaced in the proposed-model role by SpikeRA dual-SoW, while structured Q/R remains a comparator.
- P2-A0 must not proceed unchanged; an AR1 amendment must add ST-primary dual-SoW extension points before implementation.

## 15. Final review disposition

Scientific architecture review results:

```text
Abstract claim audit                 COMPLETE
Single gate vs dual SoW             COMPLETE
MEKF/Split architecture analysis    COMPLETE
Sensor configuration review         COMPLETE
SNN role and falsification plan      COMPLETE
Prior-art and novelty review         COMPLETE
Existing phase-plan impact review    COMPLETE
User decision packet                 COMPLETE
```

Formal exact-repository result:

```text
Status: BLOCKED_SPIKERA_EVIDENCE_OR_ARCHITECTURE_INSUFFICIENT
Blocker: live repository and current fingerprints unavailable in this runtime
SpikeRA architecture decision: NOT FROZEN
P2-A0 implementation: NOT AUTHORIZED
SpikeRA implementation: NOT AUTHORIZED
Recommended decision state: READY_FOR_USER_SPIKERA_ARCHITECTURE_DECISION
Repository integration remains on hold until the bundle is copied into the live
repository and entry/exit fingerprints are verified without modifying frozen areas.
```

## Source register

### Project and submission sources

- **[PROMPT]** `P2-AR0-SPIKERA-REBASELINE-REVIEW-v1`, uploaded exact instruction artifact; local SHA-256 `aaa5db0bff03ef53f9f2a25cf43434c3858a1c1ea461a7014252cecc07bd64c5`.
- **[ABS]** *Spike-Based Reliability Adaptation for Neural Kalman Filtering in Spacecraft Attitude Estimation*, IAA SpaceAI 2026 abstract, one page. Read as parsed text and page image from the uploaded/File Library asset. No numerical table, baseline matrix, split, model size, or result trace is present.
- **[P01]** `docs/research/phase0a/decision_lock/P0_05_MEKF_MATH_CONTRACT.md` and Phase 0–1 evidence index: right-local 6D MEKF convention.
- **[P1]** `docs/research/phase1b/AI_ADCS_PHASE0_1_MASTER_SUMMARY_AND_PHASE2_HANDOFF.md`, local navigation copy SHA-256 `8827f4a3996b0e1f6b736de13a33a738bab06bee30e789bc1e85876e8dd40526`; navigation only.
- **[P1-NUM]** `docs/research/index/phase0_1/NUMERIC_EVIDENCE_CATALOG.md`, local copy SHA-256 `3092f435a2d69d1748e9f46871ead8437ccc8e4b872d2fe338cfe7ce465e50d1`.
- **[DR0]** `P2_00_DESIGN_REVIEW.md` through `P2_04_METRIC_AND_GATE_CONTRACT.md` plus `P2_DR0_DESIGN_FREEZE_AUDIT.md`; read from File Library. Prior audit-declared SHA-256 values are listed in the audit document, not independently recomputed here.
- **[P2A]** `P2_A_IMPLEMENTATION_REVIEW.md`, stage plan, file/interface map, test matrix, numeric dependency ledger, and audit; read from File Library. Existing PASS and non-authorization state are retained.

### Primary literature reviewed

- **[L1]** Revach et al., *KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics*, IEEE TSP 2022, arXiv:2107.10043, DOI 10.1109/TSP.2022.3158588.
- **[L2]** Choi et al., *Split-KalmanNet: A Robust Model-Based Deep Learning Approach for SLAM*, arXiv:2210.09636.
- **[L3]** Ni, Revach, Shlezinger, *Adaptive KalmanNet: Data-Driven Kalman Filter with Fast Adaptation*, ICASSP 2024, arXiv:2309.07016.
- **[L4]** Zhang et al., *Change-Aware Self-Adaptive AI-Aided Kalman Filters With Neural Change Point Detection*, arXiv:2607.13387.
- **[L5]** Juárez-Lora et al., *Implementation of Kalman Filtering with Spiking Neural Networks*, Sensors 2022, 22(22):8845, DOI 10.3390/s22228845.
- **[L6]** Xiao et al., *Spike-Kal: A Spiking Neuron Network Assisted Kalman Filter*, arXiv:2504.12703.
- **[L7]** Liu, Xu, Ou, *Spiking Neural-Invariant Kalman Fusion for Accurate Localization Using Low-Cost IMUs*, arXiv:2601.08248.
- **[L8]** Diker and Klein, *Neural Aided Adaptive Innovation-Based Invariant Kalman Filter*, arXiv:2603.26709.
- **[L9]** Vogt et al., *FlexKalmanNet: A Modular AI-Enhanced Kalman Filter Framework Applied to Spacecraft Motion Estimation*, arXiv:2405.03034.
- **[L10]** Park and D'Amico, *Adaptive Neural Network-based Unscented Kalman Filter for Robust Pose Tracking of Noncooperative Spacecraft*, JGCD 2023, arXiv:2206.03796, DOI 10.2514/1.G007387.
- **[L11]** Hashim, Abouheaf, Vamvoudakis, *Neural-adaptive Stochastic Attitude Filter on SO(3)*, IEEE Control Systems Letters 2022, DOI 10.1109/LCSYS.2021.3123227.
- **[L12-WITHDRAWN]** Mehrfard et al., *Adaptive Learned State Estimation based on KalmanNet*, arXiv:2604.02441. The latest arXiv record is withdrawn; it is treated only as a weak contextual signal and not as stable authority.
