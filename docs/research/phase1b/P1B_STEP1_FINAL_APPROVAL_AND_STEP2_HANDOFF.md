# Phase 1B Step 1 Final Approval and Step 2 Handoff

- 결정일: 2026-08-02
- Step 1 상태: **PASS_P1B_STEP1_UNIT_ST_CLASSICAL**
- 다음 단계: **Phase 1B Step 2 — Magnetometer/Sun-Sensor Fusion, C4 Combined Event, and P1 Exit Review**
- Phase 2/Neural 진행: **승인하지 않음**

## 1. Step 1 최종 승인

다음을 승인한다.

- stationary UNIT-ST F-BASE 기준선
- train/validation-only fixed Q/R tuning
- fixed Q/R mismatch sensitivity
- C2 gyro process-uncertainty pilot
- C3 star-tracker inlier-reliability pilot
- C5 innovation-RMS-matched process/measurement A/B pilot
- paired test N=50 per required condition
- 600 s stationary long-horizon subset
- raw sensor artifact와 simulation-only oracle sidecar 분리
- fixed/tuned estimator의 oracle/event-label 비의존성
- Gate C canonical metrics와 Phase 1A regression 유지

## 2. Step 1에서 잠근 해석

### Primary classical baseline

```text
F-BASE
```

F-BASE는 stationary matched 조건과 600 s long-horizon에서 안정성과 consistency를
보였다. 이후 Phase 1 classical 비교의 primary reference로 유지한다.

### F-TUNED의 지위

동결 값:

```text
s_Qg = 0.125
s_Qb = 0.125
s_R  = 8.0
```

F-TUNED은 short-horizon attitude/bias objective를 약간 개선했으나 매우 낮은
NIS/NEES와 long-horizon attitude penalty를 보였다. 따라서 primary baseline이
아니며 frozen sensitivity comparator로만 사용한다. test 결과를 보고 재튜닝하지 않는다.

### Preliminary hypotheses

- H1: time-varying adaptation 필요성은 UNIT-ST에서 예비 지원됨. C3가 강하고 C2는 완만함.
- H2: correct-side oracle은 C3 accuracy/consistency와 C2 consistency에서 유용함.
- H3: process-side와 measurement-side action 분리는 예비 지원됨.
- H4: scalar innovation RMS만으로 원인을 식별하기 어렵다는 주장은 해당 C5 pair에 한해 지원됨.
- C2 oracle의 attitude RMSE 개선은 확인되지 않음.
- 일반적 정보이론적 식별 불가능성을 주장하지 않음.

## 3. Step 2 목적

Step 2는 Phase 1의 마지막 구현·실험 단계다.

수행:

- parameterized magnetometer model
- parameterized sun-vector model과 validity
- gyro+mag `STRESS-MAG`
- gyro+mag+sun+low-rate ST `MAIN-FUSION`
- asynchronous four-sensor event ordering
- magnetometer/sun update replay
- sensor-specific NIS와 consistency
- C4 slow process + fast measurement combined event
- process-only / measurement-only / full oracle comparison
- paired Monte Carlo
- Phase 1 Exit Review

미수행:

- false star-tracker solution
- heavy-tailed outlier
- learned robust gate
- nonzero latency/OOSM
- flight-product sensor tuning
- KalmanNet/Split-KalmanNet
- ANN/SNN/FPGA
- closed-loop control

## 4. Step 2 설계 결정

### Truth

Basilisk는 rigid-body attitude/rate truth source로 유지한다.
이번 단계에서 full orbit/WMM/eclipse dynamics를 필수로 만들지 않는다.

Magnetic and sun inertial references are versioned, deterministic,
parameterized reference-vector providers. They are research benchmark inputs,
not flight environment claims.

### Magnetometer

```text
z_mag_B = C_BN(q_true) r_mag_N + b_mag + n_mag
```

Step 2 primary path uses zero mean bias and full-rank inlier noise.
Mean interference, saturation and gross outliers are deferred.

### Sun sensor

```text
h_sun_B = C_BN(q_true) r_sun_N
z_sun_B = normalized tangent-perturbed body sun vector
```

The update uses a deterministic 2D tangent residual/Jacobian.
Validity/FOV/eclipse-style mask may skip updates; invalid rows are never converted
to zero measurements.

### Same-time order

```text
gyro → magnetometer → sun → star tracker
```

This order is deterministic and versioned.

### C4 primary combined event

```text
slow process:
  gyro bias random-walk intensity alpha_b(t)

fast measurement:
  magnetometer inlier covariance alpha_R_mag(t)
```

The fast event is covariance change, not mean interference or outlier.
A secondary ST reliability combined case may be included only after the primary C4
case is complete and must not replace it.

## 5. Phase 1 Exit principle

Step 2 completion alone does not automatically authorize Phase 2.
The final `P1_EXIT_REVIEW.md` must explicitly decide:

```text
GO
CONDITIONAL_GO
STOP
```

based on classical baseline stability, sensor fusion, consistency, problem
existence, oracle usefulness, information boundaries and reproducibility.

No neural implementation begins in the Step 2 execution.
