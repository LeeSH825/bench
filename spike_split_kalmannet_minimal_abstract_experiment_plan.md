# Spike-Split KalmanNet Minimal Abstract Experiment Plan
## Reuse-Oriented Plan for IAA Conference on AI for Space

> **Historical legacy plan:** this plan targets a Euclidean six-dimensional
> observation benchmark in which gyro is an observation. It does not describe
> or validate the current right-local Phase 2 estimator, where gyro is a
> propagation/process input. Its proposed claims are not current evidence.

**Method name:** Spike-Split KalmanNet  
**Target submission:** IAA Conference on AI for Space  
**Goal:** produce the minimum experimental evidence needed to write a defensible abstract.

This document replaces the earlier "build everything from scratch" plan. The current repository audit indicates that the general ADCS/Basilisk benchmark infrastructure, NTD dataset format, Basilisk-based attitude/IMU data generation, EKF baseline, Split-KalmanNet baseline, runner/report/test infrastructure, and generic metrics are already implemented. Therefore, this plan focuses only on the missing Spike-Split-specific components.

---

# 0. Purpose

The purpose of this short plan is to add **Spike-Split KalmanNet** on top of the existing ADCS/Basilisk + Split-KalmanNet benchmark stack, not to rebuild the benchmark from zero.

The abstract-level claim should be limited:

> We extend the existing Split-KalmanNet ADCS benchmark with a spiking neural module in the innovation covariance branch and evaluate whether this targeted SNN insertion improves measurement-event robustness while preserving nominal tracking behavior.

The implementation focus is therefore:

```text
Existing benchmark infrastructure
+ event-aware measurement disturbance layer
+ Spike-Split model adapter
+ G2-SNN branch
+ event/spike metrics
+ minimal report extensions
```

This plan explicitly avoids claiming that SNNs are universally superior, that SNNs guarantee hardware energy savings, or that Basilisk directly validates real low-cost versus high-cost IMU hardware behavior.

---

# 1. Reuse Inventory

The following items should be treated as **already implemented and reusable**. They should not appear as new implementation tasks.

## 1.1 Benchmark common infrastructure

### NTD data format and save/load

Reuse:

```text
bench/tasks/data_format.py
CANONICAL_LAYOUT_V0 = "NTD"
save_npz_split_v0
load_npz_split_v0
```

Official datasets should remain in the existing `.npz + meta_json` NTD cache format. Do not switch the plan to standalone `.pt` files except as an optional export.

### Task/generator dispatch

Reuse:

```text
bench/tasks/bench_generated.py
```

Existing routing includes:

```text
basilisk_adcs_v0
basilisk_imu_*
adcs_replay_v0
```

### Model registry

Reuse:

```text
bench/models/registry.py
```

Existing registered models include:

```text
kalmannet_tsp
split_knet
me_split_knet_v0
basilisk_mrp_ekf
mb_kf_*
```

Spike-Split should be added as a new model entry rather than replacing existing entries.

### Runner CLI

Reuse:

```text
bench/runners/run_suite.py --help
```

The existing runner should remain the official entry point for the abstract experiments.

### Test suite

Reuse:

```text
python -m bench.tests.run_all --device cpu
```

The current audit reports:

```text
ALL TESTS PASSED
```

Any Spike-Split changes should preserve this status and add a smoke suite/test rather than bypassing the existing test path.

---

## 1.2 ADCS/Basilisk/IMU data generation

### Basilisk spacecraft attitude trajectory

Reuse:

```text
bench/tasks/generator/basilisk_adcs.py
```

Already supports spacecraft attitude trajectories with:

```text
x = [sigma_BN, omega_BN_B]
```

### IMU-like gyro measurement

Reuse:

```text
bench/tasks/generator/basilisk_imu_adcs.py
```

Already supports gyro and delta-angle-like measurement generation.

### Gyro bias state dataset

Reuse:

```text
bench/tasks/generator/basilisk_imu_adcs.py
```

Already supports `x_dim=9` with a bias state included.

### Sparse attitude reference dataset

Reuse existing sparse reference and measurement mask support.

### Structured corruption / low-cost sensor corruption profile

Reuse:

```text
bench/configs/suite_basilisk_structured_corruption_smoke.yaml
```

This should be treated as an existing sensor-corruption path, not a new generator requirement.

### Noise schedule

Reuse and extend:

```text
bench/tasks/generator/noise_schedule.py
```

Existing support includes:

```text
step_change
per_step_jump
jump_events
```

Spike-Split event datasets should extend this mechanism rather than implement a separate event scheduler.

---

## 1.3 Existing models

### EKF baseline

Reuse:

```text
bench/models/basilisk_mrp_ekf.py
bench/configs/suite_basilisk_mrp_ekf_smoke.yaml
```

Do not write "implement EKF baseline" in the task list. Use the existing `basilisk_mrp_ekf` baseline.

### Split-KalmanNet baseline

Reuse:

```text
bench/models/split_knet.py
bench/configs/suite_basilisk_me_split_smoke.yaml
```

Do not write "implement Split-KalmanNet baseline" in the task list. Use the existing `split_knet` model.

### Measurement-Enhanced Split-KalmanNet

Optional existing baseline:

```text
bench/models/me_split_knet.py
me_split_knet_v0
```

This can be included only if time permits. It is not required for the minimal abstract.

### Generic model-based KF baselines

Reuse:

```text
bench/models/mb_kf.py
```

These are optional for the abstract-level run unless a generic baseline is already in the chosen suite.

---

## 1.4 Existing metrics and reports

### Generic metrics

Reuse:

```text
bench/metrics/core.py
```

Existing support includes:

```text
generic MSE
generic RMSE
MSE(dB)
Gaussian NLL
compute_shift_recovery_k
```

### Report and plot infrastructure

Reuse the existing report/plot smoke-tested infrastructure. Spike-Split-specific plots should be implemented as extensions to the existing report system rather than as standalone one-off scripts when possible.

---

# 2. Gap List

Only the following items remain in scope as new work for the Spike-Split abstract experiment.

## 2.1 Measurement-event dataset semantics

Current infrastructure has noise schedules and structured corruption, but the exact event semantics required for the Spike-Split abstract are not yet implemented.

Implement or extend:

```text
event_flag saved in NTD extras and/or meta_json
event_start / event_end / event_duration metadata
event-window gyro noise scale metadata
transient gyro bias jump
metric-readable event/non-event segmentation
```

Implementation direction:

```text
Extend bench/tasks/generator/basilisk_imu_adcs.py
and/or bench/tasks/generator/noise_schedule.py
```

Do not create a separate incompatible dataset generator unless absolutely necessary.

---

## 2.2 Spike-Split model components

Implement:

```text
bench/models/spike_split_knet.py
```

or implement a separate adapter that extends the existing `split_knet.py` behavior.

Required new components:

```text
SpikeSplitKNetAdapter
G2SNNBranch
recurrent LIF SNN
surrogate gradient training path
spike-rate logging
active_ops_proxy logging
```

P1 ablation:

```text
G1SNNBranch or G1-SNN ablation adapter
```

Optional P2 baseline:

```text
Event-Gated G2-GRU baseline
```

Definition of Spike-Split KalmanNet:

```text
Base: existing Split-KalmanNet
Keep: G1 branch as existing RNN branch
Replace: G2 branch with recurrent SNN branch
Input to G2-SNN: innovation-derived features, optionally event flag
Output of G2-SNN: constrained positive innovation-covariance inverse factor/vector
```

The model should be registered in:

```text
bench/models/registry.py
```

with a name such as:

```text
spike_split_knet
```

---

## 2.3 Spike-Split-specific metrics

Generic MSE/RMSE/recovery already exists, but the following ADCS event and SNN metrics are missing.

Implement in a separate module rather than overloading `bench/metrics/core.py`:

```text
bench/metrics/adcs_event.py
```

Required event metrics:

```text
full-trajectory attitude RMSE [deg]
angular velocity RMSE
event-window attitude RMSE [deg]
peak attitude error after event
event-based recovery time
```

Required SNN metrics:

```text
average spike rate
non-event spike rate
event-window spike rate
active_ops_proxy
```

Notes:

- `compute_shift_recovery_k` can be reused internally, but a wrapper is needed to compute recovery around `event_start` / `event_end`.
- Attitude RMSE must account for the state representation. If the dataset uses MRP, convert MRP attitude error to an angle in degrees. If quaternion is later used, use quaternion angular distance.
- `active_ops_proxy` must be reported as an activity proxy, not as hardware energy.

---

## 2.4 Spike-Split reports and plots

Extend:

```text
bench/reports/plots.py
bench.reports.make_report
```

Required report additions:

```text
Spike-Split event/spike metrics in aggregate CSV/report
event-response plot
error-comparison-around-event plot
```

Event-response plot should include at least:

```text
event_flag
innovation norm
G2-SNN spike rate
attitude error
```

Error-comparison plot should compare around event windows:

```text
basilisk_mrp_ekf
split_knet
spike_split_knet
optional: g1_snn_split_knet
```

---

# 3. Revised Minimal Experiment

The minimal abstract experiment should reuse existing benchmark components and add only event/spike-specific pieces.

## 3.1 Dataset A: Nominal ADCS/IMU dataset

Purpose:

```text
Verify that Spike-Split does not destroy nominal attitude tracking behavior.
```

Implementation:

```text
Reuse existing Basilisk ADCS/IMU generators.
No event_flag required, or event_flag all false.
Use existing NTD `.npz + meta_json` cache through bench_generated.
```

Official output:

```text
bench_generated cache
train.npz / val.npz / test.npz through existing NTD save/load path
associated meta_json
```

Do not use the previous plan's `data/nominal_train.pt` as an official artifact. If needed, `.pt` export can be an optional debugging export only.

Recommended minimal size under time pressure:

```text
train: 100 trajectories
val: 20 trajectories
test: 20 trajectories
T: existing smoke/short trajectory length, or 200-500 if already feasible
```

---

## 3.2 Dataset B: Measurement-event ADCS/IMU dataset

Purpose:

```text
Evaluate whether replacing only the G2 branch with an SNN is useful under measurement-side event disturbances.
```

Implementation:

```text
Reuse existing Basilisk ADCS/IMU generator.
Extend noise_schedule / structured corruption path with:
- event_flag
- transient gyro bias jump
- event-window gyro noise scale
- event_start / event_end / duration metadata
```

Official output:

```text
bench_generated NTD `.npz + meta_json`
```

Suggested event semantics:

```text
event_start: random or configured location in middle 30%-70% of trajectory
event_duration: 5%-15% of trajectory length
gyro_noise_scale_event: 5x-20x nominal
transient_bias_jump: small-to-moderate angular-rate offset
```

Minimum checks:

```text
event_flag exists and is aligned with the event window
metadata records event_start/end/duration/noise scale
bias jump is applied only in event window or with specified decay
innovation norm increases around event for EKF or Split-KalmanNet baseline
```

---

## 3.3 Models

Use the existing model registry and runner.

### Required models for the minimal abstract

```text
1. EKF baseline: existing basilisk_mrp_ekf
2. Split-KalmanNet baseline: existing split_knet
3. Spike-Split KalmanNet: new spike_split_knet
```

### Strongly recommended P1 ablation

```text
4. G1-SNN Split-KalmanNet: new ablation
```

This ablation is recommended because the abstract's methodological claim is not merely "SNN works," but that **targeted G2 replacement** is a compact insertion strategy for measurement-side event uncertainty.

### Optional if time permits

```text
5. ME-Split-KalmanNet: existing me_split_knet_v0
6. Event-Gated G2-GRU: new optional baseline
```

If Event-Gated G2-GRU is not implemented, avoid claiming that Spike-Split outperforms all event-aware dense recurrent alternatives.

---

## 3.4 Metrics

Reuse existing generic metrics:

```text
MSE
RMSE
MSE(dB)
Gaussian NLL if already available in the suite
compute_shift_recovery_k where applicable
```

Add Spike-Split-specific metrics:

```text
full_attitude_rmse_deg
omega_rmse
event_attitude_rmse_deg
event_peak_attitude_error_deg
event_recovery_time
avg_spike_rate
non_event_spike_rate
event_spike_rate
active_ops_proxy
```

Minimum metrics needed for abstract wording:

```text
nominal full_attitude_rmse_deg
event full_attitude_rmse_deg
event_attitude_rmse_deg
event_peak_attitude_error_deg
avg_spike_rate
event_spike_rate
active_ops_proxy
```

---

# 4. Implementation Plan

## P0: Required for abstract evidence

### P0.1 Add event semantics to NTD dataset outputs

Task:

```text
Add event_flag and event metadata to existing NTD dataset extras/meta_json.
```

Target files:

```text
bench/tasks/generator/basilisk_imu_adcs.py
bench/tasks/generator/noise_schedule.py
bench/tasks/data_format.py only if extras/meta handling needs extension
```

Deliverables:

```text
event_flag available per trajectory/time step
event_start, event_end, event_duration in meta_json
event_noise_scale and bias_jump parameters in meta_json
```

---

### P0.2 Add transient gyro bias jump / event measurement disturbance

Task:

```text
Extend existing Basilisk IMU generator or corruption/noise schedule to support event-window measurement disturbance.
```

Minimum disturbance:

```text
gyro noise scale increase during event window
transient gyro bias jump during event window
```

Do not implement a full vibration physics model for the abstract. Use a parameterized disturbance and describe it as a measurement-event disturbance.

---

### P0.3 Implement SpikeSplitKNetAdapter

Task:

```text
Add Spike-Split model as a new adapter/model, preferably in bench/models/spike_split_knet.py.
```

Reuse:

```text
existing split_knet implementation
existing runner/model registry path
```

Expected registry name:

```text
spike_split_knet
```

Definition:

```text
G1: existing RNN branch from Split-KalmanNet
G2: new recurrent SNN branch
```

---

### P0.4 Implement G2SNNBranch

Task:

```text
Implement recurrent LIF SNN branch for G2.
```

Minimum design:

```text
input: innovation-derived features from existing Split-KNet feature path
optional input: event_flag
hidden: recurrent LIF layer
training: surrogate gradient
output: positive diagonal vector or positive factor for G2
logging: spike counts per step
```

Safety constraint:

```text
G2 output must be finite and positive, e.g., softplus(z) + epsilon.
```

---

### P0.5 Add ADCS event metrics

Task:

```text
Implement event-window attitude metrics in a new module.
```

Target:

```text
bench/metrics/adcs_event.py
```

Minimum functions:

```text
compute_attitude_rmse_deg(...)
compute_event_window_attitude_rmse_deg(...)
compute_event_peak_attitude_error_deg(...)
compute_event_recovery_time(...)
```

Notes:

```text
Use MRP-to-angle or quaternion angular distance depending on representation.
Reuse compute_shift_recovery_k where appropriate.
```

---

### P0.6 Add spike-rate and active_ops_proxy logging

Task:

```text
Log SNN spike statistics into metrics.json or model output diagnostics.
```

Minimum fields:

```text
avg_spike_rate
non_event_spike_rate
event_spike_rate
active_ops_proxy
```

Definition:

```text
active_ops_proxy = spike_count * fanout
```

Guardrail:

```text
This is not hardware energy. It is only an activity proxy.
```

---

### P0.7 Add smoke suite

Task:

```text
Add a minimal suite for Spike-Split smoke testing.
```

Suggested file:

```text
bench/configs/suite_basilisk_spike_split_smoke.yaml
```

The suite should run quickly on CPU and verify:

```text
Dataset generation succeeds
basilisk_mrp_ekf runs
split_knet runs
spike_split_knet runs
metrics.json contains event/spike fields where applicable
```

---

## P1: Strongly recommended if time permits

### P1.1 Implement G1-SNN ablation

Task:

```text
Add G1-SNN ablation using the same SNN branch style as G2-SNN.
```

Purpose:

```text
Test whether the SNN insertion location matters.
```

Suggested registry name:

```text
g1_snn_split_knet
```

---

### P1.2 Add event-response plot

Target:

```text
bench/reports/plots.py
```

Plot signals:

```text
event_flag
innovation norm
G2-SNN spike rate
attitude error
```

Purpose:

```text
Internal verification for abstract wording.
```

---

### P1.3 Add error-comparison-around-event plot

Target:

```text
bench/reports/plots.py
```

Models:

```text
basilisk_mrp_ekf
split_knet
spike_split_knet
optional: g1_snn_split_knet
```

Purpose:

```text
Check whether Spike-Split improves or at least changes event-window response.
```

---

### P1.4 Extend report aggregates

Task:

```text
Add Spike-Split-specific event/spike metric columns to existing report aggregate/summary outputs.
```

Do not force a standalone `results/summary_minimal.csv`. Use or extend the existing report pipeline.

---

## P2: Optional after abstract evidence

### P2.1 Event-Gated G2-GRU baseline

Purpose:

```text
Separate event-awareness from spiking dynamics.
```

Guardrail:

```text
If event_flag is used as input to Spike-Split, then Event-Gated G2-GRU or a no-event-flag ablation is needed before claiming that the benefit comes specifically from spiking dynamics.
```

---

### P2.2 Basilisk event scenario refinement

Optional refinements:

```text
multiple event windows
control-correlated event windows
structured saturation/clipping
more realistic actuator-induced disturbance
```

Do not block abstract submission on these.

---

### P2.3 Hardware-oriented measurement

Optional future work:

```text
MCU/FPGA timing
neuromorphic backend
real energy measurement
```

Until then, report only `active_ops_proxy`, not energy.

---

# 5. Output Contract

Use the existing runner/report output structure.

## 5.1 Official runner outputs

Expected official artifacts:

```text
run_dir/config_snapshot.yaml
run_dir/run_plan.json
run_dir/budget_ledger.json
run_dir/checkpoints/model.pt
run_dir/metrics.json
run_dir/metrics_step.csv
run_dir/timing.csv
reports/summary_*.csv
reports/aggregate_*.csv
```

These replace the earlier plan's one-off artifacts such as:

```text
data/*.pt
results/*.json
results/summary_minimal.csv
```

Those old paths are incompatible with the current benchmark structure and should be removed from the official plan. They may exist only as optional debug exports.

---

## 5.2 Spike-Split additional output fields

Spike-Split runs should add the following fields to `metrics.json` and aggregate reports where possible:

```text
full_attitude_rmse_deg
omega_rmse
event_attitude_rmse_deg
event_peak_attitude_error_deg
event_recovery_time
avg_spike_rate
non_event_spike_rate
event_spike_rate
active_ops_proxy
```

---

## 5.3 Additional report plots

If P1 plotting is implemented, produce:

```text
event-response plot
error-comparison-around-event plot
```

These plots are for internal inspection and potential future paper/poster use. They are not required inside the IAA abstract text, but they help guard against overclaiming.

---

# 6. Abstract Claim Guardrails

Use only claims supported by results from the same runner, split, seeds, and dataset configuration.

## 6.1 Baseline guardrail

Because EKF and Split-KalmanNet baselines already exist, any abstract claim must compare against results generated using the same runner and dataset split.

Allowed:

```text
Using the existing benchmark runner and identical Basilisk/IMU dataset splits, Spike-Split is compared with basilisk_mrp_ekf and split_knet.
```

Avoid:

```text
Comparisons across different ad hoc scripts, different seeds, or different dataset generation paths.
```

---

## 6.2 Performance claim guardrail

If Spike-Split does not improve event-window RMSE or peak attitude error relative to Split-KalmanNet, do not claim performance superiority.

Allowed if event-window metrics improve:

```text
Spike-Split improves event-window attitude estimation under measurement-event disturbances.
```

Allowed if accuracy is similar but sparsity improves:

```text
Spike-Split achieves comparable attitude estimation accuracy while reducing active neural operation proxy.
```

Avoid unless supported:

```text
Spike-Split outperforms Split-KalmanNet.
```

---

## 6.3 Energy claim guardrail

`active_ops_proxy` is not hardware energy.

Allowed:

```text
Spike-Split reduces an active neural operation proxy through sparse spiking activity.
```

Avoid:

```text
Spike-Split reduces onboard power consumption.
Spike-Split is energy efficient in hardware.
```

These require hardware or backend-specific measurements.

---

## 6.4 Event-awareness guardrail

If `event_flag` is used as input to Spike-Split, the abstract must not imply that all improvements come purely from spiking dynamics unless a proper ablation is included.

Required to isolate the effect:

```text
Event-Gated G2-GRU baseline
or
Spike-Split without event_flag ablation
```

If neither is implemented, phrase the claim as:

```text
event-aware spiking innovation covariance adaptation
```

not:

```text
spiking dynamics alone explains the improvement
```

---

## 6.5 Low-cost/high-cost IMU guardrail

Do not claim that the experiment proves real low-cost or high-cost IMU behavior. Basilisk does not provide a direct low/high IMU hardware option in this benchmark path.

Allowed:

```text
parameterized IMU measurement disturbances
structured sensor corruption profile
low-grade-like synthetic IMU profile, if explicitly defined as synthetic
```

Avoid:

```text
proves low-cost IMU is worse than high-cost IMU
validates real low-cost IMU vibration sensitivity
```

---

# 7. Minimal Execution Checklist

Before drafting the IAA abstract, the following should exist:

```text
[ ] Existing tests still pass: python -m bench.tests.run_all --device cpu
[ ] Nominal ADCS/IMU NTD dataset generated through bench_generated
[ ] Measurement-event ADCS/IMU NTD dataset generated through bench_generated
[ ] event_flag stored in extras/meta_json and readable by metrics
[ ] transient gyro bias jump and event-window noise scale applied
[ ] basilisk_mrp_ekf result from existing runner
[ ] split_knet result from existing runner
[ ] spike_split_knet result from existing runner
[ ] metrics.json contains event metrics
[ ] metrics.json contains spike metrics for Spike-Split
[ ] aggregate/summary report includes key event/spike fields
[ ] Optional but recommended: g1_snn_split_knet ablation result
[ ] Optional but recommended: event-response plot inspected
```

---

# 8. Minimal Abstract Template

Use this only after results are available.

```text
Neural network-aided Kalman filters have shown promise for spacecraft state estimation under partially known dynamics, but their recurrent modules are typically implemented as dense sequence processors. This work proposes Spike-Split KalmanNet, a spiking neural extension of Split-KalmanNet for spacecraft attitude estimation. Building on an existing Basilisk-based ADCS benchmark, the proposed method preserves the Split-KalmanNet prediction-update structure while replacing only the innovation covariance branch with a recurrent spiking module driven by innovation-derived event features. We extend the benchmark with parameterized measurement-event disturbances, including event-window gyro noise scaling and transient gyro bias jumps, and evaluate the method against existing EKF and Split-KalmanNet baselines using the same runner and dataset splits. Preliminary results show that Spike-Split KalmanNet [preserves/improves] nominal tracking accuracy and [improves/matches] event-window attitude estimation while maintaining sparse spiking activity measured by an active neural operation proxy. These results suggest that targeted spiking replacement of the innovation covariance branch is a compact and interpretable strategy for event-aware neural Kalman filtering in spacecraft applications.
```

Replace the bracketed phrases only after the actual metrics are available.

---

# 9. Final Conclusion

The current repository state already provides:

```text
general ADCS/Basilisk data generation: implemented
EKF baseline: implemented
Split-KalmanNet baseline: implemented
runner/report/metric infrastructure: implemented
```

The missing pieces are:

```text
Spike-Split-specific SNN branch: not implemented
event_flag / transient bias jump / event-window semantics: not implemented
event-window attitude metrics: not implemented
spike metrics / active_ops_proxy: not implemented
Spike-Split report/plot extensions: not implemented
```

Therefore, the next step is **not** to build a new benchmark from scratch. The next step is to layer an **event-aware measurement disturbance path**, a **G2-SNN branch**, and **event/spike-specific metrics and reports** on top of the existing ADCS/Basilisk + Split-KalmanNet benchmark.
