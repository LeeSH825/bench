# KalmanNet Family — Task Catalog & Benchmark Generator Mapping
*(KalmanNet / Adaptive-KalmanNet(AKNet) / MAML-KalmanNet / Split-KalmanNet)*

This note is designed to be **pasted into prompts / design docs** when implementing new benchmark tasks under `bench/tasks/generator/`.
It extracts the **tasks (simulation scenarios + datasets)** used in the four papers and reformulates them as a small set of **task families**
that can be generated parametrically (instead of hard-coding “paper-by-paper” tasks).

---

## 0) Common definitions (benchmark-friendly)

### Task (in these papers)
A **task** is (implicitly) a *state-space model* + a *mismatch/shift regime* + *data split protocol*:

- Dynamics: `x_{t+1} = f(x_t) + e_t`, `e_t ~ (0, Q_t)`
- Measurement: `y_t = h(x_t) + v_t`, `v_t ~ (0, R_t)`
- Plus: **mismatch** (true model vs assumed model), and/or **shift** (time-varying `Q_t`, `R_t`, or switching dynamics).

### Task-set (meta-learning / adaptation papers)
A **task-set** is a distribution over tasks, often parameterized by **noise levels**:
- MAML-KalmanNet: tasks are indexed by `(q^2, r^2)` (or `V = 10 log10(q^2/r^2)`), i.e., **different noise settings** share the same SSM.
- AKNet: focuses on **time-varying noise statistics** summarized by `SoW_t = q_t^2 / r_t^2` (or in dB).

### “True vs assumed” (critical for benchmark fairness)
Many experiments explicitly generate data with a **true** model and run baselines with an **assumed** (mismatched) model.
For benchmark generators, this is best represented as:
- `meta.ssm.true = {...}` (used for data generation)
- `meta.ssm.assumed = {...}` (what model-based filters / baselines are told)

---

## 1) Task families across papers (the smallest useful basis)

Instead of adding “KalmanNet paper tasks” one-by-one, implement **~6 generator families** and instantiate them via YAML/params.

| Task family (generator) | KalmanNet (2022) | AKNet (2024) | MAML-KalmanNet (2025) | Split-KalmanNet (2023) |
|---|---:|---:|---:|---:|
| **F1. UCM (uniform circular motion)** | (ref in [4] of MAML) |  | ✅ | ✅ |
| **F2. Linear Gaussian + mismatch (rotated F/H)** | ✅ | ✅ (baseline) | ✅ (as part of UCM linear) | ✅ |
| **F3. Synthetic nonlinear (sinusoidal f, polynomial h)** | ✅ |  |  |  |
| **F4. Lorenz attractor family** | ✅ |  | ✅ |  |
| **F5. Time-varying noise schedule / SoW** |  | ✅ |  | ✅ |
| **F6. Switching dynamics (abrupt model changes)** |  |  | ✅ |  |
| **F7. Real-world datasets (NCLT / UZH-FPV)** | ✅ (NCLT) |  | ✅ (UZH-FPV) | ✅ (NCLT) |

---

## 2) Paper-by-paper task extraction (what they actually run)

### 2.1 KalmanNet (IEEE TSP 2022) — main tasks
**(A) Linear SS model with partial information**
- **State-evolution mismatch**: generate with rotated evolution matrix `F_{α}` while filters use canonical `F0`.
  - Relationship: `F_{α} = R_{xy}(α) * F0`, with `α ∈ {10°, 20°}` (paper Eq. (16), PDF p.9).
- **State-observation mismatch**: generate with rotated `H_{α=10°}`, while filters use `H = I`.
  - Interpreted as sensor misalignment (~5%) (PDF p.10).

**(B) Synthetic nonlinear SS model**
- Dynamics: `f(x) = α sin(β x + φ) + δ` (component-wise, `x ∈ R^2`)
- Measurement: `h(x) = a (b x + c)^2` (component-wise, `y ∈ R^2`)
  - Paper Eq. (17), PDF p.10.

**(C) Lorenz attractor family (continuous→discrete approximation + mismatch)**
- Continuous-time Lorenz ODE with `A(x)` (paper Eq. (18)).
- Discretization: `F(x) ≈ I + sum_{j=1..J} (A(x) Δτ)^j / j!` (paper Eq. (20)).
- Default: `J=5`, `Δτ=0.02`.
- Experiments include:
  1) **Noisy state observations**: `h = identity`, long test trajectories (`T=2000`) while trained on short (`T=100`) (PDF p.11).
  2) **Nonlinear observations** (paper Table VI; described around PDF p.11).
  3) **State-evolution mismatch**: generate with `J=5`, run filters with `J=2` approximation (PDF p.12).
  4) **Observation rotation mismatch**: data generated with a tiny `θ=1°` rotation (sensor misalignment ~0.55%) (PDF p.12).
  5) **Sampling mismatch / decimation**: generate with dense sampling `Δτ=1e-5`, then subsample by `1/2000` to `Δτd=0.02` (PDF p.12–13).

**(D) Real-world: Michigan NCLT dataset**
- Task: localize Segway robot using noisy odometry/velocity vs ground truth trajectory.
- Split protocol in KalmanNet paper: 85% train (23 sequences, `T=200`), 10% val (2 sequences, `T=200`), 5% test (1 sequence, `T=277`) (PDF p.14).

---

### 2.2 Adaptive-KalmanNet (AKNet, ICASSP 2024) — main tasks
AKNet is about **fast adaptation to time-varying noise statistics** using an input summarizer:
- `SoW_t = q_t^2 / r_t^2` (SoW = “signal-over-…”, used as conditioning input; PDF p.1).

**(A) Linear Gaussian system — generalization across noise settings**
- Train on **only 4** discrete `(q_t^2, r_t^2)` settings (caption Fig.2, PDF p.4).
- Test on:
  - unseen **scales** with same ratio, and
  - unseen **ratios** (SoW not seen in training).

**(B) Linear non-Gaussian system**
- Replace Gaussian noise with **exponential** noise; still generalize across unseen SoW/ratios (PDF p.4 around Fig.3).

**(C) Time-varying SoW + noisy SoW (online estimation errors)**
- Time variations simulated as **abrupt per-timestep jumps**:
  - previous `(q_{t-1}^2, r_{t-1}^2) = (1, 1)`, jump to `q_t^2 = 0.1`, `r_t^2 ∈ {0.01, 0.05, 0.1, 0.5, 1, 5, 10}` (PDF p.4).
- Also considers that the SoW used by AKNet at inference is **noisy** due to online estimation errors (PDF p.4, “Noisy SoWs”).

---

### 2.3 MAML-KalmanNet (IEEE TSP 2025) — main tasks
Core idea: treat different noise settings `(q^2, r^2)` as different **tasks**, meta-learn initial parameters, then few-shot fine-tune.

**Task inventory (paper Section IV “SIMULATIONS”, PDF p.9):**
A) UCM (uniform circular motion) — proof-of-concept + compare with AKNet  
B) Lorenz attractor — trajectory-length mismatch + model mismatch + data mismatch  
C) Reentry Vehicle Tracking (RVT) — handle abrupt model changes  
D) Real-world: UZH-FPV dataset — drone localization under scarce labels

#### (A) UCM: 2D rotation + linear/nonlinear measurement (PDF p.9)
- Dynamics (Eq. (26)):  
  `x_t = R(θ) x_{t-1} + e_t`, where `θ = 10°`, `x_t ∈ R^2`.
- Measurements (Eq. (27)):
  - linear: `y_t = x_t + v_t`
  - nonlinear: `[||x_t||, atan(y/x)]^T + v_t`
- They organize comparisons across supervised/unsupervised/semi-supervised/pretraining vs MAML-KalmanNet,
  and also compare vs **AKNet** under varying noise ratios.

#### (B) Lorenz: robustness to length mismatch, model mismatch, data mismatch (PDF p.11–12)
- Same Lorenz discretization structure as KalmanNet (J=5, Δτ=0.02).
- Subtasks:
  1) **Trajectory length mismatch**: `T_test=300` while AAL pretraining uses `T_AAL_train=30` (PDF p.11).
  2) **Model mismatch**: true uses `J=5`; mismatched pretraining uses `J=2` or `J=1` (PDF p.12, Table III).
  3) **Data mismatch (time misalignment)**: generate synchronized sequence at 2000 Hz length 6,000,000,
     remove first 2000 points, resample, add noise → measurement time-axis mismatch (PDF p.12, Fig.8).

#### (C) RVT: abrupt model changes + online re-training window (PDF p.13–14)
- RVT nonlinear SSM with radar measurements (Eq. (33)–(35)).
- Abrupt model changes:
  - switch between two models (Eq. (33) vs Eq. (36)) (PDF p.13).
  - In results discussion: abrupt changes occur during **t=200..250** (PDF p.14).
  - When change detected (residual > threshold): **collect 50 steps** and retrain/fine-tune (PDF p.14).
- Baseline includes IMM filter; MAML-KalmanNet aims to regain performance quickly after change.

#### (D) UZH-FPV dataset: constant acceleration linear SSM (PDF p.14)
- State: `x_t = [p, v, a] ∈ R^9`, measurement `y_t = a ∈ R^3`.
- Linear SSM (Eq. (40)–(43)), `Δτ=0.01`.
- Session: “6th indoor forward-facing”, sampled at 100 Hz → 3020 steps.
- Split protocol: 2/3 for training (**25 sequences**, `T=80`), remaining for testing (**1 sequence**, `T=1020`) (PDF p.14).

---

### 2.4 Split-KalmanNet (IEEE TVT 2023) — main tasks
Split-KalmanNet emphasizes robustness to **noise heterogeneity** and **time-varying noise statistics**.

**(A) UCM with linear/nonlinear measurement (PDF p.4)**
- Dynamics (Eq. (18)): `x_{t+1} = R(θ) x_t + w_t`, `x_t ∈ R^2`.
- Measurements (Eq. (19)):
  - linear: `y_t = x_t + v_t`
  - nonlinear: `[||x_t||, atan2(x_t)]^T + v_t`
- Noise: `Q_t = σ_w^2 I`, `R_t = σ_v^2 I`, define heterogeneity `ν = σ_v^2/σ_w^2` (PDF p.4).
- Train/test lengths shown in figure captions:
  - Example: training `T_ℓ = 15`, testing `T_ℓ = 100` (Fig.4 caption, PDF p.5).

**(B) Time-varying measurement noise statistics (PDF p.4)**
- `R_t = σ_{v,t}^2 I`, with time-varying `σ_{v,t}^2`:
  ```
  σ_{v,t}^2[dB] =
    floor(t/2) + t0 mod 50,   0<=t<=14 (train)
    floor(t/10)+30 mod 50,    0<=t<=999 (test)
  ```
  where `t0 ∈ {0,10,20,30}` sampled per training sequence (paper Eq. (20), PDF p.4).

**(C) Real-world: Michigan NCLT dataset (PDF p.5)**
- Task: localize Segway robot using noisy odometry.
- Protocol:
  - Train/val: session date **2012-01-22**, 1 Hz
  - Test: session date **2012-04-29**, 1 Hz
  - Split lengths: train `T_ℓ=50, L=80`, val `T_ℓ=200, L=5`, test `T_ℓ=2000, L=1` (PDF p.5).

---

## 3) Benchmark mapping: what to add under `bench/tasks/generator/`

### 3.1 Recommended generator modules (minimal set)
Implement these as *parametric* generators; each YAML task is just a parameterization.

1) `generator/ucm.py`
- Supports: linear measurement, nonlinear measurement.
- Params: `theta_deg`, `q2`, `r2`, `T`, `L_train/val/test`, noise type (Gaussian; optional).
- Matches: Split-KalmanNet Eq.(18)–(19), MAML Eq.(26)–(27).

2) `generator/linear_mismatch.py`
- Supports: **rotated-F mismatch** and **rotated-H mismatch**.
- Params: `alpha_deg` for true rotation; `assumed_F/H` or rotation=0.
- Matches: KalmanNet linear partial-information experiments.

3) `generator/sine_poly.py`
- Implements KalmanNet synthetic nonlinear:
  - `f(x) = α sin(β x + φ) + δ`, `h(x) = a (b x + c)^2`.
- Params: vector/scalar parameters; component-wise control; `T`, `L`, `q2`, `r2`.

4) `generator/lorenz.py`
- Implements Lorenz discretization with parameters:
  - `J_true`, `J_assumed`, `delta_tau_true`, `delta_tau_assumed` (often `Δτ_assumed=0.02`).
- Subtasks toggles:
  - `obs_mode`: `identity` vs `nonlinear` (if implementing nonlinear obs variants)
  - `obs_rotation_deg`
  - `sampling_mismatch`: dense sampling + decimation ratio (KalmanNet)
  - `data_time_mismatch`: time-axis shift/resample (MAML)
  - `T_train`, `T_test`
- Matches: KalmanNet + MAML Lorenz experiments.

5) `generator/noise_schedule.py` (shared utility)
- For **time-varying** `Q_t`, `R_t`:
  - piecewise-constant shift (`t0` step change)
  - per-timestep jumps
  - Split-KalmanNet Eq.(20) schedule
- Outputs both:
  - schedules (`q2_t`, `r2_t`) and
  - derived stats (`SoW_t`, `SoW_dB_t`)
- Needed by: Split-KalmanNet, AKNet, (and also useful for your existing `suite_shift.yaml` tasks).

6) `generator/switching_dynamics.py`
- For model switching / abrupt changes:
  - pre-change model A, post-change model B
  - `t_change_range` or explicit `[t0,t1]`
  - optional “retrain window” metadata (e.g., 50 steps for MAML RVT)
- Matches: MAML RVT abrupt model changes.

7) `generator/datasets/nclt.py` and `generator/datasets/uzh_fpv.py`
- Dataset loaders + conversion to benchmark format (NPZ/NTD).
- Include split rules exactly as in papers (sequence lengths, counts, session ids/dates).

---

## 4) Minimal meta schema (so runners can implement MB/EKF + adaptive/meta-learning fairly)

### 4.1 `meta.json` (suggested)
```jsonc
{
  "task_family": "lorenz|ucm|linear_mismatch|sine_poly|nclt|uzh_fpv|rvt_switch|...",
  "dims": {"x_dim": 3, "y_dim": 3, "T": 2000},
  "ssm": {
    "true": { "type": "...", "params": {...}, "Q": "...", "R": "..." },
    "assumed": { "type": "...", "params": {...}, "Q": "...", "R": "..." }
  },
  "mismatch": {
    "enabled": true,
    "kind": ["F_rotation", "H_rotation", "lorenz_J", "sampling_decimation", "time_axis_shift"],
    "params": {...}
  },
  "noise_schedule": {
    "enabled": true,
    "kind": "split_eq20|per_step_jump|step_change|...",
    "q2_t": "stored-or-derived",
    "r2_t": "stored-or-derived",
    "SoW_t": "stored-or-derived",
    "SoW_hat_t": "optional (for noisy-SoW experiments)"
  },
  "switching": {
    "enabled": false,
    "models": ["A","B"],
    "t_change": [200, 250],
    "retrain_window": 50
  },
  "splits": {
    "train": {"L": 80, "T": 50, "session": "..."},
    "val":   {"L": 5,  "T": 200},
    "test":  {"L": 1,  "T": 2000}
  }
}
```

### 4.2 Why this helps
- Model-based baselines (KF/EKF/IMM) can use `ssm.assumed`.
- Neural methods can train on the same `x,y` but optionally consume:
  - `SoW_t` / `SoW_hat_t` (AKNet)
  - `task_key` (MAML-style: `(q2,r2)` per sequence)
  - `t_change` markers (switching dynamics)

---

## 5) Suggested task IDs (suite-level naming)
Use: `<family>_<variant>_v0` so you can scale without breaking history.

### MVP-friendly subset
- `ucm_linear_v0` (Split/MAML)
- `lorenz_Jmismatch_v0` (KalmanNet/MAML)
- `noise_schedule_splitEq20_v0` (Split)
- `sow_jump_noisy_v0` (AKNet)
- `nclt_segway_v0` (KalmanNet/Split)
- `uzh_fpv_ca_v0` (MAML)
- `rvt_switch_cvca_v0` (MAML)

---

## 6) Implementation checklist for `bench/tasks/generator/*`
When adding a generator, ensure:

1) **Determinism**: same `seed` → identical outputs.
2) **Canonical output tensors**:
   - `x`: shape `[L, T, x_dim]` (ground truth state)
   - `y`: shape `[L, T, y_dim]` (measurement)
3) **Meta completeness**:
   - true vs assumed
   - mismatch params
   - schedules (if time-varying)
4) **Shift support**:
   - implement shift in generator instead of in model code, so all models see same data.
5) **Task-set support** (for MAML/AKNet):
   - store per-sequence “task key” (e.g., `(q2,r2)` or `V_dB`) so the runner can sample episodes.
6) **Paper-aligned splits** (for real datasets):
   - NCLT: session/date + sampling rate + sequence lengths/counts
   - UZH-FPV: 3020 steps; 25×80 train; 1×1020 test

---

## 7) Quick reference (page anchors)
- **KalmanNet 2022**: linear mismatch (PDF p.9–10), sine/poly (p.10), Lorenz variants (p.11–13), NCLT split (p.14)
- **AKNet 2024**: Gaussian/non-Gaussian generalization + time-varying jumps + noisy SoW (PDF p.4)
- **MAML-KalmanNet 2025**: UCM equations (PDF p.9), Lorenz length/model/data mismatch (p.11–12), RVT + 50-step retrain + t=200..250 (p.13–14), UZH-FPV split (p.14)
- **Split-KalmanNet 2023**: UCM equations (PDF p.4), time-varying `R_t` schedule Eq.(20) (p.4), NCLT split/dates (p.5)

---

## 8) Notes (practical)
- **Lorenz “sampling mismatch” vs “data mismatch”**:
  - KalmanNet’s decimation mismatch: dense Δτ then subsample.
  - MAML’s data mismatch: time-axis offset / misalignment between state and measurement sequences.
  - Both should be expressible as generator flags so you can test robustness consistently.

- **AKNet inputs**:
  - If your AKNet implementation expects `SoW_t` (or dB) at each time step, generator should emit it.
  - For “noisy SoW” evaluation, emit `SoW_hat_t` where `SoW_hat_t = SoW_t + ε` or obtained from the same estimator used in baselines.

- **Fairness**:
  - MAML/Adaptive methods are inherently “adaptation-first”. In reports, compare both:
    - Frozen (no adaptation / no fine-tune)
    - Budgeted (fixed max_updates for adaptation)
  - This aligns with your `DECISIONS.md` fairness track concept.

---

*End of note.*
