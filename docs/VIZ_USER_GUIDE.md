# Run Inspector User Guide

This is the source of truth for the Run Inspector user documentation. `docs/VIZ_QUICKSTART.md` and `docs/VIZ_TROUBLESHOOTING.md` (if present) are short excerpts of this document, not separate content — if something here changes, update it here first. The in-app "Help & guide" popover is a condensed version of this same material.

## 1. Tool overview

**This is an offline artifact viewer. It is not a live sensor, live inference, or training dashboard.**

The flow is:

```
Run model evaluation
  → emit visualization artifacts (opt-in)
  → launch Streamlit
  → select a run and a trajectory
  → inspect and compare results
```

Visualization artifacts are not written automatically. The benchmark runner (`bench/runners/run_suite.py`) only writes them when you pass `--emit-viz-artifacts` (default: off). If you point Run Inspector at a `runs/` directory with no `meta.json` files, it has nothing to show.

## 2. Quick start

1. Select the data split and suite.
2. Select task, scenario, seed, track, and the primary run (Model / Init-checkpoint).
3. Select a representative trajectory (Trajectory view).
4. Choose which runs to overlay in **Models to display**.
5. Read the A–F panels.

Launch command (adjust `VIZ_RUNS_ROOT` to your artifact directory and the Python path to your environment):

```bash
env VIZ_RUNS_ROOT=/path/to/runs \
MPLCONFIGDIR=/tmp/matplotlib \
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
-m streamlit run viz/app/main.py \
--server.headless true \
--server.address 127.0.0.1 \
--server.port 8501
```

`VIZ_RUNS_ROOT` defaults to `runs` (relative to the current working directory) if unset.

## 3. Navigation controls

These selectors, at the top of the page, filter the pool of available runs down to one **primary run**:

| Control | What it filters on |
|---|---|
| Data split | The evaluation subset the artifact belongs to (e.g. `test`). Runs from different splits are never overlaid. |
| Suite | The experiment collection the run was produced by. |
| Task | The benchmark task definition. |
| Scenario | The physical or synthetic test condition used to generate the run. |
| Model | The model identifier (e.g. `kalmannet_tsp`, `oracle_kf`). |
| Seed | The run's seed value (see §4 for what this seed actually controls). |
| Track | The benchmark execution track (e.g. `frozen`, `budgeted` — whether the model adapts online during evaluation). |
| Init/checkpoint | The initialization/training label of this run (e.g. `trained`, `pretrained`, `untrained`). |

**The selected Model is the primary navigation run.** It stays the basis of Dataset Summary and the artifact metadata badges even if you later hide it from the plots using the "Models to display" toggles (§5).

## 4. Primary model vs. Models to display

These two things look similar but control different parts of the screen. Confusing them is the single most common source of "why isn't X updating" questions.

**Primary model** (chosen by the navigation controls in §3):
- Is the basis for Dataset Summary and the artifact metadata badges (commit, run status, artifact version, ...).
- Can still be turned OFF in the A–F plots via its own checkbox in Models to display.

**Models to display** (§5, right below Trajectory view):
- Controls which run variants actually appear as traces in the A–F panels.
- Only variants toggled ON are loaded and plotted. Nothing else is.
- At least one run must stay selected at all times.

### Initialization/training labels are provenance, not a hard filter

A run's `init_id` (shown as `init=...` in the toggle labels) is a training/initialization **label**, not a physical compatibility condition. It does not, by itself, exclude a run from being a candidate or from being overlaid. For example, you may see all of these together:

```
oracle_kf · init=pretrained
kalmannet_tsp · init=trained
split_knet · init=trained
```

If the runs share the same evaluation context (suite/task/scenario/split/seed/track) and the panel-specific compatibility check passes, they can be overlaid regardless of `init_id`. When the selected runs' `init_id` values differ, the app shows a non-blocking notice reminding you to interpret the comparison as a baseline/ablation/adaptation comparison rather than as identical training conditions — it never blocks the overlay by itself.

## 5. Dataset Summary vs. Selected Trajectory

- **Dataset Summary**: aggregate metrics computed over the *entire* evaluation split, always based on the **primary navigation run**. It does not change when you toggle Models to display or switch the Trajectory view — that is intentional, not a bug.
- **Selected Trajectory**: metrics and time-series panels for the one Source ID you picked in "Trajectory view". This is what the A–F panels actually plot.

## 6. Source ID and stored index

- **Stored index**: the position of a trajectory inside the artifact's own storage (an internal offset).
- **Source ID**: the identity of that trajectory within the evaluation dataset.

If a run's Source ID provenance is `test_split_row_index_fallback`, the ID is only reliable for comparing against another run built from **the same dataset file with the same row ordering** — it is not a globally stable trajectory identity across arbitrary artifacts. Run Inspector never substitutes a different stored index to work around a missing Source ID; if a candidate run doesn't have the selected Source ID, it is simply not usable for that trajectory.

## 7. Models to display

- Candidates are limited to runs matching the current suite/task/scenario/split/seed/track — the same evaluation context as the primary run.
- Multiple init/training variants of the same model (e.g. `kalmannet_tsp · init=untrained`, `kalmannet_tsp · init=trained`, `kalmannet_tsp · init=adapted`) can appear as separate, independently toggled candidates.
- A run toggled OFF is not loaded and has no trace in any panel.
- If a selected run is incompatible with one specific panel (e.g. its innovation residual definition doesn't match), it is excluded from **that panel only** — the global toggle stays ON and the run keeps appearing in every other compatible panel.
- Initialization/training provenance differences are an interpretation note, not an automatic exclusion reason (§4).

## 8. Axis and display controls

- **3-axis split**: each axis (x/y/z) gets its own subplot.
- **Combined axes**: all axis components are drawn on one shared plot. This does *not* mean "combine multiple models" — it only changes how a single trace's x/y/z components are laid out.
- **Norm only**: plots the vector norm instead of per-axis components.
- **Transient window**: excludes an initial fraction of the trajectory from the displayed metrics/plots. The underlying stored data is never modified.
- **Gain source**: which gain quantity to display in panel F (combined gain, or a Split-KalmanNet G1/G2 factor if a selected model provides one).
- **Gain display**: Frobenius norm (a single scalar per step) or a specific matrix element.
- **Matrix element row/col**: only shown in "Matrix element" mode; picks which entry of the gain matrix to plot.

## 9. Reading the A–F panels

### A. Attitude RPY

- **What it is**: Roll, Pitch, and Yaw derived from each model's canonical attitude representation, alongside Truth.
- **Why it matters**: gives an intuitive, human-readable view of attitude tracking over time.
- **How to use it**: pick models in "Models to display"; Truth is always drawn in a neutral style.
- **Watch out**: RPY is for intuition only. For quantitative comparison, use the geodesic attitude error in panel B. A model is only overlaid here if its frame and RPY/quaternion convention are declared compatible with the primary's.

### B. Attitude Error + 3σ

- **What it is**: the actual attitude error (estimate vs. truth), expressed as a geodesic (or axis) error.
- **Why it matters**: this is the quantitative attitude accuracy metric, unlike the RPY panel.
- **How to use it**: compare error magnitude/behavior across selected, compatible models.
- **Watch out**: a **physical ±3σ band** is only drawn for a model that provides a valid physical covariance (`P`). A learned model without physical `P` never gets a physical 3σ band here — that is expected, not a bug. MRP and rotation-vector covariance use different conversion factors, so the band math differs by declared covariance space.
- **Physical ±3σ vs. empirical spread — these are not the same thing**:
  - *Physical ±3σ*: derived from the model's own predicted covariance.
  - *Empirical spread*: the sample spread of observed errors across multiple trajectories. It is not a covariance predicted by the filter, and is shown with a visually distinct (unfilled/dotted) style.

### C. Bias + 3σ

- **What it is**: gyro bias truth, estimate, and error, in deg/h.
- **Why it matters**: bias tracking is a separate failure mode from attitude tracking.
- **How to use it**: only models whose artifact declares a compatible bias state are shown here.
- **Watch out**: this panel can be disabled entirely for a model that has no bias state at all (e.g. a pure-attitude filter); a physical 3σ band additionally requires a physical bias covariance block.

### D. Innovation

- **What it is**: `innovation = measurement − predicted measurement`.
- **Why it matters**: large, structured innovation usually means the model's prediction disagrees with what was actually measured.
- **How to use it**: compare selected models' innovation magnitude and channel behavior over time.
- **Watch out**: models are only overlaid here if they share the same measurement type, residual definition, frame, units, and channel order — a gyro residual is never compared directly against an attitude-reference residual. Where `innov_valid=false` (no measurement update at that step), the value is *not shown* rather than plotted as zero.

### E. NEES / NIS

- **What it is**: **NEES** (Normalized Estimation Error Squared) checks the consistency between the state estimation error and the model's predicted state covariance `P`. **NIS** (Normalized Innovation Squared) checks the consistency between the innovation and the model's predicted innovation covariance `S`.
- **Why it matters**: these are standard filter-consistency diagnostics.
- **How to use it**: only meaningful for models that provide physical `P`/`S`.
- **Watch out**: KalmanNet and Split-KalmanNet artifacts in this repository do not provide physical `P`/`S`, so NEES/NIS is **unavailable** for them — this reflects what the artifact contains, not a model failure. χ² bounds indicate statistical consistency, not a pass/fail judgment of model quality, and can vary with a small sample count.

### F. Kalman Gain

- **What it is**: the matrix that maps innovation into a state correction. Different formulations exist: model-based gain, learned gain, and Split-KalmanNet's combined gain.
- **Why it matters**: shows how much correction weight the filter places on new measurements.
- **How to use it**: choose Frobenius norm (one scalar per step) or a specific matrix element via the display controls (§8).
- **Watch out**: raw gain is only overlaid across models when state row semantics, measurement column semantics, units, scaling, and shape all match (a "strict" compatibility check) — this is stricter than the physical-quantity checks used in panels A–C.

### Split G1/G2

`Combined gain = G1 @ H.T @ G2`. G1 and G2 are **learned internal factors specific to Split-KalmanNet** — they are:
- **not** a physical state covariance `P`,
- **not** a physical innovation covariance `S` (or its inverse),
- **not** guaranteed to be symmetric or positive semi-definite the way a real covariance matrix must be,
- **never** used by this tool to compute NEES/NIS.

A model that doesn't provide `gain_g1`/`gain_g2` is excluded from a G1/G2-selected gain view, with the reason shown in the panel caption — the global toggle for that model is not affected.

### Regime Timeline

Shows stored event/eclipse (or similar regime) flags on the same time axis as the panels above. If the artifact has no such flags, it is shown as unavailable — this is different from "no event occurred" (which the timeline would show as an empty/flat strip while still being available).

## 10. Model capability reference

| Feature | MB-KF / EKF / MEKF | KalmanNet | Split-KalmanNet |
|---|---|---|---|
| State estimate | Yes | Yes | Yes |
| Innovation | Yes | Yes | Yes |
| Standard/combined gain | Yes | Yes | Yes |
| G1/G2 | No | No | Yes |
| Physical P | Usually yes | No | No |
| Physical S | Usually yes | No | No |
| Physical 3σ | If P is available | No | No |
| NEES/NIS | If P/S are available | No | No |
| Empirical uncertainty | If multiple trajectories | If multiple trajectories | If multiple trajectories |

"Usually" and "if available" are deliberate — capability is determined by what a specific artifact actually contains, not by which model family it belongs to. Always trust the panel's own "not shown / unavailable" caption over an assumption based on the model name.

## 11. Comparing models: what's fair to overlay?

**Physical quantities you can generally compare** (subject to the panel-specific checks in §9): RPY, attitude error, bias, and physical uncertainty (when both models provide it).

**Strict internal quantities** (compared only under tighter checks): innovation, raw gain, NEES/NIS, G1/G2.

**Initialization/training provenance is a label, not a physical compatibility condition** (§4). Examples of comparisons this tool is meant to support:
- a trained learned model vs. a model-based KF,
- a trained vs. an untrained ablation of the same model,
- a before-adaptation vs. after-adaptation run.

**Overlay compatibility and benchmark fairness are two different questions.** Overlay compatibility asks "do these runs describe the same evaluation data and the same physical/internal quantity?" Benchmark fairness asks "were the training data, oracle information, tuning effort, and adaptation budget comparable?" This tool answers the first question for you (via the compatibility guards) and deliberately does **not** answer the second — it will let you overlay a heavily-tuned model against a lightly-tuned one if they're evaluated on the same data, and it does not judge which one is "better."

## 12. Troubleshooting

### Only one model is shown in Models to display
Likely causes: only one artifact currently exists for this suite/task/scenario/split/seed/track context; other runs don't store the selected Source ID; other runs belong to a different evaluation context. This is **not** assumed to mean the metadata parser failed — check the "Why only one candidate?" expander for the exact counts.

### A selected model is missing from one panel
The global toggle is still ON. The model was excluded from that **one panel only**, usually because of a panel-specific semantics/capability mismatch. Check the caption directly under that panel, or the "Advanced compatibility diagnostics" matrix at the bottom of the page.

### G1/G2 is not shown
The selected run's artifact does not contain `gain_g1`/`gain_g2`. Only Split-KalmanNet-style runs that declare these components can show them; standard/combined gain is a different source and is unaffected.

### NEES/NIS is unavailable
The model's artifact does not provide physical `P`/`S`. This is the normal, expected state for KalmanNet/Split-KalmanNet in this repository — do not confuse it with empirical uncertainty, which is a different quantity entirely.

### Physical 3σ is unavailable
Either the model has no physical covariance, or the covariance block/space metadata needed to interpret it is missing. Run Inspector never fabricates a band from an unrelated quantity.

### Legacy artifact cannot use cross-model comparison
Some artifacts predate `comparison_spec` metadata. Their own single-run A–F panels still work; cross-model overlay may be limited or unavailable for that run.

### Source ID mismatch
The candidate run does not store the same trajectory. Run Inspector never substitutes a different stored index to paper over this.

### Empty artifact root
Check that `VIZ_RUNS_ROOT` points at a directory containing `meta.json` files, and that the runner was actually invoked with `--emit-viz-artifacts` (§1).

### Slow initial scan
A `runs/` directory with many artifacts takes longer to index on first load. This scan happens exactly once per rerun (not once per panel); only the trajectories of currently-selected models are loaded afterward.

## 13. Known limitations

- Offline viewer only — see §1.
- Real-artifact validation coverage for some newer features (e.g. cross-model overlay on production-scale ADCS artifacts) remains limited.
- Source ID can be a row-index fallback rather than a stable dataset-native identity (§6).
- Physical covariance-based NEES/NIS is unavailable for learned filters that don't provide `P`/`S`.
- G1/G2 are learned internal factors, not a covariance (§9).
- Time-varying `H` and non-square Split-KalmanNet checkpoint coverage have not been fully audited.
- GPU/production-scale ADCS performance envelopes have not been fully characterized.
- `reports/` is excluded by `.gitignore` in this repository, so generated reports may not show up in a plain `git status`.

See `docs/VIZ_KNOWN_LIMITATIONS.md` for the full, evidence-tagged limitations audit.
