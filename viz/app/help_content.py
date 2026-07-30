from __future__ import annotations

"""Static help/guide content for the Run Inspector app.

Pure data + string assembly only — no Streamlit import here (the analysis
and content layers must stay renderer-agnostic; `viz/app/components/help_guide.py`
is the only place that renders this with Streamlit). Nothing in this module
reads a run artifact, so importing or evaluating it never scans `runs/` or
loads a trajectory.

`CORE_FACTS` holds the handful of sentences that the in-app guide, the
tooltips, and `docs/VIZ_USER_GUIDE.md` must all agree on. `HELP_TEXT` values
quote them verbatim (via f-strings) instead of restating them in different
words, so the wording cannot drift between the popover and the tooltips.
`tests/test_viz_user_guide.py` additionally checks these same sentences
against the markdown docs.
"""

CORE_FACTS: dict[str, str] = {
    "offline_viewer": (
        "This is an offline artifact viewer. It is not a live sensor, live "
        "inference, or training dashboard."
    ),
    "physical_vs_empirical": (
        "Physical ±3σ comes from a model's own predicted covariance; "
        "empirical spread is the sample spread of errors across trajectories "
        "and is not a predicted covariance."
    ),
    "g1_g2_not_covariance": (
        "G1 and G2 are learned Split-KalmanNet internal factors, not a "
        "physical covariance P or S."
    ),
    "init_id_not_hard_gate": (
        "A run's init_id is a training/initialization label, not a physical "
        "compatibility condition, and does not by itself exclude a run from "
        "comparison."
    ),
    "source_id_fallback": (
        "A Source ID with row-index-fallback provenance is only reliable for "
        "comparing runs built from the same dataset file with the same row "
        "ordering."
    ),
    "nees_nis_unavailable_reason": (
        "KalmanNet and Split-KalmanNet artifacts in this repository do not "
        "provide physical P/S, so NEES/NIS is unavailable for them."
    ),
    "panel_exclusion_is_local": (
        "A model excluded from one panel stays selected everywhere else; "
        "the global toggle in Models to display is not affected."
    ),
    "dataset_summary_is_primary_only": (
        "Dataset Summary is computed from the primary navigation run only "
        "and does not change when you toggle Models to display."
    ),
}

# Tooltip strings for native `help=` parameters. Each is 1-3 sentences,
# describes only the widget it is attached to, and avoids absolute claims
# ("always", "guaranteed", "optimal") — see docs/VIZ_USER_GUIDE.md for the
# full explanation these intentionally do not repeat.
HELP_TEXT: dict[str, str] = {
    "help_button": "Open the Run Inspector user guide.",
    # Navigation
    "data_split": (
        "The evaluation subset this artifact belongs to (e.g. test). Runs "
        "from a different split are never overlaid."
    ),
    "suite": (
        "The experiment collection this run came from. Models to display "
        "draws its candidates from the current suite and evaluation context."
    ),
    "task": "The benchmark task definition for this run.",
    "scenario": "The physical or synthetic test condition used to generate this run.",
    "model": "The model identifier for the primary navigation run (e.g. kalmannet_tsp, oracle_kf).",
    "seed": (
        "The run's seed. It stays part of the evaluation-context filter "
        "because evaluation and training seed are not separated in the "
        "current metadata."
    ),
    "track": (
        "The benchmark execution track (e.g. frozen, budgeted). It stays a "
        "comparison filter because it can change the evaluation protocol, "
        "such as whether the model adapts online."
    ),
    "init_checkpoint": (
        "This run's initialization/training label (e.g. trained, pretrained, "
        f"untrained). {CORE_FACTS['init_id_not_hard_gate']}"
    ),
    "trajectory_view": (
        "Choose one stored representative trajectory to plot. "
        f"{CORE_FACTS['dataset_summary_is_primary_only']}"
    ),
    # Model selection
    "models_to_display": (
        "Controls which run variants appear as traces in the A-F panels "
        f"below. {CORE_FACTS['dataset_summary_is_primary_only']}"
    ),
    "model_checkbox": (
        "Candidate discovered from the selected Source ID; its trajectory "
        "is loaded only once this checkbox is turned on."
    ),
    "model_checkbox_primary": (
        "The primary navigation run. It is on by default; you may turn it "
        "off once another run stays selected. Dataset Summary keeps using "
        "it regardless of this toggle."
    ),
    "provenance": (
        "Initialization/training labels of the selected runs. Differing "
        f"labels do not block overlay. {CORE_FACTS['init_id_not_hard_gate']}"
    ),
    # Display controls
    "axis_mode": (
        "Choose separate per-axis subplots, one combined plot for all axis "
        "components, or a single vector-norm plot."
    ),
    "transient_window": (
        "Exclude an initial fraction of the trajectory from the displayed "
        "metrics and plots. The stored data itself is not modified."
    ),
    "gain_source": (
        "Which gain quantity panel F displays: the combined/standard gain, "
        "or a Split-KalmanNet G1/G2 factor if a selected model provides one. "
        f"{CORE_FACTS['g1_g2_not_covariance']}"
    ),
    "gain_display": (
        "Show gain as a single Frobenius-norm scalar per step, or as one "
        "specific matrix element."
    ),
    "gain_row_col": (
        "Which row/column of the gain matrix to plot when Gain display is "
        "set to Matrix element."
    ),
    # Summary headings
    "dataset_summary": (
        "Aggregate metrics for every trajectory in the selected split, "
        "computed from the primary navigation run only."
    ),
    "selected_trajectory": (
        "Metrics and time-series panels for the one Source ID currently "
        "selected in Trajectory view."
    ),
    # A-F panels
    "attitude_rpy": (
        "Roll, pitch, and yaw from each model's canonical attitude. For "
        "quantitative comparison, use the geodesic attitude error panel instead."
    ),
    "attitude_error": (
        f"Attitude error vs. truth. {CORE_FACTS['physical_vs_empirical']}"
    ),
    "bias": (
        "Gyro bias truth, estimate, and error in deg/h. Shown only for "
        "models with a compatible bias state."
    ),
    "innovation": (
        "Measurement minus predicted measurement. Models are overlaid only "
        "when their measurement and residual definitions match."
    ),
    "nees_nis": (
        "Consistency checks against physical covariance P (NEES) and S "
        f"(NIS). {CORE_FACTS['nees_nis_unavailable_reason']}"
    ),
    "kalman_gain": (
        "Maps innovation into a state correction. Raw gain is overlaid "
        "only when row/column semantics, units, scaling, and shape all match."
    ),
    "regime_timeline": (
        "Stored event/eclipse regime flags on the same time axis as the "
        "panels above. Shown as unavailable if the artifact has no such flags."
    ),
    "advanced_diagnostics": (
        "Dataset-wide RMSE and a per-model compatibility summary for the "
        "selected runs. Reasons already shown under each panel are not "
        "repeated here."
    ),
}


def _panel_section(title: str, help_key: str, extra: str = "") -> str:
    body = HELP_TEXT[help_key]
    return f"**{title}**\n\n{body}" + (f"\n\n{extra}" if extra else "")


QUICK_START_MARKDOWN = f"""\
**{CORE_FACTS['offline_viewer']}**

1. Select the data split and suite.
2. Select task, scenario, seed, track, and the primary run (Model / Init-checkpoint).
3. Select a representative trajectory in **Trajectory view**.
4. Choose which runs to overlay in **Models to display**.
5. Read panels A–F.

Visualization artifacts must be emitted with `--emit-viz-artifacts` \
(opt-in, default off) before they show up here.

See `docs/VIZ_USER_GUIDE.md` (English) or `docs/VIZ_USER_GUIDE_KO.md` \
(Korean) for the full guide, and `docs/VIZ_QUICKSTART.md` for just \
the run/use sequence.
"""

PANELS_MARKDOWN = "\n\n---\n\n".join(
    [
        _panel_section("A. Attitude RPY", "attitude_rpy"),
        _panel_section("B. Attitude Error + 3σ", "attitude_error"),
        _panel_section("C. Bias + 3σ", "bias"),
        _panel_section("D. Innovation", "innovation"),
        _panel_section("E. NEES / NIS", "nees_nis"),
        _panel_section(
            "F. Kalman Gain",
            "kalman_gain",
            CORE_FACTS["g1_g2_not_covariance"],
        ),
        _panel_section("Regime Timeline", "regime_timeline"),
    ]
)

MODEL_COMPARISON_MARKDOWN = f"""\
**Primary model vs. Models to display**

The primary run (chosen by the navigation selectors) is the basis for \
artifact metadata. {CORE_FACTS['dataset_summary_is_primary_only']} Models \
to display only controls what appears in panels A–F.

**Initialization/training provenance**

{CORE_FACTS['init_id_not_hard_gate']} You may see `init=trained`, \
`init=pretrained`, `init=untrained`, or `init=adapted` runs overlaid \
together — this is intentional, e.g. a trained learned model vs. a \
model-based KF, or a trained-vs-untrained ablation.

**Overlay compatibility vs. benchmark fairness**

Overlay compatibility asks whether two runs describe the same evaluation \
data and the same physical/internal quantity — this tool checks that \
for you. Benchmark fairness asks whether training data, oracle information, \
tuning, and adaptation budget were comparable — this tool does not \
judge that, and does not rank models by performance.

**Model capability reference**

| Feature | MB-KF / EKF / MEKF | KalmanNet | Split-KalmanNet |
|---|---|---|---|
| Physical P / S | Usually yes | No | No |
| Physical 3σ | If P available | No | No |
| NEES / NIS | If P/S available | No | No |
| G1 / G2 | No | No | Yes |

Capability is decided by what a specific artifact actually contains, not \
by the model family name.
"""

TROUBLESHOOTING_MARKDOWN = f"""\
**Only one model is shown in Models to display**

Likely causes: only one artifact matches the current suite/task/scenario/\
split/seed/track context, or other runs are missing the selected Source ID. \
Check the "Why only one candidate?" expander for exact counts — this is \
not assumed to mean the metadata parser failed.

**A selected model is missing from one panel**

{CORE_FACTS['panel_exclusion_is_local']} Check the caption under that panel, \
or the Advanced compatibility diagnostics matrix at the bottom of the page.

**G1/G2 is not shown**

The run's artifact does not contain `gain_g1`/`gain_g2`. {CORE_FACTS['g1_g2_not_covariance']}

**NEES/NIS is unavailable**

{CORE_FACTS['nees_nis_unavailable_reason']} This does not mean the model failed.

**Physical 3σ is unavailable**

Either the model has no physical covariance, or the covariance block/space \
metadata needed to interpret it is missing. No band is fabricated from an \
unrelated quantity.

**Legacy artifact cannot use cross-model comparison**

Some artifacts predate `comparison_spec` metadata. Their own single-run \
panels still work; cross-model overlay may be limited.

**Source ID mismatch**

{CORE_FACTS['source_id_fallback']} No fallback substitution is performed.

**Empty artifact root**

Check `VIZ_RUNS_ROOT`, and that the runner was invoked with `--emit-viz-artifacts`.

**Slow initial scan**

A large `runs/` directory takes longer to index once per rerun; only \
selected models' trajectories are loaded afterward.

See `docs/VIZ_TROUBLESHOOTING.md` for the same table and \
`docs/VIZ_KNOWN_LIMITATIONS.md` for the full limitations audit.
"""

GUIDE_TABS: tuple[tuple[str, str], ...] = (
    ("Quick start", QUICK_START_MARKDOWN),
    ("Panels", PANELS_MARKDOWN),
    ("Model comparison", MODEL_COMPARISON_MARKDOWN),
    ("Troubleshooting", TROUBLESHOOTING_MARKDOWN),
)
