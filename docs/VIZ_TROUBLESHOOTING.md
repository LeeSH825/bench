# Run Inspector Troubleshooting

Full details and reasoning: `docs/VIZ_USER_GUIDE.md` §12. This file only lists symptoms and short fixes.

| Symptom | Likely cause | What to check |
|---|---|---|
| Only one model in Models to display | Only one artifact matches the current suite/task/scenario/split/seed/track context, or others are missing the selected Source ID | "Why only one candidate?" expander |
| A model is missing from one panel only | Panel-specific semantics/capability mismatch (global toggle is still ON) | Caption under that panel; "Advanced compatibility diagnostics" matrix |
| G1/G2 not shown | Run has no `gain_g1`/`gain_g2` | Whether the run is a Split-KalmanNet-style artifact |
| NEES/NIS unavailable | No physical `P`/`S` in the artifact | Normal for KalmanNet/Split-KalmanNet in this repository |
| Physical 3σ unavailable | No physical covariance, or missing covariance block/space metadata | Model capability reference in the user guide |
| Cross-model comparison limited on a legacy artifact | Artifact predates `comparison_spec` metadata | Single-run A–F panels still work |
| Source ID mismatch between runs | Candidate does not store the same trajectory | No fallback substitution is performed, by design |
| Empty artifact root / "No valid visualization runs found" | `VIZ_RUNS_ROOT` wrong, or runner was not run with `--emit-viz-artifacts` | Directory contains `meta.json` files |
| Slow initial page load | Large `runs/` directory being indexed | One-time per rerun; trajectories still load lazily per selected run |

None of the above are worked around by silently substituting a different trajectory, run, or artifact — Run Inspector always reports the mismatch instead.
