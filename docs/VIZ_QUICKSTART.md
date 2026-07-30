# Run Inspector Quick Start

Full details: `docs/VIZ_USER_GUIDE.md`. This file only summarizes the run/use sequence.

## Launch

```bash
env VIZ_RUNS_ROOT=/path/to/runs \
MPLCONFIGDIR=/tmp/matplotlib \
/home/dss-pc-05/.pyenv/versions/3.10.13/bin/python \
-m streamlit run viz/app/main.py \
--server.headless true \
--server.address 127.0.0.1 \
--server.port 8501
```

`VIZ_RUNS_ROOT` defaults to `runs` if unset. Artifacts under it must come from a runner invoked with `--emit-viz-artifacts` (opt-in, default off).

## Use

1. Select the data split and suite.
2. Select task, scenario, seed, track, and the primary run (Model / Init-checkpoint).
3. Select a representative trajectory in "Trajectory view".
4. Choose which runs to overlay in "Models to display" — the primary run is ON by default and can be turned off once another run is selected.
5. Read panels A–F. If a model you expected is missing from one panel, check that panel's caption first (see `docs/VIZ_USER_GUIDE.md` §12 for common cases).

This is an offline artifact viewer, not a live dashboard.
