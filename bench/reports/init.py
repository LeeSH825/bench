"""
bench.reports package

Report generation from run_dir artifacts.
- Scans runs/ for metrics.json / metrics_step.csv / failure.json (+ run_plan/budget/timing if present)
- Produces reports/*.csv and plots/*.png (baseline + optional S6 plan/ops/budget views)
"""
from .make_report import main as make_report_main  # noqa: F401
