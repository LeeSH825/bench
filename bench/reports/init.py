"""
bench.reports package

Step 7: Report generation from run_dir artifacts.
- Scans runs/ for metrics.json / metrics_step.csv / failure.json
- Produces reports/*.csv and plots/*.png
"""
from .make_report import main as make_report_main  # noqa: F401
