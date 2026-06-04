#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import audit_basilisk_imu_pilot as audit


ROOT = Path(__file__).resolve().parents[1]

audit.RUN_ROOT = ROOT / "runs" / "gpu_basilisk_imu_pilot_pretrained_enhancer"
audit.METRICS_CSV = ROOT / "reports" / "basilisk_imu_pretrained_enhancer_gpu_pilot_metrics.csv"
audit.SUMMARY_CSV = ROOT / "reports" / "basilisk_imu_pretrained_enhancer_gpu_pilot_summary.csv"
audit.AUDIT_JSON = ROOT / "reports" / "basilisk_imu_pretrained_enhancer_gpu_pilot_acceptance_audit.json"
audit.PLOT_MSE = ROOT / "plots" / "basilisk_imu_pretrained_enhancer_mse_db.png"
audit.PLOT_IMPROVEMENT = ROOT / "plots" / "basilisk_imu_pretrained_enhancer_improvement_db.png"
audit.PLOT_DELTA = ROOT / "plots" / "basilisk_imu_pretrained_enhancer_delta_ratio.png"
audit.PLOT_IMU_REDUCTION = ROOT / "plots" / "basilisk_imu_pretrained_enhancer_imu_mse_reduction.png"
audit.PLOT_ALIGNMENT = ROOT / "plots" / "basilisk_imu_pretrained_enhancer_alignment.png"
audit.REQUIRE_IMU_ENHANCER_DIAG = True


if __name__ == "__main__":
    raise SystemExit(audit.main())
