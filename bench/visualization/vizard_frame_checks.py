from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ZERO_ATTITUDE_FILENAME = "zero_attitude_dataFileToViz_input.csv"
POSITIVE_YAW_FILENAME = "small_positive_yaw_dataFileToViz_input.csv"
TRUE_ESTIMATED_OFFSET_FILENAME = (
    "true_vs_estimated_offset_dataFileToViz_input.csv"
)
FRAME_CHECK_README_FILENAME = "README_frame_convention_check.md"

_COLUMNS = (
    "time_s",
    "sc_name",
    "r_BN_N_x_m",
    "r_BN_N_y_m",
    "r_BN_N_z_m",
    "v_BN_N_x_m_s",
    "v_BN_N_y_m_s",
    "v_BN_N_z_m_s",
    "attitude_type",
    "sigma_BN_1",
    "sigma_BN_2",
    "sigma_BN_3",
    "omega_BN_B_x_rad_s",
    "omega_BN_B_y_rad_s",
    "omega_BN_B_z_rad_s",
)


def _frame(
    time_s: np.ndarray,
    *,
    names: list[str],
    sigma_by_name: dict[str, tuple[float, float, float]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for time_value in time_s:
        for index, name in enumerate(names):
            sigma = sigma_by_name[name]
            rows.append(
                {
                    "time_s": float(time_value),
                    "sc_name": name,
                    "r_BN_N_x_m": 7000.0e3,
                    "r_BN_N_y_m": float(20.0 * index),
                    "r_BN_N_z_m": 0.0,
                    "v_BN_N_x_m_s": 0.0,
                    "v_BN_N_y_m_s": 0.0,
                    "v_BN_N_z_m_s": 0.0,
                    "attitude_type": "MRP",
                    "sigma_BN_1": sigma[0],
                    "sigma_BN_2": sigma[1],
                    "sigma_BN_3": sigma[2],
                    "omega_BN_B_x_rad_s": 0.0,
                    "omega_BN_B_y_rad_s": 0.0,
                    "omega_BN_B_z_rad_s": 0.0,
                }
            )
    return pd.DataFrame(rows, columns=_COLUMNS)


def _readme() -> str:
    return """# Vizard Frame Convention Check

These deterministic fixtures validate the benchmark visualization convention
manually in Vizard. They do not affect official benchmark metrics.

## Fixtures

1. `zero_attitude_dataFileToViz_input.csv`
   - `SC_zero` uses `sigma_BN = [0, 0, 0]`.
   - Expected: body and inertial frame axes align.
2. `small_positive_yaw_dataFileToViz_input.csv`
   - `SC_yaw_pos` uses an MRP for a positive 10 degree z-axis rotation.
   - Expected: the displayed yaw direction matches the benchmark `sigma_BN`
     convention.
3. `true_vs_estimated_offset_dataFileToViz_input.csv`
   - `SC_true` is identity and `SC_estimated` has the same known yaw offset.
   - Expected: the two spacecraft have visibly different orientations.

If yaw appears inverted, investigate `sigma_BN` versus `sigma_NB`. If true and
estimated orientations appear swapped, investigate spacecraft/source mapping.
The positions are fixed visualization-only offsets and are not scenario truth.
Live streaming and Vizard launch remain outside this fixture generator.
"""


def generate_frame_check_fixtures(
    out_dir: str | Path,
    *,
    duration_s: float = 4.0,
    dt_s: float = 1.0,
) -> tuple[list[Path], Path]:
    if not np.isfinite(duration_s) or duration_s < 0.0:
        raise ValueError(f"duration_s must be finite and non-negative, got {duration_s}")
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError(f"dt_s must be finite and positive, got {dt_s}")

    count = int(np.floor(duration_s / dt_s + 1.0e-12)) + 1
    time_s = np.arange(count, dtype=np.float64) * float(dt_s)
    yaw_sigma = float(np.tan(np.deg2rad(10.0) / 4.0))
    fixtures = {
        ZERO_ATTITUDE_FILENAME: _frame(
            time_s,
            names=["SC_zero"],
            sigma_by_name={"SC_zero": (0.0, 0.0, 0.0)},
        ),
        POSITIVE_YAW_FILENAME: _frame(
            time_s,
            names=["SC_yaw_pos"],
            sigma_by_name={"SC_yaw_pos": (0.0, 0.0, yaw_sigma)},
        ),
        TRUE_ESTIMATED_OFFSET_FILENAME: _frame(
            time_s,
            names=["SC_true", "SC_estimated"],
            sigma_by_name={
                "SC_true": (0.0, 0.0, 0.0),
                "SC_estimated": (0.0, 0.0, yaw_sigma),
            },
        ),
    }

    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    readme_path = output_dir / FRAME_CHECK_README_FILENAME
    with tempfile.TemporaryDirectory(
        prefix=".frame_checks_",
        dir=output_dir,
    ) as tmp:
        staging_dir = Path(tmp)
        for filename, frame in fixtures.items():
            frame.to_csv(staging_dir / filename, index=False)
        (staging_dir / FRAME_CHECK_README_FILENAME).write_text(
            _readme(),
            encoding="utf-8",
        )
        for filename in fixtures:
            (staging_dir / filename).replace(output_dir / filename)
        (staging_dir / FRAME_CHECK_README_FILENAME).replace(readme_path)

    return [output_dir / filename for filename in fixtures], readme_path
