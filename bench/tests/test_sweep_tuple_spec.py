from __future__ import annotations

from dataclasses import dataclass

from bench.tasks.bench_generated import expand_scenarios_from_sweep
from bench.utils.sweep import expand_sweep_grid


@dataclass
class SweepTupleSpecResult:
    ok: bool
    note: str


def _extract_vals(grid: list[dict[str, object]], key: str) -> list[float]:
    vals: list[float] = []
    for row in grid:
        raw = row.get(key)
        vals.append(float(raw))  # type: ignore[arg-type]
    return vals


def run_sweep_tuple_spec() -> SweepTupleSpecResult:
    key = "shift.post_shift.R_scale"

    grid_tuple_key = expand_sweep_grid(
        {
            key: {
                "tuple": [3.0, 30.0, 3],
            }
        },
        sort_keys=False,
    )
    vals_tuple_key = _extract_vals(grid_tuple_key, key)
    if vals_tuple_key != [3.0, 16.5, 30.0]:
        return SweepTupleSpecResult(
            ok=False,
            note=f"tuple-key sweep expected [3.0, 16.5, 30.0], got {vals_tuple_key}",
        )

    grid_tuple_str = expand_sweep_grid(
        {key: "(3.0, 30.0, 3)"},
        sort_keys=False,
    )
    vals_tuple_str = _extract_vals(grid_tuple_str, key)
    if vals_tuple_str != [3.0, 16.5, 30.0]:
        return SweepTupleSpecResult(
            ok=False,
            note=f"tuple-string sweep expected [3.0, 16.5, 30.0], got {vals_tuple_str}",
        )

    grid_list = expand_sweep_grid(
        {key: [3.0, 10.0, 30.0]},
        sort_keys=False,
    )
    vals_list = _extract_vals(grid_list, key)
    if vals_list != [3.0, 10.0, 30.0]:
        return SweepTupleSpecResult(
            ok=False,
            note=f"explicit-list sweep must stay unchanged, got {vals_list}",
        )

    shift_task = {
        "task_id": "C_shift_alias_check",
        "noise": {
            "pre_shift": {"R": {"r2": 1.0e-3}},
            "shift": {"t0": 100, "post_shift": {"R_scale": 10.0}},
        },
        "sweep": {"inv_r2_db": {"start": 0, "stop": 10, "step": 10}},
    }
    shift_scens = expand_scenarios_from_sweep(shift_task)
    shift_r2_vals = []
    for s in shift_scens:
        shift_r2_vals.append(float((((s.get("noise") or {}).get("pre_shift") or {}).get("R") or {}).get("r2")))
    if shift_r2_vals != [1.0, 0.1]:
        return SweepTupleSpecResult(
            ok=False,
            note=f"inv_r2_db alias (shift task) expected [1.0, 0.1], got {shift_r2_vals}",
        )

    linear_task = {
        "task_id": "A_linear_alias_check",
        "noise": {"R": {"r2": 1.0e-3}},
        "sweep": {"noise.inv_r2_db": [20]},
    }
    linear_scens = expand_scenarios_from_sweep(linear_task)
    linear_r2_vals = []
    for s in linear_scens:
        linear_r2_vals.append(float((((s.get("noise") or {}).get("R") or {}).get("r2"))))
    if linear_r2_vals != [0.01]:
        return SweepTupleSpecResult(
            ok=False,
            note=f"noise.inv_r2_db alias (linear task) expected [0.01], got {linear_r2_vals}",
        )

    return SweepTupleSpecResult(
        ok=True,
        note=(
            "tuple sweep spec works for mapping/string forms; explicit list remains explicit; "
            "inv_r2_db alias correctly maps to r2 sweeps"
        ),
    )
