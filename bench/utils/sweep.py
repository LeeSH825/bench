from __future__ import annotations

from itertools import product
import re
from typing import Any, Dict, List, Mapping, Optional


def _as_number(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError("bool is not a valid numeric sweep value")
    try:
        return float(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(f"invalid numeric sweep value: {value!r}") from exc


def _maybe_int(value: float, *, tol: float = 1.0e-12) -> Any:
    iv = int(round(float(value)))
    if abs(float(value) - float(iv)) <= tol:
        return iv
    return float(value)


def _values_from_range_spec(spec: Mapping[str, Any]) -> List[Any]:
    if "values" in spec:
        vals = spec.get("values")
        if not isinstance(vals, list):
            raise TypeError("sweep range spec key 'values' must be a list")
        if not vals:
            raise ValueError("sweep values list must not be empty")
        return list(vals)

    if "start" not in spec or "stop" not in spec:
        raise TypeError(
            "range sweep spec must include 'start' and 'stop' (plus 'step' or 'num')"
        )

    start = _as_number(spec.get("start"))
    stop = _as_number(spec.get("stop"))
    has_step = "step" in spec
    has_num = "num" in spec
    if has_step and has_num:
        raise ValueError("range sweep spec must use only one of {'step','num'}")

    if has_num:
        num = int(spec.get("num"))
        if num <= 0:
            raise ValueError(f"range sweep num must be > 0, got {num}")
        if num == 1:
            return [_maybe_int(start)]
        step = (stop - start) / float(num - 1)
        vals = [start + float(i) * step for i in range(num)]
        vals[-1] = stop  # ensure exact endpoint
        return [_maybe_int(v) for v in vals]

    step = _as_number(spec.get("step", 1.0))
    if step == 0.0:
        raise ValueError("range sweep step must be non-zero")
    if step > 0.0 and start > stop:
        raise ValueError(
            f"range sweep with positive step cannot satisfy start={start} > stop={stop}"
        )
    if step < 0.0 and start < stop:
        raise ValueError(
            f"range sweep with negative step cannot satisfy start={start} < stop={stop}"
        )

    vals: List[Any] = []
    cur = float(start)
    tol = abs(step) * 1.0e-9 + 1.0e-12
    if step > 0.0:
        while cur <= float(stop) + tol:
            vals.append(_maybe_int(cur))
            cur += step
    else:
        while cur >= float(stop) - tol:
            vals.append(_maybe_int(cur))
            cur += step
    if not vals:
        raise ValueError("range sweep produced no values")
    return vals


_TUPLE_SWEEP_RE = re.compile(r"^\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*\)$")


def _values_from_tuple_spec(raw: Any) -> List[Any]:
    start_raw: Any
    stop_raw: Any
    steps_raw: Any

    if isinstance(raw, str):
        m = _TUPLE_SWEEP_RE.match(raw.strip())
        if m is None:
            raise TypeError(
                "tuple sweep string must match '(start, stop, steps)'"
            )
        start_raw, stop_raw, steps_raw = m.group(1), m.group(2), m.group(3)
    elif isinstance(raw, (list, tuple)):
        if len(raw) != 3:
            raise TypeError("tuple sweep spec must have exactly 3 values: [start, stop, steps]")
        start_raw, stop_raw, steps_raw = raw[0], raw[1], raw[2]
    else:
        raise TypeError("tuple sweep spec must be a string '(start, stop, steps)' or list/tuple of length 3")

    start = _as_number(start_raw)
    stop = _as_number(stop_raw)
    steps_f = _as_number(steps_raw)
    steps_i = int(round(float(steps_f)))
    if abs(float(steps_f) - float(steps_i)) > 1.0e-9:
        raise ValueError(f"tuple sweep 'steps' must be an integer, got {steps_raw!r}")
    if steps_i <= 0:
        raise ValueError(f"tuple sweep 'steps' must be > 0, got {steps_i}")

    return _values_from_range_spec(
        {
            "start": float(start),
            "stop": float(stop),
            "num": int(steps_i),
        }
    )


def _axis_values(raw: Any) -> List[Any]:
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("(") and text.endswith(")"):
            return _values_from_tuple_spec(text)
        return [raw]
    if isinstance(raw, list):
        if not raw:
            raise ValueError("sweep axis list must not be empty")
        return list(raw)
    if isinstance(raw, Mapping):
        if "tuple" in raw:
            return _values_from_tuple_spec(raw.get("tuple"))
        if "range" in raw:
            nested = raw.get("range")
            if not isinstance(nested, Mapping):
                raise TypeError("sweep key 'range' must map to a dict")
            return _values_from_range_spec(nested)
        return _values_from_range_spec(raw)
    return [raw]


def expand_sweep_grid(
    sweep: Optional[Mapping[str, Any]],
    *,
    sort_keys: bool,
) -> List[Dict[str, Any]]:
    if not sweep:
        return [{}]

    keys = sorted(sweep.keys()) if sort_keys else list(sweep.keys())
    values_list: List[List[Any]] = []
    for k in keys:
        values_list.append(_axis_values(sweep[k]))

    out: List[Dict[str, Any]] = []
    for combo in product(*values_list):
        out.append({k: combo[i] for i, k in enumerate(keys)})
    return out
