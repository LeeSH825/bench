from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

from viz.contract import (
    ContractError,
    normalize_meta,
    require_overlay_compatible,
    sorted_run_key,
    validate_meta,
    validate_traj_arrays,
    validate_trajectory_capabilities,
)


_TRAJ_FILE_RE = re.compile(r"^traj_(\d+)\.npz$")


@dataclass(frozen=True)
class TrajectoryInfo:
    stored_index: int
    source_trajectory_id: Any | None
    source_trajectory_id_source: str
    file: str
    length_T: int | None
    time_start: float | None
    time_end: float | None
    has_event: bool | None
    has_eclipse: bool | None
    run_status: str
    legacy: bool = False


@dataclass(frozen=True)
class VizRun:
    run_dir: Path
    meta: Dict[str, Any]
    aggregate: Optional[Dict[str, np.ndarray]]
    metrics: Optional[Dict[str, Any]]
    trajectories: tuple[TrajectoryInfo, ...]

    def list_trajectories(self) -> list[TrajectoryInfo]:
        return list(self.trajectories)

    def trajectory_by_stored_index(self, stored_index: int) -> TrajectoryInfo:
        target = int(stored_index)
        for info in self.trajectories:
            if info.stored_index == target:
                return info
        available = [item.stored_index for item in self.trajectories]
        raise KeyError(f"stored trajectory index {target} is unavailable; available={available}")

    def trajectory_by_source_id(self, source_trajectory_id: Any) -> TrajectoryInfo:
        matches = [
            info
            for info in self.trajectories
            if info.source_trajectory_id is not None
            and type(info.source_trajectory_id) is type(source_trajectory_id)
            and info.source_trajectory_id == source_trajectory_id
        ]
        if not matches:
            available = [item.source_trajectory_id for item in self.trajectories if item.source_trajectory_id is not None]
            raise KeyError(
                f"source trajectory ID {source_trajectory_id!r} is unavailable; available={available}"
            )
        if len(matches) != 1:
            raise ContractError(f"duplicate source trajectory ID: {source_trajectory_id!r}")
        return matches[0]

    def load_trajectory(
        self,
        *,
        stored_index: int | None = None,
        source_trajectory_id: Any | None = None,
    ) -> Dict[str, np.ndarray]:
        if (stored_index is None) == (source_trajectory_id is None):
            raise ValueError("provide exactly one of stored_index or source_trajectory_id")
        info = (
            self.trajectory_by_stored_index(int(stored_index))
            if stored_index is not None
            else self.trajectory_by_source_id(source_trajectory_id)
        )
        return _load_trajectory_info(self, info)


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.array(data[key], copy=True) for key in data.files}


def _load_json_object(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _manifest_trajectories(root: Path, meta: Mapping[str, Any]) -> tuple[TrajectoryInfo, ...]:
    manifest = meta.get("trajectories")
    if not isinstance(manifest, list):
        raise ContractError("meta.trajectories must be a list")
    infos: list[TrajectoryInfo] = []
    expected_paths: set[Path] = set()
    for item in manifest:
        try:
            relative_file = str(item["file"])
            stored_index = int(item["stored_index"])
            source_trajectory_id = item["source_trajectory_id"]
            length_t = int(item["length_T"])
            time_start = float(item["time_start"])
            time_end = float(item["time_end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"invalid trajectory manifest entry: {item!r}") from exc
        path = (root / relative_file).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ContractError(f"trajectory manifest path escapes run directory: {relative_file}") from exc
        if not path.exists():
            raise FileNotFoundError(
                f"trajectory manifest entry is missing its NPZ file: stored_index={item['stored_index']} path={path}"
            )
        expected_paths.add(path)
        infos.append(
            TrajectoryInfo(
                stored_index=stored_index,
                source_trajectory_id=source_trajectory_id,
                source_trajectory_id_source=str(item.get("source_trajectory_id_source") or "provided_trajectory_id"),
                file=relative_file,
                length_T=length_t,
                time_start=time_start,
                time_end=time_end,
                has_event=item.get("has_event"),
                has_eclipse=item.get("has_eclipse"),
                run_status=str(item.get("run_status") or meta.get("run_status") or "unknown"),
                legacy=False,
            )
        )
    actual_paths = {path.resolve() for path in (root / "series").glob("traj_*.npz")}
    if actual_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - actual_paths)
        extra = sorted(str(path) for path in actual_paths - expected_paths)
        raise ContractError(f"trajectory manifest/file mismatch: missing={missing}, extra={extra}")
    return tuple(sorted(infos, key=lambda item: item.stored_index))


def _legacy_trajectories(root: Path, meta: Mapping[str, Any]) -> tuple[TrajectoryInfo, ...]:
    infos: list[TrajectoryInfo] = []
    length = int(meta.get("T", 0) or 0) or None
    dt = meta.get("dt")
    try:
        dt_value = float(dt) if dt is not None else None
    except (TypeError, ValueError):
        dt_value = None
    time_start = 0.0 if length else None
    time_end = float((length - 1) * dt_value) if length and dt_value is not None else None
    for path in sorted((root / "series").glob("traj_*.npz")):
        match = _TRAJ_FILE_RE.match(path.name)
        if match is None:
            continue
        infos.append(
            TrajectoryInfo(
                stored_index=int(match.group(1)),
                source_trajectory_id=None,
                source_trajectory_id_source="legacy_unknown",
                file=f"series/{path.name}",
                length_T=length,
                time_start=time_start,
                time_end=time_end,
                has_event=None,
                has_eclipse=None,
                run_status=str(meta.get("run_status") or "unknown"),
                legacy=True,
            )
        )
    return tuple(infos)


def load_run(
    run_dir: str | Path,
    *,
    load_aggregate: bool = True,
    load_metrics: bool = True,
) -> VizRun:
    root = Path(run_dir).expanduser().resolve()
    meta_path = root / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"visualization meta not found: {meta_path}")
    raw_meta = _load_json_object(meta_path)
    if raw_meta is None:
        raise FileNotFoundError(f"visualization meta not found: {meta_path}")
    had_manifest = isinstance(raw_meta.get("trajectories"), list)
    meta = normalize_meta(raw_meta)
    validate_meta(meta)
    trajectories = _manifest_trajectories(root, meta) if had_manifest else _legacy_trajectories(root, meta)
    if not had_manifest:
        data_spec = dict(meta["data_spec"])
        data_spec["num_stored_trajectories"] = len(trajectories)
        data_spec["num_trajectories"] = max(
            int(data_spec.get("num_trajectories", 0) or 0),
            len(trajectories),
        )
        meta["data_spec"] = data_spec
    aggregate = None
    aggregate_path = root / "series" / "aggregate.npz"
    if load_aggregate and aggregate_path.exists():
        aggregate = _load_npz(aggregate_path)
    metrics = _load_json_object(root / "metrics.json") if load_metrics else None
    return VizRun(
        run_dir=root,
        meta=meta,
        aggregate=aggregate,
        metrics=metrics,
        trajectories=trajectories,
    )


def load_runs(run_dirs: Iterable[str | Path]) -> List[VizRun]:
    runs = [load_run(path) for path in run_dirs]
    return sorted(runs, key=lambda run: sorted_run_key(run.meta, run.run_dir))


def _load_trajectory_info(run: VizRun, info: TrajectoryInfo) -> Dict[str, np.ndarray]:
    path = (run.run_dir / info.file).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"trajectory artifact not found: stored_index={info.stored_index} "
            f"source_trajectory_id={info.source_trajectory_id!r} path={path}"
        )
    arrays = _load_npz(path)
    validate_traj_arrays(arrays)
    validate_trajectory_capabilities(run.meta, arrays)
    if info.length_T is not None and "t" in arrays and int(arrays["t"].shape[0]) != int(info.length_T):
        raise ContractError(
            f"trajectory length mismatch for stored_index={info.stored_index}: "
            f"manifest={info.length_T}, npz={arrays['t'].shape[0]}"
        )
    return arrays


def load_trajectory(
    run_dir: str | Path,
    traj_idx: int | None = None,
    *,
    stored_index: int | None = None,
    source_trajectory_id: Any | None = None,
) -> Dict[str, np.ndarray]:
    if traj_idx is not None:
        if stored_index is not None or source_trajectory_id is not None:
            raise ValueError("positional traj_idx cannot be combined with stored/source selectors")
        stored_index = int(traj_idx)
    run = load_run(run_dir, load_aggregate=False, load_metrics=False)
    return run.load_trajectory(stored_index=stored_index, source_trajectory_id=source_trajectory_id)


def assert_overlay_compatible(
    base: VizRun | Mapping[str, Any],
    overlay: VizRun | Mapping[str, Any],
    *,
    source_trajectory_id: Any | None = None,
) -> None:
    base_meta = base.meta if isinstance(base, VizRun) else base
    overlay_meta = overlay.meta if isinstance(overlay, VizRun) else overlay
    require_overlay_compatible(base_meta, overlay_meta)
    if source_trajectory_id is None:
        return
    if not isinstance(base, VizRun) or not isinstance(overlay, VizRun):
        raise ValueError("source trajectory overlay validation requires loaded VizRun objects")
    try:
        base_info = base.trajectory_by_source_id(source_trajectory_id)
    except KeyError as exc:
        raise ContractError(
            f"overlay unavailable for Source ID {source_trajectory_id!r}: base run did not store this trajectory"
        ) from exc
    try:
        overlay_info = overlay.trajectory_by_source_id(source_trajectory_id)
    except KeyError as exc:
        raise ContractError(
            f"overlay unavailable for Source ID {source_trajectory_id!r}: "
            "the selected run did not store this trajectory"
        ) from exc
    base_t = _load_trajectory_info(base, base_info).get("t")
    overlay_t = _load_trajectory_info(overlay, overlay_info).get("t")
    if base_t is None or overlay_t is None or not np.array_equal(base_t, overlay_t):
        raise ContractError(
            f"overlay blocked: time axis mismatch for Source ID {source_trajectory_id!r}; interpolation is not allowed"
        )
