from __future__ import annotations

"""Pure state helpers for the Run Inspector's suite-wide model picker."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ModelToggleCandidate:
    run_dir: str
    model_id: str
    label: str
    metadata: Mapping[str, Any]


CONTEXT_FIELDS = ("suite", "task", "scenario_id", "split", "seed", "track_id", "init_id")


def model_context_key(meta: Mapping[str, Any]) -> tuple[Any, ...]:
    data_spec = meta.get("data_spec") if isinstance(meta.get("data_spec"), Mapping) else {}
    return tuple(meta.get(field, data_spec.get(field)) for field in CONTEXT_FIELDS)


def candidate_label(meta: Mapping[str, Any]) -> str:
    model = str(meta.get("model_id") or "unknown model")
    track = str(meta.get("track_id") or "unknown track")
    seed = str(meta.get("seed") if meta.get("seed") is not None else "unknown")
    return f"{model} — {track} / seed {seed}"


def suite_candidates(
    primary: Any,
    indexed_runs: Sequence[Any],
    *,
    source_trajectory_id: Any,
) -> list[ModelToggleCandidate]:
    """Discover manifest candidates only; this function never loads an NPZ."""
    context = model_context_key(primary.meta)
    runs = [primary, *indexed_runs]
    result: dict[str, ModelToggleCandidate] = {}
    for run in runs:
        if model_context_key(run.meta) != context:
            continue
        try:
            run.trajectory_by_source_id(source_trajectory_id)
        except KeyError:
            continue
        run_dir = str(run.run_dir)
        result[run_dir] = ModelToggleCandidate(
            run_dir=run_dir,
            model_id=str(run.meta.get("model_id") or "unknown model"),
            label=candidate_label(run.meta),
            metadata=run.meta,
        )
    return sorted(result.values(), key=lambda item: (item.model_id, item.run_dir))


def reconcile_selection(
    candidates: Sequence[ModelToggleCandidate],
    *,
    primary_run_dir: str,
    previous: Sequence[str] | None,
) -> set[str]:
    """Drop stale context entries; default to the primary alone when nothing valid remains.

    The primary is only force-selected as a *default* (first load, or a
    suite/task/scenario/split/seed/track/init change that invalidates every
    previously selected run). Once a valid selection exists, the primary can
    be excluded from it like any other candidate — the caller is responsible
    for the "at least one model" invariant on live toggle interactions.
    """
    valid = {item.run_dir for item in candidates}
    selected = valid.intersection(str(value) for value in (previous or ()))
    if not selected:
        selected = {str(primary_run_dir)} if str(primary_run_dir) in valid else set(valid)
    return selected
