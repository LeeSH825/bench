from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from viz.analysis import comparison
from viz.io.loader import TrajectoryInfo, VizRun


@dataclass(frozen=True)
class ComparisonCandidate:
    run: VizRun
    trajectory: Optional[TrajectoryInfo]
    compatibility: Mapping[str, Any]


def comparison_run_label(run: VizRun) -> str:
    return (
        f"{run.meta.get('model_id')} · {run.meta.get('formulation')} · "
        f"seed {run.meta.get('seed')} · {run.meta.get('track_id')}"
    )


def _is_relevant(base: VizRun, candidate: VizRun) -> bool:
    base_spec = base.meta.get("comparison_spec")
    candidate_spec = candidate.meta.get("comparison_spec")
    if isinstance(base_spec, Mapping) and isinstance(candidate_spec, Mapping):
        base_identity = base_spec.get("identity", {})
        candidate_identity = candidate_spec.get("identity", {})
        base_physical = base_identity.get("physical_scenario_id") if isinstance(base_identity, Mapping) else None
        candidate_physical = (
            candidate_identity.get("physical_scenario_id") if isinstance(candidate_identity, Mapping) else None
        )
        if base_physical is not None and candidate_physical == base_physical:
            return True
    return (
        base.meta.get("task") == candidate.meta.get("task")
        or base.meta.get("scenario_id") == candidate.meta.get("scenario_id")
    )


def comparison_candidates(
    base: VizRun,
    runs: Sequence[VizRun],
    *,
    source_trajectory_id: Any,
    mode: str,
    metric: str,
) -> list[ComparisonCandidate]:
    evaluator = (
        comparison.evaluate_physical_metric_compatibility
        if mode == "physical"
        else comparison.evaluate_internal_metric_compatibility
    )
    candidates: list[ComparisonCandidate] = []
    for run in runs:
        if run.run_dir == base.run_dir or not _is_relevant(base, run):
            continue
        info = None
        source_value = None
        try:
            info = run.trajectory_by_source_id(source_trajectory_id)
            source_value = info.source_trajectory_id
        except KeyError:
            pass
        status = evaluator(
            base.meta,
            run.meta,
            metric=metric,
            base_source_trajectory_id=source_trajectory_id,
            candidate_source_trajectory_id=source_value,
        )
        candidates.append(ComparisonCandidate(run=run, trajectory=info, compatibility=status))
    return sorted(candidates, key=lambda item: comparison_run_label(item.run))
