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


CONTEXT_FIELDS = ("suite", "task", "scenario_id", "split", "seed", "track_id")
"""Evaluation-compatibility hard gate: same test data and evaluation protocol.

`init_id` is deliberately NOT here. It is training/initialization provenance
(trained/pretrained/untrained/adapted/...), not evidence the runs evaluate
different data — a model-based KF compared against a trained learned filter,
or a trained-vs-untrained ablation of the same model, are exactly the
comparisons this tool exists to support. See
reports/VIZ_INIT_PROVENANCE_COMPARISON_FIX_REPORT.md for the audit that
justified this (and why `track_id`/`seed` stayed as hard gates).
"""


def model_context_key(meta: Mapping[str, Any]) -> tuple[Any, ...]:
    data_spec = meta.get("data_spec") if isinstance(meta.get("data_spec"), Mapping) else {}
    return tuple(meta.get(field, data_spec.get(field)) for field in CONTEXT_FIELDS)


def variant_label(meta: Mapping[str, Any]) -> str:
    """Model + training/init provenance identity for one run variant.

    Two candidates can share `model_id` once `init_id` is out of the hard
    context gate (e.g. the same model trained vs. untrained), so nothing
    downstream — checkbox label, trace name/legend, trace color, or the
    Advanced diagnostics matrix — may use bare `model_id` as an identity key;
    that would silently merge distinct variants. This is that shared
    identity string, reused everywhere a variant needs to be named.
    """
    model = str(meta.get("model_id") or "unknown model")
    init = str(meta.get("init_id") or "unknown")
    checkpoint = meta.get("checkpoint")
    if checkpoint:
        return f"{model} · init={init} · ckpt={checkpoint}"
    return f"{model} · init={init}"


def candidate_label(meta: Mapping[str, Any]) -> str:
    track = str(meta.get("track_id") or "unknown track")
    seed = str(meta.get("seed") if meta.get("seed") is not None else "unknown")
    return f"{variant_label(meta)} · {track} / seed {seed}"


def _disambiguate_labels(candidates: Sequence["ModelToggleCandidate"]) -> list["ModelToggleCandidate"]:
    """Append a short run identifier to any candidates whose label collides.

    `candidate_label` already encodes model_id/init_id/checkpoint/track/seed,
    so a collision only remains for genuinely distinct runs that agree on
    all of those (e.g. two checkpoints of the same trained variant with no
    checkpoint id recorded) — a real edge case the label alone cannot resolve.
    """
    counts: dict[str, int] = {}
    for item in candidates:
        counts[item.label] = counts.get(item.label, 0) + 1
    if all(count == 1 for count in counts.values()):
        return list(candidates)
    disambiguated = []
    for item in candidates:
        if counts[item.label] == 1:
            disambiguated.append(item)
            continue
        short_id = f"{abs(hash(item.run_dir)):x}"[:6]
        disambiguated.append(
            ModelToggleCandidate(
                run_dir=item.run_dir,
                model_id=item.model_id,
                label=f"{item.label} · run={short_id}",
                metadata=item.metadata,
            )
        )
    return disambiguated


def suite_candidates(
    primary: Any,
    indexed_runs: Sequence[Any],
    *,
    source_trajectory_id: Any,
) -> list[ModelToggleCandidate]:
    """Discover manifest candidates only; this function never loads an NPZ.

    Uniqueness is by `run_dir`, not `model_id`: multiple init/training
    variants of the same model_id (e.g. trained vs. untrained) are distinct
    candidates and must all survive this pass.
    """
    context = model_context_key(primary.meta)
    primary_run_dir = str(primary.run_dir)
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
    ordered = sorted(
        result.values(),
        key=lambda item: (
            item.run_dir != primary_run_dir,
            item.model_id,
            str(item.metadata.get("init_id") or ""),
            item.run_dir,
        ),
    )
    return _disambiguate_labels(ordered)


def reconcile_selection(
    candidates: Sequence[ModelToggleCandidate],
    *,
    primary_run_dir: str,
    previous: Sequence[str] | None,
) -> set[str]:
    """Drop stale context entries; default to the primary alone when nothing valid remains.

    The primary is only force-selected as a *default* (first load, or a
    suite/task/scenario/split/seed/track change that invalidates every
    previously selected run — see CONTEXT_FIELDS; init_id changes do not
    invalidate anything). Once a valid selection exists, the primary can
    be excluded from it like any other candidate — the caller is responsible
    for the "at least one model" invariant on live toggle interactions.
    """
    valid = {item.run_dir for item in candidates}
    selected = valid.intersection(str(value) for value in (previous or ()))
    if not selected:
        selected = {str(primary_run_dir)} if str(primary_run_dir) in valid else set(valid)
    return selected
