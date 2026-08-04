"""Canonical training-path selection and the legacy/resumable characterization.

Two things are under test:

1. The path is decided **once**, from the full certification tuple, recorded as
   structural provenance, and never re-derived or retroactively promoted
   (ADR-WC-001 … ADR-WC-005).
2. The legacy ``train()`` loop and the ``resumable_train()`` loop are compared
   directly, which the previous tranche listed as an open risk (ADR-WC-006).
"""

from __future__ import annotations

import copy
import dataclasses
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from bench.control.config.resolver import resolve_run_spec, resolved_from_dict
from bench.control.config.schema import (
    DatasetSection,
    ExperimentSection,
    RunSpecDraft,
    SystemSection,
    structural_document,
)
from bench.control.checkpoints.batchplan import BatchPlan
from bench.control.checkpoints.training import TrainingSchedule
from bench.control.identity import ImplementationId, ModelId
from bench.control.training_path import (
    PathReasonCode,
    TrainingPathId,
    path_from_spec,
    select_training_path,
)
from tests.checkpoint_fixtures import (
    build_adapter,
    configure_threads,
    module_hash,
    seed_all,
    tiny_fixture,
)

#: The implementation ids the system actually derives (see
#: bench/control/identity.py). These must match what draft_from_suite produces
#: for a real suite entry — a certification row keyed on anything else is
#: unreachable, which is exactly the bug the real-worker E2E exposed.
CERTIFIED_IMPL = {
    "kalmannet_tsp": "bench_kalmannet_tsp_adapter_v1",
    "split_knet": "bench_split_adapter_v1",
}
UPDATES, BATCH_SIZE, SEED = 6, 2, 17


def _draft(model_id: str, *, device: str = "cpu", precision: str = "fp32",
           num_workers: int = 0, enabled: bool = True) -> RunSpecDraft:
    draft = RunSpecDraft(
        experiment=ExperimentSection(experiment_id=str(uuid.uuid4()), name="n"),
        model_id=ModelId(model_id),
        implementation_id=ImplementationId(
            CERTIFIED_IMPL.get(model_id, f"bench_{model_id}_adapter_v1")
        ),
        system=SystemSection(task_id="t", scenario_id="s", state_dim=2, observation_dim=2),
        dataset=DatasetSection(dataset_id="d"),
    )
    return dataclasses.replace(
        draft,
        training=dataclasses.replace(draft.training, enabled=enabled, max_updates=UPDATES),
        runtime=dataclasses.replace(
            draft.runtime, device=device, precision=precision, num_workers=num_workers
        ),
    )


# -- selection ---------------------------------------------------------------


@pytest.mark.parametrize("model_id", ["kalmannet_tsp", "split_knet"])
def test_certified_tuple_resolves_to_the_resumable_path(model_id: str) -> None:
    spec = resolve_run_spec(_draft(model_id))
    assert spec.draft.execution.training_path_id == str(TrainingPathId.CONTROL_RESUMABLE_V1)
    assert spec.draft.execution.certification_id


@pytest.mark.parametrize(
    "kwargs,expected_code",
    [
        ({"device": "cuda"}, PathReasonCode.UNCERTIFIED_DEVICE),
        ({"precision": "fp16"}, PathReasonCode.UNCERTIFIED_PRECISION),
        ({"num_workers": 4}, PathReasonCode.UNCERTIFIED_NUM_WORKERS),
    ],
)
def test_uncertified_envelope_falls_to_legacy_with_a_reason(kwargs, expected_code) -> None:
    """One envelope mismatch is enough; the model name never overrides it."""
    spec = resolve_run_spec(_draft("kalmannet_tsp", **kwargs))
    assert spec.draft.execution.training_path_id == str(TrainingPathId.LEGACY_TRAIN_V1)
    assert str(expected_code) in spec.draft.execution.training_path_reason_codes


@pytest.mark.parametrize("model_id", ["adaptive_knet", "maml_knet", "me_split_knet_v0"])
def test_uncertified_models_never_get_the_resumable_path(model_id: str) -> None:
    decision = select_training_path(
        model_id=model_id,
        implementation_id=f"bench_{model_id}_adapter_v1",
        training_enabled=True,
        device="cpu",
        precision="fp32",
        num_workers=0,
    )
    assert decision.training_path_id is TrainingPathId.LEGACY_TRAIN_V1
    assert not decision.is_resumable


def test_disabled_training_is_not_applicable() -> None:
    decision = select_training_path(
        model_id="kalmannet_tsp",
        implementation_id=CERTIFIED_IMPL["kalmannet_tsp"],
        training_enabled=False,
        device="cpu",
        precision="fp32",
        num_workers=0,
    )
    assert decision.training_path_id is TrainingPathId.NOT_APPLICABLE


# -- provenance --------------------------------------------------------------


def test_training_path_is_structural_provenance() -> None:
    """Two runs differing only in path must not share a structural hash."""
    certified = resolve_run_spec(_draft("kalmannet_tsp"))
    legacy = resolve_run_spec(_draft("kalmannet_tsp", device="cuda"))
    assert certified.structural_config_hash != legacy.structural_config_hash
    assert "training_path" in structural_document(certified.draft)


def test_spec_roundtrips_the_decided_path() -> None:
    document = resolve_run_spec(_draft("split_knet")).as_dict()
    assert document["execution"]["training_path_id"] == str(TrainingPathId.CONTROL_RESUMABLE_V1)
    assert resolved_from_dict(document).draft.execution.training_path_id == str(
        TrainingPathId.CONTROL_RESUMABLE_V1
    )


def test_old_spec_without_the_field_is_legacy_not_promoted() -> None:
    """A pre-contract spec is legacy forever; it is never reinterpreted."""
    document = resolve_run_spec(_draft("kalmannet_tsp")).as_dict()
    old = copy.deepcopy(document)
    old.pop("execution")
    assert resolved_from_dict(old).draft.execution.training_path_id == str(
        TrainingPathId.LEGACY_TRAIN_V1
    )
    assert path_from_spec(old) is TrainingPathId.LEGACY_TRAIN_V1
    assert path_from_spec(None) is TrainingPathId.LEGACY_TRAIN_V1
    assert path_from_spec({"training_path_id": "nonsense"}) is TrainingPathId.LEGACY_TRAIN_V1


# -- no silent fallback ------------------------------------------------------


def test_resumable_path_on_a_non_resumable_adapter_raises_not_falls_back() -> None:
    """The whole point of ADR-WC-003: refuse, never quietly run train()."""
    from bench.runners.run_suite import _try_call_train

    class NotResumable:
        def train(self, *a, **k):  # pragma: no cover - must never be reached
            raise AssertionError("legacy train() must not be called for a certified run")

    with pytest.raises(RuntimeError, match="does not implement the resumable contract"):
        _try_call_train(
            NotResumable(),
            None,
            None,
            {"train_max_updates": 1},
            Path(tempfile.mkdtemp()),
            execution_contract={"training_path_id": "control_resumable_v1"},
            train_arrays={"x": np.zeros((2, 2, 2), np.float32), "y": np.zeros((2, 2, 2), np.float32)},
        )


def test_legacy_contract_still_calls_train() -> None:
    """A legacy/absent contract must reach the untouched train() loop."""
    from bench.runners.run_suite import _try_call_train

    calls: list[str] = []

    class Legacy:
        def train(self, tl, vl, budget=None, ckpt_dir=None):
            calls.append("train")
            return {"status": "ok"}

    _try_call_train(Legacy(), None, None, {"train_max_updates": 1}, Path(tempfile.mkdtemp()))
    _try_call_train(
        Legacy(), None, None, {"train_max_updates": 1}, Path(tempfile.mkdtemp()),
        execution_contract={"training_path_id": "legacy_train_v1"},
    )
    assert calls == ["train", "train"]


# -- direct legacy vs resumable characterization (ADR-WC-006) ----------------


class _SeqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x, self.y = x, y

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int):
        return {"x": self.x[index], "y": self.y[index]}


def _legacy_run(model_id: str) -> dict:
    from bench.models.kalmannet_tsp import KalmanNetTSPAdapter
    from bench.models.split_knet import SplitKNetAdapter

    configure_threads()
    seed_all(SEED)
    cfg, system_info, x, y = tiny_fixture(model_id)
    cfg = dict(cfg)
    cfg.update({
        "lr": 1e-3, "weight_decay": 0.0, "val_eval_interval_updates": 2,
        "patience_evals": 0, "min_delta": 0.0, "max_grad_norm": None,
        "val_max_batches": 0, "gradient_clip_norm": None,
    })
    adapter = KalmanNetTSPAdapter() if model_id == "kalmannet_tsp" else SplitKNetAdapter()
    adapter.setup(cfg, system_info, {"seed": SEED, "deterministic": True, "device": "cpu"})
    module = adapter.model if model_id == "kalmannet_tsp" else adapter._filter_obj.kf_net
    initial = module_hash(module)
    # shuffle=False so both paths consume the *same* batch sequence; any
    # difference that survives is a difference between the loops themselves.
    loader = DataLoader(_SeqDataset(x, y), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val = [{"x": x[:2].copy(), "y": y[:2].copy()}]
    with tempfile.TemporaryDirectory() as tmp:
        result = adapter.train(
            loader, val, budget={"train_max_updates": UPDATES}, ckpt_dir=Path(tmp)
        )
    return {
        "initial": initial,
        "final": module_hash(module),
        "updates": int(result["updates_used"]),
        "best_step": int(result["best_step"]),
    }


def _resumable_run(model_id: str) -> dict:
    adapter, x, _ = build_adapter(model_id, SEED)
    module = adapter._ckpt_model_module()
    initial = module_hash(module)
    plan = BatchPlan(
        dataset_length=int(x.shape[0]), batch_size=BATCH_SIZE, seed=SEED, shuffle=False
    )
    result = adapter.resumable_train(
        plan=plan, schedule=TrainingSchedule(max_updates=UPDATES, eval_interval=2)
    )
    adapter.finish_resumable_training()
    return {
        "initial": initial,
        "final": module_hash(module),
        "updates": int(result.progress.global_update),
        "best_step": int(result.progress.best_step),
    }


@pytest.mark.parametrize("model_id", ["kalmannet_tsp", "split_knet"])
def test_legacy_and_resumable_agree_on_an_identical_batch_sequence(model_id: str) -> None:
    """The loops are equivalent; only batch-order derivation differs.

    This closes the risk the previous tranche left open. With the same batch
    sequence the two loops are bitwise identical, which means the paths differ
    only in *how they draw the order* — not in their update, validation,
    best-state, or early-stop semantics.
    """
    legacy = _legacy_run(model_id)
    resumable = _resumable_run(model_id)

    assert legacy["initial"] == resumable["initial"], "fixtures must start identically"
    assert legacy["final"] == resumable["final"], (
        f"{model_id}: loops diverged on an identical batch sequence"
    )
    assert legacy["updates"] == resumable["updates"] == UPDATES
    assert legacy["best_step"] == resumable["best_step"]


def test_shuffled_orders_differ_between_the_paths() -> None:
    """Documented, deliberate: the two paths derive shuffled order differently.

    ``DataLoader`` consumes an extra draw per ``iter()`` for its worker base
    seed and ``RandomSampler`` discards a trailing permutation, so matching it
    would couple us to torch internals. The batch plan is explicit instead —
    which is why a shuffled legacy run and a shuffled control run are not
    numerically comparable and carry different structural hashes.
    """
    dataset_length = 6

    class _Idx(Dataset):
        def __len__(self):
            return dataset_length

        def __getitem__(self, i):
            return i

    generator = torch.Generator()
    generator.manual_seed(SEED)
    loader = DataLoader(
        _Idx(), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, generator=generator
    )
    loader_order = [[int(v) for v in batch] for batch in loader]

    plan = BatchPlan(dataset_length=dataset_length, batch_size=BATCH_SIZE, seed=SEED)
    iterator = plan.iter_from(0)
    plan_order = [
        [int(v) for v in next(iterator)[1]] for _ in range(plan.batches_per_epoch)
    ]

    assert plan_order != loader_order

    # ...and with shuffling off they agree exactly, which is what makes the
    # comparison above a fair one.
    unshuffled = BatchPlan(
        dataset_length=dataset_length, batch_size=BATCH_SIZE, seed=SEED, shuffle=False
    )
    iterator = unshuffled.iter_from(0)
    assert [
        [int(v) for v in next(iterator)[1]] for _ in range(unshuffled.batches_per_epoch)
    ] == [[0, 1], [2, 3], [4, 5]]


def test_characterization_uses_real_third_party_modules() -> None:
    """Guard against the comparison silently running on a stub."""
    import sys

    build_adapter("kalmannet_tsp", SEED)
    build_adapter("split_knet", SEED)
    loaded = {name for name in sys.modules if "KNet" in name or "GSSFiltering" in name}
    assert any("KalmanNet_nn" in name for name in loaded), loaded
    assert any("GSSFiltering" in name for name in loaded), loaded


# -- registry propagation (found by real-worker E2E) -------------------------


def test_prepare_run_persists_the_training_path_on_the_run_row(tmp_path) -> None:
    """The decided path must reach the registry, not just the spec file.

    Found by a real WorkerManager launch: the resolved spec correctly said
    ``control_resumable_v1`` and the worker genuinely ran ``resumable_train()``,
    but the run row silently defaulted to ``legacy_train_v1`` because
    ``prepare_run`` never copied the decision across. Eligibility is answered
    from the registry, so that gap would have made every real control run look
    ineligible for resume.
    """
    from bench.control.process.manager import WorkerManager
    from bench.control.registry.sqlite import SqliteRegistry

    registry = SqliteRegistry(tmp_path / "registry.sqlite3")
    manager = WorkerManager(registry, control_root_path=tmp_path)

    spec = resolve_run_spec(_draft("kalmannet_tsp"))
    assert spec.draft.execution.training_path_id == str(TrainingPathId.CONTROL_RESUMABLE_V1)

    manager.prepare_run(spec)
    row = registry.get_run(spec.run_id.value)
    assert row.training_path_id == str(TrainingPathId.CONTROL_RESUMABLE_V1)
    assert row.training_path_reason_code == str(PathReasonCode.CERTIFIED)
    assert row.training_path_contract_version >= 1


def test_prepare_run_keeps_an_uncertified_run_legacy(tmp_path) -> None:
    from bench.control.process.manager import WorkerManager
    from bench.control.registry.sqlite import SqliteRegistry

    registry = SqliteRegistry(tmp_path / "registry.sqlite3")
    manager = WorkerManager(registry, control_root_path=tmp_path)

    spec = resolve_run_spec(_draft("kalmannet_tsp", device="cuda"))
    manager.prepare_run(spec)
    row = registry.get_run(spec.run_id.value)
    assert row.training_path_id == str(TrainingPathId.LEGACY_TRAIN_V1)
    assert row.training_path_reason_code == str(PathReasonCode.UNCERTIFIED_DEVICE)


def test_certified_implementation_ids_match_what_the_system_derives() -> None:
    """A certification keyed on an id the system never derives is unreachable.

    Split's certification row said ``bench_split_knet_adapter_v1`` while
    ``draft_from_suite`` derives ``bench_split_adapter_v1``, so Split silently
    resolved to the legacy path on every real run — found only when a real
    worker refused to honour a stop request. This pins the two together.
    """
    from bench.control.checkpoints.certification import CERTIFIED
    from bench.control.config.compatibility import draft_from_suite

    adapters = {
        "kalmannet_tsp": "bench.models.kalmannet_tsp:KalmanNetTSPAdapter",
        "split_knet": "bench.models.split_knet:SplitKNetAdapter",
    }
    for record in CERTIFIED:
        suite = {
            "suite": {"name": "x"},
            "tasks": [{"task_id": "t", "x_dim": 2, "y_dim": 2}],
            "models": [{"model_id": record.model_id, "adapter": adapters[record.model_id]}],
            "runner": {"device": "cpu", "precision": "fp32"},
        }
        draft = draft_from_suite(
            suite, task=suite["tasks"][0], model=suite["models"][0],
            seed=0, track_id="frozen", init_id="trained",
        )
        assert str(draft.implementation_id) == record.implementation_id, (
            f"{record.model_id}: certification row is keyed on "
            f"{record.implementation_id!r} but the system derives "
            f"{draft.implementation_id!r}; the certification would be unreachable"
        )
