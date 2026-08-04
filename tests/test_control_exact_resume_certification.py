"""Exact-resume certification for kalmannet_tsp and split_knet.

The claim under test: interrupting training at a completed optimizer update,
writing a checkpoint, and resuming into a *freshly constructed* adapter
produces bitwise-identical results to training straight through.

Every resume here builds its adapter with a **different seed** than the run
being resumed. That is deliberate: if any seed-dependent state were being
carried implicitly instead of through the checkpoint, these tests would pass
for the wrong reason. Finding exactly such a case — Split-KalmanNet's
randn-initialised GRU hidden state, which lives outside ``state_dict()`` — is
what motivated the adapter's ``_ckpt_extra_state`` hook.

Certified envelope: CPU / fp32 / deterministic algorithms / num_workers=0
(ADR-CSR-002). Nothing here certifies GPU, AMP, or a multi-worker loader.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from bench.control.checkpoints import (
    BatchPlan,
    CheckpointKind,
    CheckpointService,
    TrainingCursor,
    capture_rng,
    restore_rng,
)
from bench.control.checkpoints.training import TrainingProgress, TrainingSchedule
from tests.checkpoint_fixtures import build_adapter, fingerprint

MODELS = ["kalmannet_tsp", "split_knet"]
TOTAL_UPDATES = 6
INTERRUPT_AT = 3
SEED = 17
RESUME_SEED = SEED + 999

REPO_ROOT = Path(__file__).resolve().parents[1]


def _plan() -> BatchPlan:
    return BatchPlan(dataset_length=6, batch_size=2, seed=SEED)


def _schedule(max_updates: int = TOTAL_UPDATES) -> TrainingSchedule:
    return TrainingSchedule(max_updates=max_updates, eval_interval=2)


def _run_continuous(model_id: str) -> dict:
    adapter, _, _ = build_adapter(model_id, SEED)
    result = adapter.resumable_train(plan=_plan(), schedule=_schedule())
    return fingerprint(adapter, result)


def _run_to_interrupt(model_id: str):
    """Train until the stop flag trips at a safe boundary, then capture."""
    adapter, _, _ = build_adapter(model_id, SEED)
    progress = TrainingProgress()
    result = adapter.resumable_train(
        plan=_plan(),
        schedule=_schedule(),
        progress=progress,
        stop_requested=lambda: progress.global_update >= INTERRUPT_AT,
    )
    assert result.interrupted, "expected the stop flag to interrupt training"
    assert result.progress.global_update == INTERRUPT_AT
    cursor = TrainingCursor(
        global_update=result.progress.global_update,
        batch_plan_position=result.progress.batch_plan_position,
        batch_plan_id=_plan().plan_id,
    )
    state = adapter.capture_training_state(cursor)
    return adapter, state, capture_rng(), result.progress.as_dict()


def _resume(model_id: str, state, rng, progress_dict: dict) -> dict:
    adapter, _, _ = build_adapter(model_id, RESUME_SEED)
    adapter.restore_training_state(state)
    restore_rng(rng)
    result = adapter.resumable_train(
        plan=_plan(),
        schedule=_schedule(),
        progress=TrainingProgress.from_dict(progress_dict),
    )
    return fingerprint(adapter, result)


# -- in-process certification ------------------------------------------------


@pytest.mark.parametrize("model_id", MODELS)
def test_resume_matches_continuous_bitwise(model_id: str) -> None:
    continuous = _run_continuous(model_id)
    _, state, rng, progress = _run_to_interrupt(model_id)
    resumed = _resume(model_id, state, rng, progress)

    assert resumed["updates"] == TOTAL_UPDATES
    for key in continuous:
        assert resumed[key] == continuous[key], f"{model_id}: {key} diverged after resume"


@pytest.mark.parametrize("model_id", MODELS)
def test_resume_through_checkpoint_package(model_id: str, tmp_path: Path) -> None:
    """The same claim, but the state round-trips through a real package on disk."""
    continuous = _run_continuous(model_id)
    adapter, state, rng, progress = _run_to_interrupt(model_id)

    service = CheckpointService(tmp_path / "run", control_root=tmp_path)
    saved = service.save(
        run_id="run-under-test",
        kind=CheckpointKind.INTERRUPT,
        cursor=TrainingCursor(
            global_update=progress["global_update"],
            batch_plan_position=progress["batch_plan_position"],
            batch_plan_id=_plan().plan_id,
        ),
        adapter_state=state,
        rng=rng,
        identity={"model_id": model_id, "implementation_id": f"{model_id}_v1"},
        structural_config_hash="sha256:test",
        dataset_fingerprint="sha256:test-data",
        batch_plan=_plan(),
        capabilities=adapter.checkpoint_capabilities(),
    )
    assert service.validate(saved.checkpoint_id).valid

    _, cursor, restored_state, restored_rng, payload = service.load(saved.checkpoint_id)
    assert cursor.global_update == INTERRUPT_AT
    assert payload["batch_plan"]["plan_id"] == _plan().plan_id

    fresh, _, _ = build_adapter(model_id, RESUME_SEED)
    fresh.restore_training_state(restored_state)
    restore_rng(restored_rng)
    result = fresh.resumable_train(
        plan=BatchPlan.from_dict(payload["batch_plan"]),
        schedule=_schedule(),
        progress=TrainingProgress.from_dict(progress),
    )
    resumed = fingerprint(fresh, result)
    for key in continuous:
        assert resumed[key] == continuous[key], f"{model_id}: {key} diverged via package"


# -- fresh-process certification ---------------------------------------------

_CHILD = r"""
import json, sys
sys.path.insert(0, {repo!r})
from pathlib import Path
from bench.control.checkpoints import BatchPlan, CheckpointService, restore_rng
from bench.control.checkpoints.training import TrainingProgress, TrainingSchedule
from tests.checkpoint_fixtures import build_adapter, fingerprint

model_id, root, ckpt_id, progress_json, resume_seed = sys.argv[1:6]
service = CheckpointService(Path(root) / "run", control_root=Path(root))
_, cursor, state, rng, payload = service.load(ckpt_id)

adapter, _, _ = build_adapter(model_id, int(resume_seed))
adapter.restore_training_state(state)
restore_rng(rng)
result = adapter.resumable_train(
    plan=BatchPlan.from_dict(payload["batch_plan"]),
    schedule=TrainingSchedule(max_updates={total}, eval_interval=2),
    progress=TrainingProgress.from_dict(json.loads(progress_json)),
)
print("__RESULT__" + json.dumps(fingerprint(adapter, result)))
"""


@pytest.mark.parametrize("model_id", MODELS)
def test_resume_in_a_fresh_process(model_id: str, tmp_path: Path) -> None:
    """Certification requires a brand-new process, not just a new object.

    An in-process resume can accidentally inherit module-level or RNG state.
    Spawning a child interpreter removes that whole class of false pass.
    """
    continuous = _run_continuous(model_id)
    adapter, state, rng, progress = _run_to_interrupt(model_id)

    service = CheckpointService(tmp_path / "run", control_root=tmp_path)
    saved = service.save(
        run_id="run-under-test",
        kind=CheckpointKind.INTERRUPT,
        cursor=TrainingCursor(
            global_update=progress["global_update"],
            batch_plan_position=progress["batch_plan_position"],
            batch_plan_id=_plan().plan_id,
        ),
        adapter_state=state,
        rng=rng,
        identity={"model_id": model_id, "implementation_id": f"{model_id}_v1"},
        batch_plan=_plan(),
        capabilities=adapter.checkpoint_capabilities(),
    )
    del adapter

    script = _CHILD.format(repo=str(REPO_ROOT), total=TOTAL_UPDATES)
    completed = subprocess.run(
        [sys.executable, "-c", script, model_id, str(tmp_path), saved.checkpoint_id,
         json.dumps(progress), str(RESUME_SEED)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert completed.returncode == 0, f"child failed:\n{completed.stdout}\n{completed.stderr}"
    marker = [l for l in completed.stdout.splitlines() if l.startswith("__RESULT__")]
    assert marker, f"child produced no result:\n{completed.stdout}\n{completed.stderr}"
    resumed = json.loads(marker[0][len("__RESULT__"):])

    for key in continuous:
        assert resumed[key] == continuous[key], (
            f"{model_id}: {key} diverged after a fresh-process resume"
        )


# -- mutation probes ---------------------------------------------------------
#
# A parity test that cannot fail proves nothing. Each probe removes exactly one
# piece of checkpointed state and asserts the comparison notices.


@pytest.mark.parametrize("model_id", MODELS)
def test_probe_omitting_optimizer_state_is_detected(model_id: str) -> None:
    continuous = _run_continuous(model_id)
    _, state, rng, progress = _run_to_interrupt(model_id)

    adapter, _, _ = build_adapter(model_id, RESUME_SEED)
    adapter.restore_training_state(state)
    # Adam's exponential moving averages are what make the next step depend on
    # history; dropping them is the classic "resume" that silently is not one.
    adapter._ckpt_session.optimizer.state.clear()
    restore_rng(rng)
    result = adapter.resumable_train(
        plan=_plan(), schedule=_schedule(), progress=TrainingProgress.from_dict(progress)
    )
    mutated = fingerprint(adapter, result)
    assert mutated["weights"] != continuous["weights"], (
        f"{model_id}: dropping optimizer state did not change the result — "
        "the parity assertion is not sensitive"
    )


@pytest.mark.parametrize("model_id", MODELS)
def test_probe_cursor_off_by_one_is_detected(model_id: str) -> None:
    continuous = _run_continuous(model_id)
    _, state, rng, progress = _run_to_interrupt(model_id)

    shifted = dict(progress)
    shifted["batch_plan_position"] = int(shifted["batch_plan_position"]) - 1
    resumed = _resume(model_id, state, rng, shifted)
    assert resumed["weights"] != continuous["weights"], (
        f"{model_id}: replaying one batch did not change the result"
    )


def test_probe_omitting_split_hidden_state_is_detected() -> None:
    """Split-specific: the state that ``state_dict()`` does not carry.

    ``DNN_SKalmanNet_GSS`` creates its GRU initial hidden states with
    ``torch.randn`` as plain attributes. They are not registered buffers, so a
    checkpoint built only from ``state_dict()`` silently omits them and a
    resume into a differently-seeded process diverges.
    """
    continuous = _run_continuous("split_knet")
    _, state, rng, progress = _run_to_interrupt("split_knet")

    stripped = dict(state.extra_state)
    stripped["adapter"] = {}
    state.extra_state = stripped

    adapter, _, _ = build_adapter("split_knet", RESUME_SEED)
    with pytest.raises(Exception) as excinfo:
        adapter.restore_training_state(state)
    assert "hidden state" in str(excinfo.value).lower()

    # And if the guard were absent, the numbers really would move: restore the
    # weights/optimizer only and show divergence.
    adapter2, _, _ = build_adapter("split_knet", RESUME_SEED)
    module = adapter2._ckpt_model_module()
    module.load_state_dict(state.model_slots["model"], strict=True)
    adapter2._ckpt_session.optimizer.load_state_dict(state.optimizer_slots["main"])
    restore_rng(rng)
    result = adapter2.resumable_train(
        plan=_plan(), schedule=_schedule(), progress=TrainingProgress.from_dict(progress)
    )
    assert fingerprint(adapter2, result)["weights"] != continuous["weights"]


# -- capability declarations -------------------------------------------------


@pytest.mark.parametrize("model_id", MODELS)
def test_capabilities_declare_the_certified_envelope(model_id: str) -> None:
    adapter, _, _ = build_adapter(model_id, SEED)
    capabilities = adapter.checkpoint_capabilities()
    assert capabilities.supports_exact_resume is True
    assert capabilities.resume_boundary == "optimizer_update"
    assert capabilities.certified_device == "cpu"
    assert capabilities.certified_precision == "fp32"
    assert capabilities.certified_num_workers == 0
    assert capabilities.model_slots and capabilities.optimizer_slots
    if model_id == "split_knet":
        assert "kf_net.hn1_init" in capabilities.required_conditional_state


def test_uncertified_adapters_do_not_claim_exact_resume() -> None:
    """Adaptive/MAML/ME-Split must stay uncertified (ADR-CSR-006)."""
    from bench.models.registry import get_adapter_class

    for model_id in ("adaptive_knet", "maml_knet"):
        adapter_cls = get_adapter_class(model_id)
        assert not hasattr(adapter_cls, "capture_training_state") or not issubclass(
            adapter_cls,
            __import__(
                "bench.models.checkpoint_support", fromlist=["CheckpointableAdapterMixin"]
            ).CheckpointableAdapterMixin,
        ), f"{model_id} must not inherit exact-resume support without certification"
