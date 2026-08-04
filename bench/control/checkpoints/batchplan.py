"""Deterministic, resumable batch plan.

Why this exists
---------------
The adapters' training loops are ``while updates < max: for batch in loader:``
over a ``DataLoader(shuffle=True, generator=g)``. That is perfectly
reproducible from the start, but it is *not* resumable: the batch order is an
emergent property of a ``RandomSampler`` consuming ``g``'s state, so to land on
update N you must replay every draw that came before it.

Rather than reach into DataLoader internals — which would make the exact-resume
claim depend on torch's sampler implementation and on worker semantics we have
not certified — the plan is made explicit (§7 option B). The batch order for a
run is a pure function of ``(dataset_length, batch_size, seed, drop_last)``,
addressable by a global position, and therefore checkpointable as two integers.

This is *opt-in*. Runs that do not enable checkpointing keep using the existing
DataLoader path untouched, so no existing numerical result moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np

from ..canonical import content_hash

#: Per-epoch permutations are drawn from an independent generator seeded from
#: (base_seed, epoch). Independence per epoch is what makes an arbitrary global
#: position seekable in O(1) epochs rather than by replaying every prior draw.
_EPOCH_SALT = 0x9E3779B9


@dataclass(frozen=True)
class BatchPlan:
    """A reproducible schedule of dataset indices.

    The plan is infinite in the epoch direction: training stops on the update
    budget, not on the plan running out.
    """

    dataset_length: int
    batch_size: int
    seed: int
    drop_last: bool = False
    #: When False the plan yields the dataset in index order, matching
    #: ``DataLoader(shuffle=False)``. This is what makes a direct
    #: legacy-vs-resumable numerical comparison possible: with the same batch
    #: sequence, any remaining difference is a difference between the *loops*
    #: rather than between two ways of drawing a permutation.
    shuffle: bool = True

    def __post_init__(self) -> None:
        if self.dataset_length <= 0:
            raise ValueError("dataset_length must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

    @property
    def plan_id(self) -> str:
        """Content hash of the plan's defining parameters.

        Stored in the checkpoint so a resume into a differently-shaped dataset
        is rejected instead of silently training on a different order.
        """
        return content_hash(
            {
                "kind": "batch_plan_v1",
                "dataset_length": int(self.dataset_length),
                "batch_size": int(self.batch_size),
                "seed": int(self.seed),
                "drop_last": bool(self.drop_last),
                "shuffle": bool(self.shuffle),
            }
        )

    @property
    def batches_per_epoch(self) -> int:
        full, remainder = divmod(self.dataset_length, self.batch_size)
        if self.drop_last or remainder == 0:
            return max(1, full) if not self.drop_last else full
        return full + 1

    def epoch_permutation(self, epoch: int) -> np.ndarray:
        """Indices for one epoch, independent of every other epoch."""
        if not self.shuffle:
            return np.arange(int(self.dataset_length))
        generator = np.random.default_rng((int(self.seed) + _EPOCH_SALT * int(epoch)) & 0xFFFFFFFF)
        return generator.permutation(int(self.dataset_length))

    def epoch_batches(self, epoch: int) -> list[np.ndarray]:
        permutation = self.epoch_permutation(epoch)
        batches: list[np.ndarray] = []
        for start in range(0, self.dataset_length, self.batch_size):
            chunk = permutation[start : start + self.batch_size]
            if self.drop_last and len(chunk) < self.batch_size:
                continue
            batches.append(chunk)
        return batches

    def batch_at(self, position: int) -> tuple[int, int, np.ndarray]:
        """Resolve a global batch position to ``(epoch, index_in_epoch, indices)``.

        This is the seek operation resume depends on: position is absolute, so
        restoring a cursor never requires replaying earlier batches.
        """
        if position < 0:
            raise ValueError("position must be >= 0")
        per_epoch = self.batches_per_epoch
        epoch, index = divmod(int(position), per_epoch)
        return epoch, index, self.epoch_batches(epoch)[index]

    def iter_from(self, position: int) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(position, indices)`` starting at ``position``, unbounded."""
        per_epoch = self.batches_per_epoch
        epoch, index = divmod(int(position), per_epoch)
        current = int(position)
        while True:
            batches = self.epoch_batches(epoch)
            while index < len(batches):
                yield current, batches[index]
                index += 1
                current += 1
            epoch += 1
            index = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "dataset_length": int(self.dataset_length),
            "batch_size": int(self.batch_size),
            "seed": int(self.seed),
            "drop_last": bool(self.drop_last),
            "shuffle": bool(self.shuffle),
            "batches_per_epoch": int(self.batches_per_epoch),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "BatchPlan":
        return cls(
            dataset_length=int(data["dataset_length"]),  # type: ignore[arg-type]
            batch_size=int(data["batch_size"]),  # type: ignore[arg-type]
            seed=int(data["seed"]),  # type: ignore[arg-type]
            drop_last=bool(data.get("drop_last", False)),
            shuffle=bool(data.get("shuffle", True)),
        )


def dataset_fingerprint(x: np.ndarray, y: np.ndarray, extras: Optional[Sequence[np.ndarray]] = None) -> str:
    """Content hash of the training arrays.

    Resuming into different data is a silent corruption of a research result,
    so the fingerprint is a hard compatibility key rather than advisory
    metadata (ADR-CSR §4.1).
    """
    import hashlib

    digest = hashlib.sha256()
    for array in (x, y, *(extras or ())):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(repr(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return f"sha256:{digest.hexdigest()}"
