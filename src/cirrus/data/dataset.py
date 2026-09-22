"""PyTorch ``Dataset`` over a cirrus store.

Deliberately thin. Everything with an index in it lives in
:mod:`cirrus.data.windows`, which is torch-free and therefore testable
without a model. This file converts numpy to tensors and nothing else.

On dataloader workers: ``WindowSource`` opens its zarr handle lazily, on
first access rather than in ``__init__``. Workers are separate processes, so
an open handle created in the parent would need pickling across to them.
Opening per worker sidesteps that, and is why ``num_workers > 0`` works here
without special handling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cirrus.data.ingest import IngestSpec
from cirrus.data.normalise import Normaliser, NormaliseSpec
from cirrus.data.splits import Period, Splits
from cirrus.data.windows import AugmentSpec, WindowSource, WindowSpec


class ERA5Dataset(Dataset[dict[str, torch.Tensor]]):
    """Normalised ERA5 windows as tensors.

    Each item is a dict with ``input`` ``(n_input, C_in, H, W)``, ``time``
    (a scalar int64, nanoseconds since epoch), and — unless ``n_target`` is
    zero — ``target`` ``(n_target, C_dynamic, H, W)``.
    """

    def __init__(
        self,
        store: str | Path,
        spec: IngestSpec,
        period: Period,
        window: WindowSpec,
        normaliser: Normaliser,
        augment: AugmentSpec | None = None,
    ) -> None:
        self.source = WindowSource(
            store, spec, period, window, normaliser, augment=augment
        )

    @classmethod
    def from_configs(
        cls,
        split: str = "train",
        data: str | Path = "configs/data/era5_5625.yaml",
        splits: str | Path = "configs/data/splits.yaml",
        window: str | Path = "configs/data/windows.yaml",
        normalise: str | Path = "configs/data/normalise.yaml",
        augment: str | Path = "configs/data/augment.yaml",
    ) -> ERA5Dataset:
        """Build a dataset for one split from the project's config files."""
        spec = IngestSpec.from_yaml(data)
        all_splits = Splits.from_yaml(splits)
        period = getattr(all_splits, split, None)
        if period is None:
            raise ValueError(f"unknown split {split!r}; expected train, val or test")
        nspec = NormaliseSpec.from_yaml(normalise)
        # Augmentation on the training split only. Augmenting validation or
        # test would make scores depend on a random draw and, for the roll,
        # would quietly change which grid cells a regional score covers.
        augmentation = AugmentSpec.from_yaml(augment) if split == "train" else None
        return cls(
            store=spec.output,
            spec=spec,
            period=period,
            window=WindowSpec.from_yaml(window),
            normaliser=Normaliser.load(nspec.output),
            augment=augmentation,
        )

    @property
    def input_channels(self) -> tuple[str, ...]:
        """Channel names of the input tensor, in order."""
        return self.source.input_channels

    @property
    def target_channels(self) -> tuple[str, ...]:
        """Channel names of the target tensor, in order."""
        return self.source.target_channels

    def set_epoch(self, epoch: int) -> None:
        """Change the augmentation draw. Call once per epoch when training."""
        self.source.set_epoch(epoch)

    def __len__(self) -> int:
        """Return the number of valid windows in this split."""
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one sample as tensors."""
        # asarray, not ascontiguousarray: the latter promotes the scalar
        # timestamp to shape (1,), which then batches to (B, 1) instead of
        # (B,). Arrays from WindowSource are already contiguous.
        return {
            key: torch.from_numpy(np.asarray(value))
            for key, value in self.source.sample(index).items()
        }
