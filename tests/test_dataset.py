"""Tests for the PyTorch wrapper.

The wrapper is thin by design, so these check the seam: numpy becomes
tensors of the right dtype and shape, batching works, and multiple worker
processes can each open the store. That last one validates the lazy-open
design in WindowSource — an open zarr handle created in __init__ would fail
to pickle across to workers.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from test_windows import TRAIN, build_store, quiet
from torch.utils.data import DataLoader

from cirrus.data.dataset import ERA5Dataset
from cirrus.data.normalise import NormaliseSpec, compute_stats
from cirrus.data.splits import Period
from cirrus.data.windows import WindowSpec


@pytest.fixture
def dataset(tmp_path: Path) -> ERA5Dataset:
    path, spec = build_store(tmp_path / "store.zarr")
    norm = compute_stats(path, spec, TRAIN, NormaliseSpec(), log=quiet)
    return ERA5Dataset(path, spec, TRAIN, WindowSpec(), norm)


def test_items_are_float32_tensors(dataset: ERA5Dataset):
    item = dataset[0]
    assert isinstance(item["input"], torch.Tensor)
    assert item["input"].dtype == torch.float32
    assert item["target"].dtype == torch.float32
    assert item["input"].shape[1] == len(dataset.input_channels)
    assert item["target"].shape[1] == len(dataset.target_channels)


def test_dataloader_batches(dataset: ERA5Dataset):
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))
    n_input, n_channels = dataset[0]["input"].shape[:2]
    assert batch["input"].shape[:3] == (4, n_input, n_channels)
    assert batch["time"].shape == (4,)


def test_shuffling_is_reproducible(dataset: ERA5Dataset):
    """Same seed, same order — ablations depend on this."""

    def first_times() -> torch.Tensor:
        generator = torch.Generator().manual_seed(0)
        loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=generator)
        return next(iter(loader))["time"]

    assert torch.equal(first_times(), first_times())


def test_multiple_workers_can_each_open_the_store(dataset: ERA5Dataset):
    loader = DataLoader(dataset, batch_size=2, num_workers=2)
    batches = [b["input"].shape[0] for b in loader]
    assert sum(batches) == len(dataset)


def test_period_outside_data_gives_empty_dataset(tmp_path: Path):
    path, spec = build_store(tmp_path / "store.zarr")
    norm = compute_stats(path, spec, TRAIN, NormaliseSpec(), log=quiet)
    empty = ERA5Dataset(
        path, spec, Period("1990-01-01", "1990-12-31"), WindowSpec(), norm
    )
    assert len(empty) == 0
