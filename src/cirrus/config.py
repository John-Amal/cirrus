"""Run configuration.

A run is fully described by one YAML file. No hidden defaults scattered
through the code: if it changes the result, it lives here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    """Everything needed to reproduce a run."""

    name: str = "tiny"
    seed: int = 0

    # Data (Phase 0: random tensors standing in for ERA5 fields)
    n_samples: int = 512
    n_channels: int = 4
    grid_height: int = 32
    grid_width: int = 64

    # Model
    hidden_dim: int = 64

    # Optimisation
    batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 1e-3

    # Execution
    device: str | None = None  # None means autodetect

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a config, rejecting unknown keys.

        Rejecting unknown keys matters more than it looks: a silently ignored
        typo (``learning_rate_`` instead of ``learning_rate``) means a run that
        quietly used the default and results you cannot explain later.
        """
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"unknown config keys in {path}: {sorted(unknown)}. "
                f"Known keys: {sorted(known)}"
            )
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict, for logging alongside results."""
        return asdict(self)
