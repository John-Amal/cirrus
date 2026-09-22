"""Train / validation / test periods, defined in exactly one place.

Every component that needs to know which years are for training — the
normalisation statistics, the dataset, the evaluation — reads them from here.
A split defined in two places will eventually be defined two different ways.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from cirrus.config import read_yaml_mapping

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def check_date(value: str, name: str) -> None:
    """Require ISO ``YYYY-MM-DD``, so dates also compare correctly as strings."""
    if not _DATE.match(value):
        raise ValueError(f"{name} must be YYYY-MM-DD, got {value!r}")


@dataclass(frozen=True)
class Period:
    """An inclusive date range."""

    start: str
    end: str

    def __post_init__(self) -> None:
        """Validate format and order."""
        check_date(self.start, "start")
        check_date(self.end, "end")
        if self.start > self.end:
            raise ValueError(f"period start {self.start} is after end {self.end}")

    @property
    def years(self) -> range:
        """Calendar years touched by the period."""
        return range(int(self.start[:4]), int(self.end[:4]) + 1)

    def clip_to_year(self, year: int) -> tuple[str, str]:
        """Return the part of this period falling inside ``year``."""
        return max(self.start, f"{year}-01-01"), min(self.end, f"{year}-12-31")


@dataclass(frozen=True)
class Splits:
    """Three consecutive, non-overlapping periods."""

    train: Period
    val: Period
    test: Period

    def __post_init__(self) -> None:
        """Enforce train < val < test with no shared days."""
        if not self.train.end < self.val.start:
            raise ValueError("train period must end before validation starts")
        if not self.val.end < self.test.start:
            raise ValueError("validation period must end before test starts")

    @classmethod
    def from_yaml(cls, path: str | Path) -> Splits:
        """Load from a mapping of name -> [start, end]."""
        raw = read_yaml_mapping(path, ("train", "val", "test"))
        missing = {"train", "val", "test"} - set(raw)
        if missing:
            raise ValueError(f"{path} is missing splits: {sorted(missing)}")
        return cls(**{name: Period(*bounds) for name, bounds in raw.items()})
