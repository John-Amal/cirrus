"""Device selection.

Every script in this repo picks its device through :func:`get_device` so the
same code runs unchanged on a Mac (``mps``), on a Kaggle/HPC GPU (``cuda``)
and in CI (``cpu``).
"""

from __future__ import annotations

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available device.

    Priority: cuda -> mps -> cpu.

    Args:
        prefer: Force a specific device string (``"cpu"``, ``"mps"``,
            ``"cuda"``). Used by tests and by ``--device`` on the CLI to
            override autodetection.

    Raises:
        RuntimeError: If ``prefer`` names a device that is not available.

    """
    if prefer is not None:
        if prefer == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("cuda requested but not available")
        if prefer == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("mps requested but not available")
        return torch.device(prefer)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_report(device: torch.device) -> str:
    """One-line human-readable description of the device, for logging."""
    if device.type == "cuda":
        return f"cuda: {torch.cuda.get_device_name(0)}"
    if device.type == "mps":
        return "mps: Apple Silicon GPU (unified memory)"
    return "cpu"
