"""Exporting a trained arm to a portable format.

What gets exported is the **parameter predictor**, not a sampler. The CSGD
head's draws come from ``Gamma.rsample``, which is not a traceable graph
operation -- and exporting it would be the wrong interface anyway. A consumer
that receives shape, scale and shift gets the entire predictive distribution
and can draw as many samples as it likes, compute any quantile, or evaluate
an exceedance probability directly. A consumer that receives 24 draws gets
whatever someone else decided 24 was good for.

Point heads export their single value, so both kinds present the same shape
of interface: fields in, per-cell prediction out.

Two things are checked rather than assumed. **Numerical equivalence**: the
exported model must reproduce the PyTorch outputs, because a silent
divergence in an export is the kind of bug that surfaces in production and
nowhere else. And **the quantisation trade**: dynamic quantisation is only
worth it if the latency gain exceeds the accuracy cost, which is a
measurement, not a default.

Quantisation also needs a backend compiled into the PyTorch build, and some
builds ship without one. That is an environment limitation rather than a
failure of the model, so it is detected and reported instead of raised.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from cirrus.models.vit import ViT
from cirrus.train.finetune import FinetuneSpec

PARAMETER_NAMES = {
    "csgd": ("shape", "scale", "shift"),
    "point": ("precipitation_mm",),
}

# qnnpack is the ARM backend and the one that matters on Apple Silicon; the
# others are x86. Ordered by preference, filtered against what the build has.
QUANTISATION_ENGINES = ("qnnpack", "x86", "fbgemm", "onednn")


class ParameterPredictor(nn.Module):
    """Backbone and head as one traceable module.

    Returns ``(batch, n_parameters, lat, lon)``: three channels for a
    distributional head, one for a point head. No sampling, no randomness --
    the graph is deterministic, which is what makes it exportable and what
    makes the equivalence check meaningful.
    """

    def __init__(self, backbone: ViT, head: nn.Module, distributional: bool) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.distributional = distributional

    @property
    def kind(self) -> str:
        """Return which parameter set this predictor emits."""
        return "csgd" if self.distributional else "point"

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, H, W)`` fields to per-cell parameters."""
        tokens: torch.Tensor = self.backbone(fields)
        if self.distributional:
            distribution = self.head(tokens)
            return torch.stack(
                [distribution.shape, distribution.scale, distribution.shift], dim=1
            )
        point: torch.Tensor = self.head(tokens)
        return point.unsqueeze(1)


@dataclass
class ExportReport:
    """What an export produced and whether it is trustworthy."""

    path: Path
    format: str
    max_difference: float
    seconds_per_batch: float
    note: str = ""

    @property
    def is_equivalent(self) -> bool:
        """Report whether the export matches PyTorch closely enough.

        1e-4 in normalised parameter units: tighter than float32 accumulation
        order differences would justify demanding, looser than anything that
        would change a prediction.
        """
        return self.max_difference < 1e-4


def example_input(predictor: ParameterPredictor, batch: int = 1) -> torch.Tensor:
    """Build a correctly shaped dummy batch for tracing."""
    channels = predictor.backbone.patch_embed.in_channels
    return torch.randn(batch, channels, *predictor.backbone.grid)


def measure_latency(
    model: nn.Module, example: torch.Tensor, repeats: int = 20
) -> float:
    """Measure mean seconds per forward pass, discarding the first.

    The first call compiles kernels and caches them, so including it would
    overstate the cost several-fold.
    """
    with torch.no_grad():
        model(example)
        start = time.perf_counter()
        for _ in range(repeats):
            model(example)
    return (time.perf_counter() - start) / repeats


def export_torchscript(
    predictor: ParameterPredictor, path: Path, example: torch.Tensor
) -> ExportReport:
    """Trace to TorchScript and verify it reproduces the original.

    Tracing emits warnings about the shape assertions in ``forward``: they are
    Python conditionals on tensor shapes, so the trace records their outcome
    as a constant. That is harmless here, because the exported model is fixed
    to one input shape anyway -- position embeddings are tied to the grid.
    """
    predictor.eval()
    with torch.no_grad():
        expected = predictor(example)
        traced = torch.jit.trace(predictor, example)
        traced = torch.jit.freeze(traced)  # folds the frozen backbone's constants
        actual = traced(example)

    path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(path))
    return ExportReport(
        path=path,
        format="torchscript",
        max_difference=float((expected - actual).abs().max()),
        seconds_per_batch=measure_latency(traced, example),
    )


def export_onnx(
    predictor: ParameterPredictor, path: Path, example: torch.Tensor
) -> ExportReport:
    """Export to ONNX and verify it against PyTorch through onnxruntime.

    The batch axis is dynamic; the grid is not. A model trained at 32x64 has
    position embeddings tied to that grid and cannot accept another.
    """
    predictor.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        expected = predictor(example)

    torch.onnx.export(
        predictor,
        (example,),
        str(path),
        input_names=["fields"],
        output_names=["parameters"],
        dynamic_axes={"fields": {0: "batch"}, "parameters": {0: "batch"}},
        opset_version=17,
    )

    import onnxruntime  # imported here so the dependency stays optional

    session = onnxruntime.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    inputs = {"fields": example.numpy()}
    actual = session.run(None, inputs)[0]

    started = time.perf_counter()
    for _ in range(20):
        session.run(None, inputs)
    seconds = (time.perf_counter() - started) / 20

    difference = float(torch.from_numpy(actual).sub(expected).abs().max())
    return ExportReport(path, "onnx", difference, seconds)


def quantisation_engine() -> str | None:
    """Return a usable quantisation backend, or None if the build has none.

    ``supported_engines`` reports what was compiled in. A build without a
    backend raises ``NoQEngine`` deep inside the conversion, which is an
    unhelpful place to discover it.
    """
    available = set(torch.backends.quantized.supported_engines) - {"none"}
    for engine in QUANTISATION_ENGINES:
        if engine in available:
            return engine
    return next(iter(available), None)


def quantise(
    predictor: ParameterPredictor, example: torch.Tensor
) -> ExportReport | None:
    """Dynamically quantise the linear layers and measure what it costs.

    Dynamic quantisation stores weights as int8 and computes activations in
    float, which suits transformers: almost all of the parameters are in
    linear layers. Whether it is worth using is a measurement -- report the
    latency gain beside the accuracy loss and let the caller decide.

    Returns None when the PyTorch build has no quantisation backend, which is
    a property of the installation rather than of the model.
    """
    engine = quantisation_engine()
    if engine is None:
        return None
    torch.backends.quantized.engine = engine

    predictor.eval()
    with torch.no_grad():
        expected = predictor(example)

    quantised = torch.ao.quantization.quantize_dynamic(
        predictor, {nn.Linear}, dtype=torch.qint8
    )
    with torch.no_grad():
        actual = quantised(example)

    return ExportReport(
        path=Path("(in memory)"),
        format="quantised",
        max_difference=float((expected - actual).abs().max()),
        seconds_per_batch=measure_latency(quantised, example),
        note=f"engine={engine}",
    )


def build_predictor(
    run_dir: Path, device: torch.device | None = None
) -> tuple[FinetuneSpec, ParameterPredictor]:
    """Rebuild a trained arm as a traceable parameter predictor."""
    from cirrus.eval.compare import load_arm
    from cirrus.train.finetune import load_backbone

    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    spec = FinetuneSpec(**state["spec"])
    backbone, _ = load_backbone(spec.pretrained, freeze=True)
    target = device or torch.device("cpu")
    _, head = load_arm(run_dir, backbone.to(target), target)
    predictor = ParameterPredictor(
        backbone.to(target), head, spec.is_distributional
    ).eval()
    return spec, predictor
