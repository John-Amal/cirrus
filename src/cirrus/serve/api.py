"""HTTP inference service.

Serves the exported TorchScript predictor. The model returns the parameters
of a censored shifted gamma per grid cell; this turns those into the things
a caller actually wants -- an expected value, a quantile, the probability of
exceeding a threshold.

Those are computed **analytically** here. Training needed sample-based CRPS
because the incomplete gamma function has no implemented derivative with
respect to its shape parameter, so a likelihood could not be backpropagated
through. Inference has no such constraint: there are no gradients, so the
closed forms are available and exact. The sampling in training was a
workaround for autograd, not a property of the distribution.

The service takes fields in physical units and normalises them internally
using the statistics the model was trained with. Requiring callers to
normalise would export a detail they cannot verify and would silently
produce nonsense if they used different statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scipy.stats import gamma as gamma_distribution

from cirrus.data.ingest import IngestSpec
from cirrus.data.normalise import Normaliser, NormaliseSpec
from cirrus.data.thresholds import Thresholds
from cirrus.data.windows import FORCING_CHANNELS, time_encodings

DEFAULT_QUANTILES = (0.5, 0.9, 0.99)


class PredictRequest(BaseModel):
    """Fields for one forecast, in physical units."""

    fields: list[list[list[list[float]]]] = Field(
        ...,
        description=(
            "Nested array (time, channel, lat, lon) of the dynamic and static "
            "channels in physical units; see /info for the expected order."
        ),
    )
    timestamps: list[str] = Field(
        ...,
        description="ISO timestamp per input step, used for the time encodings.",
    )
    quantiles: list[float] = Field(
        default=list(DEFAULT_QUANTILES),
        description="Quantiles of the predictive distribution to return.",
    )
    threshold_mm: float | None = Field(
        default=None,
        description=(
            "Exceedance probability is computed against this, or against each "
            "cell's climatological threshold when omitted."
        ),
    )


class PredictResponse(BaseModel):
    """Per-cell predictive distribution and the quantities derived from it."""

    grid: list[int]
    mean_mm: list[list[float]]
    probability_of_rain: list[list[float]]
    exceedance_probability: list[list[float]]
    quantiles_mm: dict[str, list[list[float]]]
    parameters: dict[str, list[list[float]]]


@dataclass
class ServiceState:
    """Everything loaded once at startup."""

    model: torch.jit.ScriptModule
    normaliser: Normaliser
    data_spec: IngestSpec
    thresholds: np.ndarray
    arm: str

    @property
    def dynamic_channels(self) -> tuple[str, ...]:
        """Names of the per-step dynamic channels, in order."""
        return self.data_spec.time_channels

    @property
    def channel_order(self) -> list[str]:
        """Channels the caller must supply per timestep, in order."""
        return [*self.dynamic_channels, *self.data_spec.static_variables]


def load_state(
    model_path: str | Path,
    arm: str = "twcrps_p90",
    data_config: str | Path = "configs/data/era5_5625.yaml",
    normalise_config: str | Path = "configs/data/normalise.yaml",
    thresholds_path: str | Path = "data/stats/thresholds_train.json",
) -> ServiceState:
    """Load the exported model and the statistics it was trained with."""
    data_spec = IngestSpec.from_yaml(data_config)
    normaliser = Normaliser.load(NormaliseSpec.from_yaml(normalise_config).output)
    thresholds = Thresholds.load(thresholds_path).values
    model = torch.jit.load(str(model_path))
    model.eval()
    return ServiceState(model, normaliser, data_spec, thresholds, arm)


def build_input(state: ServiceState, request: PredictRequest) -> torch.Tensor:
    """Normalise the caller's fields and append the time encodings.

    Mirrors the dataset exactly: dynamic and static channels normalised with
    the training statistics, then four forcing channels per step derived from
    the timestamp, then time folded into channels.
    """
    supplied = np.asarray(request.fields, dtype=np.float32)
    n_steps = supplied.shape[0]
    expected = len(state.channel_order)
    if supplied.ndim != 4 or supplied.shape[1] != expected:
        raise HTTPException(
            status_code=422,
            detail=(
                f"expected (time, {expected}, lat, lon); got {list(supplied.shape)}"
            ),
        )
    if len(request.timestamps) != n_steps:
        raise HTTPException(
            status_code=422,
            detail=f"{n_steps} input steps but {len(request.timestamps)} timestamps",
        )

    normalised = state.normaliser.subset(state.channel_order).normalise(supplied)
    times = np.array(request.timestamps, dtype="datetime64[ns]")
    encodings = time_encodings(times)  # (steps, 4)
    height, width = supplied.shape[-2:]
    forcings = np.broadcast_to(
        encodings[:, :, None, None], (n_steps, len(FORCING_CHANNELS), height, width)
    )

    stacked = np.concatenate([normalised, forcings], axis=1)
    folded = stacked.reshape(1, n_steps * stacked.shape[1], height, width)
    return torch.from_numpy(np.ascontiguousarray(folded, dtype=np.float32))


def derive(
    parameters: np.ndarray,
    thresholds: np.ndarray,
    quantiles: list[float],
    threshold_mm: float | None,
) -> dict[str, Any]:
    """Turn (shape, scale, shift) into the quantities a caller wants.

    All closed form. For ``Y = max(0, G + shift)`` with ``G`` gamma:
    ``P(Y = 0) = F_G(-shift)``, the quantile is ``max(0, F_G^-1(p) + shift)``,
    and ``P(Y > u) = 1 - F_G(u - shift)``.
    """
    shape, scale, shift = parameters
    frozen = gamma_distribution(a=shape, scale=scale)

    dry = frozen.cdf(-shift)
    limit = (
        thresholds if threshold_mm is None else np.full_like(thresholds, threshold_mm)
    )
    exceedance = frozen.sf(limit - shift)

    # The censored mean is not shape*scale: everything below zero is piled at
    # zero. Integrating the positive part gives the expression below. Note the
    # first term needs shape+1, so it cannot use the frozen distribution --
    # a frozen scipy distribution will not accept new parameters.
    mean = shape * scale * gamma_distribution.sf(
        -shift, a=shape + 1, scale=scale
    ) + shift * frozen.sf(-shift)

    return {
        "probability_of_rain": (1.0 - dry),
        "exceedance_probability": exceedance,
        "mean_mm": np.clip(mean, 0.0, None),
        "quantiles_mm": {
            f"{q:g}": np.clip(frozen.ppf(q) + shift, 0.0, None) for q in quantiles
        },
    }


def create_app(state: ServiceState) -> FastAPI:
    """Build the application around an already-loaded model."""
    app = FastAPI(
        title="cirrus",
        description=(
            "Probabilistic precipitation forecasts from a small weather "
            "foundation model. Returns a predictive distribution per grid "
            "cell, not a single number."
        ),
        version="0.1.0",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        """Report that the service is up."""
        return {"status": "ok", "arm": state.arm}

    @app.get("/info")
    def info() -> dict[str, Any]:
        """Describe the inputs the model expects and what it returns."""
        return {
            "arm": state.arm,
            "grid": list(state.thresholds.shape),
            "channels_per_step": state.channel_order,
            "forcing_channels": list(FORCING_CHANNELS),
            "note": (
                "Supply channels_per_step in physical units; the four forcing "
                "channels are derived from the timestamps server-side."
            ),
            "outputs": ["shape", "scale", "shift"],
            "distribution": (
                "censored shifted gamma: Y = max(0, Gamma(shape, scale) + shift)"
            ),
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        """Return the predictive distribution six hours ahead."""
        if not all(0.0 < q < 1.0 for q in request.quantiles):
            raise HTTPException(status_code=422, detail="quantiles must be in (0, 1)")

        tensor = build_input(state, request)
        with torch.no_grad():
            parameters = state.model(tensor)[0].numpy()

        derived = derive(
            parameters, state.thresholds, request.quantiles, request.threshold_mm
        )
        return PredictResponse(
            grid=list(parameters.shape[1:]),
            mean_mm=derived["mean_mm"].tolist(),
            probability_of_rain=derived["probability_of_rain"].tolist(),
            exceedance_probability=derived["exceedance_probability"].tolist(),
            quantiles_mm={
                name: values.tolist()
                for name, values in derived["quantiles_mm"].items()
            },
            parameters={
                "shape": parameters[0].tolist(),
                "scale": parameters[1].tolist(),
                "shift": parameters[2].tolist(),
            },
        )

    return app
