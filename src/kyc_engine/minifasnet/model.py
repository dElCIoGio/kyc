"""Lazy PyTorch checkpoint loading for MiniFASNet inference."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .checkpoint import MiniFASNetModelSpec
from .errors import MiniFASNetInitializationError, MiniFASNetInputError
from .preprocessing import tensor_values


class LoadedMiniFASNetModel:
    """A loaded, eval-mode model that returns one softmax probability vector."""

    def __init__(self, model: Any, torch: Any, device: str) -> None:
        self._model = model
        self._torch = torch
        self._device = device

    def predict(self, frame: np.ndarray) -> np.ndarray:
        values = tensor_values(frame)
        try:
            tensor = self._torch.from_numpy(values).unsqueeze(0).to(self._device)
            with self._torch.inference_mode():
                probabilities = self._torch.softmax(self._model(tensor), dim=1)
            return probabilities.detach().cpu().numpy()[0]
        except (RuntimeError, ValueError) as exc:
            raise MiniFASNetInputError("MiniFASNet could not evaluate the face region") from exc


def load_model(spec: MiniFASNetModelSpec, device: str = "cpu") -> LoadedMiniFASNetModel:
    """Construct and strictly load a supported checkpoint once, on the selected CPU."""
    if device != "cpu":
        raise MiniFASNetInitializationError("MiniFASNet currently supports the CPU device only")
    try:
        import torch
        from .model_runtime import build_model
    except ModuleNotFoundError as exc:
        raise MiniFASNetInitializationError(
            "MiniFASNet requires the optional liveness dependency"
        ) from exc
    try:
        model = build_model(spec.architecture, spec.input_height, spec.input_width)
        try:
            checkpoint = torch.load(spec.path, map_location=device, weights_only=True)
        except TypeError:  # Compatibility with PyTorch releases before weights_only.
            checkpoint = torch.load(spec.path, map_location=device)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("checkpoint is not a state dictionary")
        state_dict = {
            str(key).removeprefix("module."): value
            for key, value in checkpoint.items()
        }
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise MiniFASNetInitializationError("MiniFASNet checkpoint is incompatible") from exc
    return LoadedMiniFASNetModel(model, torch, device)
