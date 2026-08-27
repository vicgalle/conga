"""Small target interface used by CONGA and the released experiments."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import torch


DTYPE = torch.float64


def _flatten_named(named: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """Flatten named tensors into one NumPy array per constrained coordinate."""
    out: Dict[str, np.ndarray] = {}
    for base, value in named.items():
        array = value.detach().cpu().numpy()
        if array.ndim == 1:
            out[base] = array
        else:
            for index in range(array.shape[1]):
                out[f"{base}[{index + 1}]"] = array[:, index]
    return out


class TargetPosterior(ABC):
    """A differentiable log density on an unconstrained Euclidean space."""

    name: str
    dim: int

    @abstractmethod
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Return the unnormalized log density for ``value[..., dim]``."""

    @abstractmethod
    def constrain(self, value: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Name coordinates used in returned posterior draws."""

    def constrained_draws(self, value: torch.Tensor) -> Dict[str, np.ndarray]:
        return _flatten_named(self.constrain(value))

    @property
    def constrained_param_names(self) -> List[str]:
        probe = torch.zeros(1, self.dim, dtype=DTYPE)
        return list(_flatten_named(self.constrain(probe)))
