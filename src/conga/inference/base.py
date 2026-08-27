"""The interface every inference algorithm shares.

The key extensibility point of this package: every algorithm implements
:meth:`InferenceAlgorithm.fit(target) -> InferenceResult`, and an
``InferenceResult`` knows how to ``sample`` constrained draws. The evaluation
harness only ever talks to these two objects, so adding a new method (NUTS,
Laplace approximation, normalizing-flow VI, ...) means writing one new subclass
-- nothing else in the codebase changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from ..core import TargetPosterior


def _make_bar(progress: object, total: int, default_desc: str):
    """A tqdm step-bar (inner, position 1) when ``progress`` is truthy, else None.

    ``progress`` may be ``True`` or a string used as the bar description.
    """
    if not progress:
        return None
    from tqdm.auto import tqdm
    desc = progress if isinstance(progress, str) else default_desc
    return tqdm(total=total, desc=desc, position=1, leave=False, unit="step")


@dataclass
class InferenceResult:
    """Outcome of running an algorithm on a target.

    ``sample(n)`` returns named *constrained* draws (dict name -> array[n]),
    directly comparable to named reference draws.
    """

    algorithm: str
    target_name: str
    _draw_fn: callable                       # n -> dict[str, np.ndarray]
    runtime_s: float = 0.0
    diagnostics: Dict[str, object] = field(default_factory=dict)

    def sample(self, n: int) -> Dict[str, np.ndarray]:
        return self._draw_fn(n)


class InferenceAlgorithm(ABC):
    name: str

    @abstractmethod
    def fit(self, target: TargetPosterior, seed: int = 0,
            progress: object = False) -> InferenceResult:
        """``progress`` may be False, True, or a str description for a tqdm bar."""
        ...
