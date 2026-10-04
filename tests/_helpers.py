from __future__ import annotations

from typing import Sequence

import numpy as np


class IdentityTextEncoder:
    """Assign a stable orthogonal vector to each distinct descriptor."""

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension
        self.indices: dict[str, int] = {}

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            if text not in self.indices:
                if len(self.indices) >= self.dimension:
                    raise RuntimeError("test encoder dimension exhausted")
                self.indices[text] = len(self.indices)
            matrix[row, self.indices[text]] = 1.0
        return matrix
