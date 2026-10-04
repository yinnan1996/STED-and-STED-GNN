"""Edit costs from the revised STED definition."""

from __future__ import annotations

import numpy as np


INSERTION_COST = 1.0
DELETION_COST = 1.0


def substitution_cost(source: np.ndarray, target: np.ndarray) -> float:
    """Return 1 - cosine(source, target), without the historical /2 factor."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    denominator = float(np.linalg.norm(source) * np.linalg.norm(target))
    if denominator == 0.0:
        raise ValueError("STED substitution cost is undefined for a zero embedding")
    cosine = float(np.dot(source, target) / denominator)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return 1.0 - cosine
