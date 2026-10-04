"""Public STED similarity API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .alignment import AlignmentEngine
from .json_tree import json_to_tree, node_descriptor
from .node_encoder import FrozenBertNodeEncoder, NodeEncoder


@dataclass(frozen=True)
class STEDResult:
    similarity: float
    distance: float
    source_nodes: int
    target_nodes: int


def compute_sted(
    source: Any,
    target: Any,
    *,
    encoder: NodeEncoder | None = None,
) -> STEDResult:
    """Compute STED using the exact normalization in the revised manuscript."""

    encoder = encoder or FrozenBertNodeEncoder()
    source_tree = json_to_tree(source)
    target_tree = json_to_tree(target)
    source_texts = [node_descriptor(source_tree, i) for i in range(len(source_tree))]
    target_texts = [node_descriptor(target_tree, i) for i in range(len(target_tree))]
    source_embeddings = encoder.encode(source_texts)
    target_embeddings = encoder.encode(target_texts)
    engine = AlignmentEngine(
        source_tree, target_tree, source_embeddings, target_embeddings
    )
    distance = engine.tree_distance()
    denominator = len(source_tree) + len(target_tree)
    similarity = 1.0 - distance / denominator
    similarity = float(np.clip(similarity, 0.0, 1.0))
    return STEDResult(similarity, distance, len(source_tree), len(target_tree))


def sted_similarity(
    source: Any,
    target: Any,
    *,
    encoder: NodeEncoder | None = None,
) -> float:
    return compute_sted(source, target, encoder=encoder).similarity
