"""JSON-specific recursive alignment for STED."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .costs import DELETION_COST, INSERTION_COST, substitution_cost
from .json_tree import JsonTree


class AlignmentEngine:
    """Compute the revised recursive STED dissimilarity between two trees."""

    def __init__(
        self,
        source_tree: JsonTree,
        target_tree: JsonTree,
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
    ) -> None:
        self.source_tree = source_tree
        self.target_tree = target_tree
        self.source_embeddings = np.asarray(source_embeddings)
        self.target_embeddings = np.asarray(target_embeddings)
        if len(self.source_embeddings) != len(source_tree):
            raise ValueError("source embedding count does not match source tree")
        if len(self.target_embeddings) != len(target_tree):
            raise ValueError("target embedding count does not match target tree")

    @lru_cache(maxsize=None)
    def deletion_cost(self, node: int) -> float:
        """D(v): delete the complete subtree rooted at source node v."""

        children = self.source_tree.nodes[node].children
        return DELETION_COST + sum(self.deletion_cost(child) for child in children)

    @lru_cache(maxsize=None)
    def insertion_cost(self, node: int) -> float:
        """I(u): insert the complete subtree rooted at target node u."""

        children = self.target_tree.nodes[node].children
        return INSERTION_COST + sum(self.insertion_cost(child) for child in children)

    @lru_cache(maxsize=None)
    def distance(self, source: int, target: int) -> float:
        """delta(v,u): semantic substitution plus child-forest alignment."""

        source_node = self.source_tree.nodes[source]
        target_node = self.target_tree.nodes[target]
        semantic = substitution_cost(
            self.source_embeddings[source], self.target_embeddings[target]
        )
        if source_node.kind == target_node.kind == "object":
            child_cost = self._align_objects(source_node.children, target_node.children)
        elif source_node.kind == target_node.kind == "array":
            child_cost = self._align_arrays(source_node.children, target_node.children)
        elif source_node.kind == target_node.kind == "scalar":
            child_cost = 0.0
        else:
            child_cost = sum(self.deletion_cost(child) for child in source_node.children)
            child_cost += sum(self.insertion_cost(child) for child in target_node.children)
        return semantic + child_cost

    def _align_arrays(self, source: Sequence[int], target: Sequence[int]) -> float:
        rows, columns = len(source), len(target)
        dp = np.zeros((rows + 1, columns + 1), dtype=np.float64)
        for i in range(1, rows + 1):
            dp[i, 0] = dp[i - 1, 0] + self.deletion_cost(source[i - 1])
        for j in range(1, columns + 1):
            dp[0, j] = dp[0, j - 1] + self.insertion_cost(target[j - 1])
        for i in range(1, rows + 1):
            for j in range(1, columns + 1):
                dp[i, j] = min(
                    dp[i - 1, j] + self.deletion_cost(source[i - 1]),
                    dp[i, j - 1] + self.insertion_cost(target[j - 1]),
                    dp[i - 1, j - 1] + self.distance(source[i - 1], target[j - 1]),
                )
        return float(dp[rows, columns])

    def _align_objects(self, source: Sequence[int], target: Sequence[int]) -> float:
        rows, columns = len(source), len(target)
        if rows == columns == 0:
            return 0.0
        size = rows + columns
        matrix = np.zeros((size, size), dtype=np.float64)
        for i, source_child in enumerate(source):
            for j, target_child in enumerate(target):
                matrix[i, j] = self.distance(source_child, target_child)
            matrix[i, columns:] = self.deletion_cost(source_child)
        for i in range(rows, size):
            for j, target_child in enumerate(target):
                matrix[i, j] = self.insertion_cost(target_child)
        row_indices, column_indices = linear_sum_assignment(matrix)
        return float(matrix[row_indices, column_indices].sum())

    def tree_distance(self) -> float:
        return self.distance(self.source_tree.root, self.target_tree.root)
