"""Construct the bidirectional document graph consumed by STED-GNN."""

from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import Data

from sted.json_tree import json_to_tree, node_descriptor
from sted.node_encoder import NodeEncoder


def build_document_graph(document: Any, node_encoder: NodeEncoder) -> Data:
    """Encode a JSON tree using the same Key+Path representation as STED."""

    tree = json_to_tree(document)
    texts = [node_descriptor(tree, index) for index in range(len(tree))]
    features = torch.from_numpy(node_encoder.encode(texts)).to(dtype=torch.float32)
    sources: list[int] = []
    targets: list[int] = []
    for node in tree.nodes:
        for child in node.children:
            sources.extend((node.index, child))
            targets.extend((child, node.index))
    if sources:
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return Data(x=features, edge_index=edge_index)
