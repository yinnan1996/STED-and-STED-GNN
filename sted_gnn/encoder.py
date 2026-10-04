"""Shared two-layer GCN document encoder from the revised manuscript."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool


class SharedDocumentEncoder(nn.Module):
    """Map 768-dimensional BERT node features to a 512-dimensional document."""

    def __init__(self, input_dim: int = 768, document_dim: int = 512) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.document_dim = int(document_dim)
        self.gcn1 = GCNConv(self.input_dim, self.document_dim)
        self.gcn2 = GCNConv(self.document_dim, self.document_dim)
        self.activation = nn.ReLU()

    def forward(self, graph: Data) -> torch.Tensor:
        features = self.activation(self.gcn1(graph.x, graph.edge_index))
        features = self.activation(self.gcn2(features, graph.edge_index))
        batch = getattr(graph, "batch", None)
        if batch is None:
            batch = torch.zeros(features.size(0), dtype=torch.long, device=features.device)
        return global_mean_pool(features, batch)
