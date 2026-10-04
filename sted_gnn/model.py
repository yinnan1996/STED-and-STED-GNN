"""Complete STED-GNN model."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data

from .encoder import SharedDocumentEncoder
from .pair_model import PairScoringHead


class STEDGNN(nn.Module):
    """Shared document encoder followed by the symmetric scoring head."""

    def __init__(self, input_dim: int = 768, document_dim: int = 512) -> None:
        super().__init__()
        self.encoder = SharedDocumentEncoder(input_dim, document_dim)
        self.scoring_head = PairScoringHead(document_dim)

    def encode(self, graph: Data) -> torch.Tensor:
        return self.encoder(graph)

    def score_embeddings(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        return self.scoring_head(z_a, z_b)

    def forward(self, graph_a: Data, graph_b: Data) -> torch.Tensor:
        return self.score_embeddings(self.encode(graph_a), self.encode(graph_b))
