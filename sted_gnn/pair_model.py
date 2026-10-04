"""Exchange-invariant STED-GNN pair representation and scoring head."""

from __future__ import annotations

import torch
from torch import nn


def pair_representation(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """Return [|z_a-z_b|; z_a*z_b], which is invariant to pair exchange."""

    if z_a.shape != z_b.shape:
        raise ValueError("document embedding shapes must match")
    return torch.cat((torch.abs(z_a - z_b), z_a * z_b), dim=-1)


class PairScoringHead(nn.Module):
    """Manuscript MLP: 1024 -> 512 -> 1 with a sigmoid output."""

    def __init__(self, document_dim: int = 512) -> None:
        super().__init__()
        self.document_dim = int(document_dim)
        self.network = nn.Sequential(
            nn.Linear(self.document_dim * 2, self.document_dim),
            nn.ReLU(),
            nn.Linear(self.document_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        return self.network(pair_representation(z_a, z_b)).squeeze(-1)
