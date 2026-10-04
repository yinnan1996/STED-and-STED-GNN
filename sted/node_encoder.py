"""Frozen multilingual BERT node encoder used by STED and STED-GNN."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class NodeEncoder(Protocol):
    """Minimal encoder interface accepted by the STED implementation."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one floating-point row per input text."""


class FrozenBertNodeEncoder:
    """Encode Key+Path text with frozen `bert-base-multilingual-cased` CLS."""

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        *,
        max_length: int = 256,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        if not texts:
            return np.empty((0, 768), dtype=np.float32)
        batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                tokens = self.tokenizer(
                    list(texts[start : start + self.batch_size]),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tokens = {name: tensor.to(self.device) for name, tensor in tokens.items()}
                hidden = self.model(**tokens).last_hidden_state[:, 0, :]
                batches.append(hidden.detach().cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(batches, axis=0)
