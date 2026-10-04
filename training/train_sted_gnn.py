"""Minimal STED-GNN training entry point with fixed source-disjoint pair files."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

from sted.node_encoder import FrozenBertNodeEncoder
from sted_gnn.graph_builder import build_document_graph
from sted_gnn.model import STEDGNN


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_pair_partitions(partitions: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Reject source-document or normalized-content leakage between pair files."""

    id_fields = ("ref_source_document_id", "perturbed_source_document_id")
    hash_fields = ("ref_content_hash", "perturbed_content_hash")
    ids: dict[str, set[str]] = {}
    hashes: dict[str, set[str]] = {}
    for split, rows in partitions.items():
        missing = [
            (index, field)
            for index, row in enumerate(rows)
            for field in (*id_fields, *hash_fields)
            if not row.get(field)
        ]
        if missing:
            raise ValueError(f"{split} is missing required leakage-audit fields: {missing[:5]}")
        ids[split] = {str(row[field]) for row in rows for field in id_fields}
        hashes[split] = {str(row[field]) for row in rows for field in hash_fields}
    report = {"source_id_overlap": {}, "content_hash_overlap": {}}
    names = list(partitions)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            label = f"{left}__{right}"
            report["source_id_overlap"][label] = len(ids[left] & ids[right])
            report["content_hash_overlap"][label] = len(hashes[left] & hashes[right])
            if report["source_id_overlap"][label]:
                raise AssertionError(f"source-document leakage between {left} and {right}")
            if report["content_hash_overlap"][label]:
                raise AssertionError(f"normalized-content leakage between {left} and {right}")
    return report


class PairDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], node_encoder: FrozenBertNodeEncoder) -> None:
        self.rows = rows
        self.graphs: dict[str, Any] = {}
        for row in rows:
            for document in (row["ref"], row["perturbed"]):
                key = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                if key not in self.graphs:
                    self.graphs[key] = build_document_graph(document, node_encoder)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        key_a = json.dumps(row["ref"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        key_b = json.dumps(row["perturbed"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return self.graphs[key_a], self.graphs[key_b], float(row["similarity"])


def collate(batch):
    graph_a, graph_b, targets = zip(*batch)
    return (
        Batch.from_data_list(list(graph_a)),
        Batch.from_data_list(list(graph_b)),
        torch.tensor(targets, dtype=torch.float32),
    )


def evaluate(model: STEDGNN, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    criterion = nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for graph_a, graph_b, targets in loader:
            predictions = model(graph_a.to(device), graph_b.to(device))
            total += float(criterion(predictions, targets.to(device)).item())
            count += len(targets)
    return total / max(count, 1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train STED-GNN from pre-generated source-disjoint pair partitions."
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    partitions = {
        "train": load_jsonl(args.train),
        "validation": load_jsonl(args.validation),
        "test": load_jsonl(args.test),
    }
    audit = audit_pair_partitions(partitions)
    node_encoder = FrozenBertNodeEncoder(device=str(device))
    datasets = {name: PairDataset(rows, node_encoder) for name, rows in partitions.items()}
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(name == "train"),
            collate_fn=collate,
        )
        for name, dataset in datasets.items()
    }

    model = STEDGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    best_validation = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for graph_a, graph_b, targets in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(graph_a.to(device), graph_b.to(device))
            loss = criterion(predictions, targets.to(device))
            loss.backward()
            optimizer.step()
        validation_mse = evaluate(model, loaders["validation"], device)
        print(f"epoch={epoch} validation_mse={validation_mse:.8f}")
        if validation_mse < best_validation:
            best_validation = validation_mse
            best_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("training did not produce a model state")
    model.load_state_dict(best_state)
    test_mse = evaluate(model, loaders["test"], device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "config": {
                "node_encoder": "bert-base-multilingual-cased",
                "node_pooling": "cls",
                "input_dim": 768,
                "document_dim": 512,
                "gcn_layers": 2,
                "pair_representation": "absolute_difference_and_elementwise_product",
                "loss": "mse",
            },
            "validation_mse": best_validation,
            "test_mse": test_mse,
            "split_audit": audit,
        },
        output,
    )
    print(f"saved={output} test_mse={test_mse:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
