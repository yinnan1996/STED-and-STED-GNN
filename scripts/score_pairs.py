from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch_geometric.data import Batch

from sted.node_encoder import FrozenBertNodeEncoder
from sted_gnn import STEDGNN, build_document_graph


def main() -> int:
    parser = argparse.ArgumentParser(description="Score JSONL pairs with STED-GNN.")
    parser.add_argument("input")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = STEDGNN().to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    encoder = FrozenBertNodeEncoder(device=str(device))
    with Path(args.input).open("r", encoding="utf-8") as source, Path(args.output).open(
        "w", encoding="utf-8"
    ) as target:
        with torch.no_grad():
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                graph_a = Batch.from_data_list([build_document_graph(row["doc_a"], encoder)]).to(device)
                graph_b = Batch.from_data_list([build_document_graph(row["doc_b"], encoder)]).to(device)
                score = float(model(graph_a, graph_b).item())
                target.write(json.dumps({"pair_id": row.get("pair_id"), "score": score}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
