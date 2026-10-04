from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sted import compute_sted


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute STED for two JSON documents.")
    parser.add_argument("document_a")
    parser.add_argument("document_b")
    args = parser.parse_args()
    document_a = json.loads(Path(args.document_a).read_text(encoding="utf-8"))
    document_b = json.loads(Path(args.document_b).read_text(encoding="utf-8"))
    result = compute_sted(document_a, document_b)
    print(json.dumps(result.__dict__, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
