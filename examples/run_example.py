from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sted import compute_sted


directory = Path(__file__).resolve().parent
document_a = json.loads((directory / "example_a.json").read_text(encoding="utf-8"))
document_b = json.loads((directory / "example_b.json").read_text(encoding="utf-8"))
print(compute_sted(document_a, document_b))
