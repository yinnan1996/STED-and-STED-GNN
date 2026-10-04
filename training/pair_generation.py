"""Generate supervision pairs independently within source partitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .source_split import SPLIT_NAMES, normalized_content_sha256


VariantFactory = Callable[[Any, str], Iterable[Any]]
ScoreFunction = Callable[[Any, Any], float]


def generate_pairs_within_partitions(
    partitions: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    variant_factory: VariantFactory,
    score_function: ScoreFunction,
) -> dict[str, list[dict[str, Any]]]:
    """Generate variants and STED targets only after source partitioning.

    The callback receives one JSON document and its source ID. It must not draw
    documents from another partition. This function deliberately has no
    pair-level split operation.
    """

    result: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    for split in SPLIT_NAMES:
        pair_index = 0
        for source in partitions.get(split, []):
            source_id = str(source["source_document_id"])
            document = source["json"]
            for variant in variant_factory(document, source_id):
                result[split].append(
                    {
                        "pair_id": f"{split}_pair_{pair_index:07d}",
                        "split": split,
                        "ref": document,
                        "perturbed": variant,
                        "ref_source_document_id": source_id,
                        "perturbed_source_document_id": source_id,
                        "ref_content_hash": normalized_content_sha256(document),
                        "perturbed_content_hash": normalized_content_sha256(variant),
                        "similarity": float(score_function(document, variant)),
                    }
                )
                pair_index += 1
    return result


def write_pair_partitions(
    output_directory: str | Path,
    pairs: Mapping[str, Iterable[Mapping[str, Any]]],
) -> None:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_NAMES:
        with (directory / f"{split}_pairs.jsonl").open("w", encoding="utf-8") as handle:
            for row in pairs.get(split, []):
                handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
