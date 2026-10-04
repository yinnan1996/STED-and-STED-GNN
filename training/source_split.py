"""Split source documents before generating variants or document pairs."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


SPLIT_NAMES = ("train", "validation", "test")


def normalized_json(document: Any) -> str:
    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def normalized_content_sha256(document: Any) -> str:
    return hashlib.sha256(normalized_json(document).encode("utf-8")).hexdigest()


def split_source_documents(
    records: Iterable[Mapping[str, Any]],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, list[dict[str, Any]]]:
    """Partition sources while keeping equal normalized content together."""

    records = [dict(record) for record in records]
    if not records:
        raise ValueError("at least one source document is required")
    ids = [str(record["source_document_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("source_document_id values must be globally unique")
    if len(ratios) != 3 or any(value < 0 for value in ratios) or sum(ratios) <= 0:
        raise ValueError("ratios must contain three non-negative values with a positive sum")
    ratio_sum = sum(ratios)
    normalized_ratios = [value / ratio_sum for value in ratios]

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        digest = normalized_content_sha256(record["json"])
        record["normalized_content_sha256"] = digest
        groups[digest].append(record)
    grouped = list(groups.values())
    random.Random(seed).shuffle(grouped)

    target_counts = [len(records) * ratio for ratio in normalized_ratios]
    partitions: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    counts = [0, 0, 0]
    for group in grouped:
        candidates = [i for i, ratio in enumerate(normalized_ratios) if ratio > 0]
        chosen = min(
            candidates,
            key=lambda i: (counts[i] / target_counts[i] if target_counts[i] else float("inf"), i),
        )
        partitions[SPLIT_NAMES[chosen]].extend(group)
        counts[chosen] += len(group)

    audit_source_partitions(partitions)
    return partitions


def audit_source_partitions(
    partitions: Mapping[str, Iterable[Mapping[str, Any]]]
) -> dict[str, Any]:
    """Assert zero source-ID and normalized-content overlap across partitions."""

    ids: dict[str, set[str]] = {}
    hashes: dict[str, set[str]] = {}
    for name in SPLIT_NAMES:
        rows = [dict(row) for row in partitions.get(name, [])]
        ids[name] = {str(row["source_document_id"]) for row in rows}
        hashes[name] = {
            str(row.get("normalized_content_sha256") or normalized_content_sha256(row["json"]))
            for row in rows
        }
        if len(ids[name]) != len(rows):
            raise AssertionError(f"duplicate source-document IDs inside {name}")

    overlap_report: dict[str, dict[str, int]] = {"source_ids": {}, "content_hashes": {}}
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            label = f"{left}__{right}"
            id_overlap = ids[left] & ids[right]
            hash_overlap = hashes[left] & hashes[right]
            overlap_report["source_ids"][label] = len(id_overlap)
            overlap_report["content_hashes"][label] = len(hash_overlap)
            if id_overlap:
                raise AssertionError(f"source-document ID overlap in {label}: {sorted(id_overlap)}")
            if hash_overlap:
                raise AssertionError(f"normalized-content hash overlap in {label}")
    return {
        "counts": {name: len(ids[name]) for name in SPLIT_NAMES},
        "overlaps": overlap_report,
    }


def write_partitions(output_directory: str | Path, partitions: Mapping[str, Iterable[Mapping[str, Any]]]) -> None:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    audit = audit_source_partitions(partitions)
    for name in SPLIT_NAMES:
        with (directory / f"{name}_sources.jsonl").open("w", encoding="utf-8") as handle:
            for row in partitions.get(name, []):
                handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    (directory / "split_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
