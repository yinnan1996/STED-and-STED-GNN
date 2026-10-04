"""Convert JSON values to the rooted tree used by STED.

Tree nodes retain keys, root-to-node paths, and container types. Scalar values
are intentionally not stored because they are outside the current STED node
representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


def _kind(value: Any) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return "scalar"


@dataclass
class TreeNode:
    index: int
    key: str
    path: tuple[str, ...]
    kind: str
    parent: int | None
    children: list[int] = field(default_factory=list)


@dataclass
class JsonTree:
    nodes: list[TreeNode]
    root: int = 0

    def __len__(self) -> int:
        return len(self.nodes)


def json_to_tree(document: Any) -> JsonTree:
    """Build a JSON tree without retaining raw scalar values."""

    nodes: list[TreeNode] = [
        TreeNode(
            index=0,
            key="$ROOT$",
            path=("$ROOT$",),
            kind=_kind(document),
            parent=None,
        )
    ]

    def add_node(key: str, value: Any, parent: int) -> int:
        index = len(nodes)
        path = nodes[parent].path + (key,)
        nodes.append(
            TreeNode(
                index=index,
                key=key,
                path=path,
                kind=_kind(value),
                parent=parent,
            )
        )
        nodes[parent].children.append(index)
        build(index, value)
        return index

    def build(parent: int, value: Any) -> None:
        if isinstance(value, dict):
            items: Iterable[tuple[str, Any]] = value.items()
            for key, child in items:
                add_node(str(key), child, parent)
        elif isinstance(value, list):
            for position, child in enumerate(value):
                add_node(f"[{position}]", child, parent)

    build(0, document)
    return JsonTree(nodes=nodes)


def node_descriptor(tree: JsonTree, node_index: int) -> str:
    """Return the manuscript's Key+Path descriptor for one node."""

    node = tree.nodes[node_index]
    return f"key={node.key}\npath={'/'.join(node.path)}"
