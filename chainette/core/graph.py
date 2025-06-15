from __future__ import annotations
"""Light-weight DAG utilities (experimental).

This module is **NOT** used by the current Chainette runtime yet.
It is introduced as part of the *elegance* refactor to model execution
as an explicit directed acyclic graph while keeping the codebase tiny.

LOC budget: ≤120 – keep it minimal.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Iterator, Optional, Set

__all__ = ["GraphNode", "GraphEdge", "Graph"]


@dataclass
class GraphNode:  # noqa: D101 – tiny data holder
    id: str
    ref: object  # Pointer to the actual Step / Apply / etc.
    downstream: List["GraphNode"] = field(default_factory=list)

    # Convenience --------------------------------------------------------- #
    def connect(self, *targets: "GraphNode") -> "GraphNode":
        """Add *targets* to *self.downstream* and return *self* to allow chaining."""
        self.downstream.extend(targets)
        return self

    # Iteration ----------------------------------------------------------- #
    def __iter__(self) -> Iterator["GraphNode"]:  # depth-first iteration
        yield self
        for child in self.downstream:
            yield from child


@dataclass
class GraphEdge:  # noqa: D101
    src: GraphNode
    dst: GraphNode


class Graph:  # noqa: D101
    def __init__(self, roots: Optional[List[GraphNode]] = None):
        self.roots: List[GraphNode] = roots or []

    # -------------------------------------------------- #
    def add_root(self, node: GraphNode):
        self.roots.append(node)

    # -------------------------------------------------- #
    def nodes(self) -> List[GraphNode]:
        """Return nodes in depth-first order (duplicates removed)."""
        seen: Set[str] = set()
        ordered: List[GraphNode] = []
        for root in self.roots:
            for n in root:
                if n.id not in seen:
                    seen.add(n.id)
                    ordered.append(n)
        return ordered

    # -------------------------------------------------- #
    def edges(self) -> List[GraphEdge]:
        es: List[GraphEdge] = []
        for node in self.nodes():
            for child in node.downstream:
                es.append(GraphEdge(node, child))
        return es

    # -------------------------------------------------- #
    def validate_dag(self) -> None:
        """Ensure the graph has no cycles (simple DFS). Raises ValueError otherwise."""
        visited: Set[str] = set()
        stack: Set[str] = set()

        def _visit(n: GraphNode):
            if n.id in stack:
                raise ValueError(f"Cycle detected at node '{n.id}'")
            if n.id in visited:
                return
            stack.add(n.id)
            for c in n.downstream:
                _visit(c)
            stack.remove(n.id)
            visited.add(n.id)

        for r in self.roots:
            _visit(r) 