from __future__ import annotations
"""CompositeNode – minimal container node for other nodes.

Keeps the runtime compatible while enabling DAG-based refactor.
LOC:  ~25.
"""
from typing import List
from .node import Node

__all__ = ["CompositeNode"]


class CompositeNode(Node):  # noqa: D101 – tiny helper
    def __init__(self, *, id: str, name: str, steps: List[Node]):
        self.id = id
        self.name = name
        self.steps = steps 