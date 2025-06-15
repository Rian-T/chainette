from __future__ import annotations
"""Static validators for Chainette graphs (I/O compatibility, cycles…)."""
from typing import List, Tuple, Type
from pydantic import BaseModel

from chainette.core.node import Node
from chainette.core.step import Step
from chainette.core.apply import ApplyNode
from chainette.core.branch import Branch

__all__ = ["validate_chain_io", "Incompat"]

Incompat = Tuple[str, Type[BaseModel] | None, Type[BaseModel] | None]


def _node_io(node: Node):
    if isinstance(node, Step):
        return node.input_model, node.output_model
    if isinstance(node, ApplyNode):
        return getattr(node, "input_model", None), getattr(node, "output_model", None)
    # Branch not supported directly here (handled in recursion)
    return None, None


def validate_chain_io(chain) -> List[Incompat]:  # noqa: D401
    """Return list of (node_id, expected, actual) mismatches in *chain*."""
    mismatches: List[Incompat] = []

    prev_out: Type[BaseModel] | None = None

    for item in chain.steps:
        if isinstance(item, list):
            # parallel branches share same input type
            for br in item:
                _walk_branch(br, prev_out, mismatches)
            continue

        _check_node(item, prev_out, mismatches)
        _, prev_out = _node_io(item)

    return mismatches


def _walk_branch(branch: Branch, incoming, mismatches):
    prev = incoming
    for n in branch.steps:
        _check_node(n, prev, mismatches)
        _, prev = _node_io(n)


def _check_node(node: Node, expected_in, mismatches):
    in_t, out_t = _node_io(node)
    if expected_in and in_t and expected_in != in_t:
        mismatches.append((node.id, expected_in, in_t)) 