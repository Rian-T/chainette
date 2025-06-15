from __future__ import annotations
"""Fluent operator-based DSL (≤40 LOC).

Example::

    chain = (qa >> filter_node >> (fr_branch | es_branch)).to_chain("My Chain")
"""
from typing import List, Union
from chainette.core.chain import Chain
from chainette.core.branch import Branch
from chainette.core.node import Node

StepLike = Union[Node, List[Branch]]  # what Chain.steps accepts

__all__ = ["pipe"]


class _Pipe:  # noqa: D401 – tiny helper
    def __init__(self, parts: List[StepLike]):
        self.parts = parts

    # sequence operator >> -------------------------------------------------- #
    def __rshift__(self, other: "_Pipe | Node | List[Branch]" ) -> "_Pipe":
        if isinstance(other, _Pipe):
            return _Pipe(self.parts + other.parts)
        return _Pipe(self.parts + [other])

    # parallel operator | --------------------------------------------------- #
    def __or__(self, other: "_Pipe | Branch") -> "_Pipe":
        # Accept Branch or Pipe containing a Branch
        def _branches(obj):
            if isinstance(obj, Branch):
                return [obj]
            if isinstance(obj, _Pipe):
                # assume single branch inside
                assert len(obj.parts) == 1 and isinstance(obj.parts[0], Branch)
                return obj.parts
            raise TypeError("| expects Branch or Pipe[Branch]")

        combined: List[Branch] = _branches(self) + _branches(other)
        return _Pipe([combined])  # list of branches is one chain step

    # join ------------------------------------------------------------- #
    def join(self, id: str):  # noqa: D401
        """Mark contained Branch(es) as join nodes with *id*."""
        from .branch import Branch
        jb_parts: List[Branch] = []
        for p in self.parts:
            if isinstance(p, list):
                jb_parts.append(p)  # postpone for recursion
            elif isinstance(p, Branch):
                jb_parts.append(p)
        new_branches = [br.join(id) for br in jb_parts]
        return _Pipe([new_branches])

    # convert --------------------------------------------------------------- #
    def to_chain(self, name: str, **kwargs) -> Chain:  # noqa: D401
        return Chain(name=name, steps=self.parts, **kwargs)


# public constructor --------------------------------------------------------- #

def pipe(node: Node | Branch) -> _Pipe:  # noqa: D401
    """Return a DSL wrapper around *node*."""
    return _Pipe([node])

# --------------------------------------------------------------------------- #
# Monkey-patch Node to enable `step1 >> step2` style syntax out of the box.
# This keeps API additive – regular Chain construction still works.
# --------------------------------------------------------------------------- #

def _node_rshift(self: Node, other):  # type: ignore[override]
    from chainette.dsl import pipe as _pipe  # local import to avoid cycles
    return _pipe(self) >> other


def _node_or(self: Node, other):  # type: ignore[override]
    from chainette.dsl import pipe as _pipe  # local import
    return _pipe(self) | other


# Attach only once
if not hasattr(Node, "__rshift__"):
    setattr(Node, "__rshift__", _node_rshift)
    setattr(Node, "__or__", _node_or) 