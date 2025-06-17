from __future__ import annotations

"""DAG helpers (no side-effects, ≤ 70 LOC).

iter_nodes(chain) yields (depth, obj) depth-first.
build_rich_tree(chain) returns a Rich *Tree* ready for printing.
"""
from typing import Iterator, List, Tuple, Any

from chainette import Chain  # public import exposes core types via __all__
from chainette.core.branch import Branch, JoinBranch
from chainette.core.node import Node

__all__ = [
    "iter_nodes",
    "build_rich_tree",
]

# --------------------------------------------------------------------------- #
# Core traverser
# --------------------------------------------------------------------------- #

def iter_nodes(chain: Chain) -> Iterator[Tuple[int, Any]]:  # noqa: D401
    """Yield *(depth, obj)* for every node/branch in *chain* (DFS)."""

    def _walk(objs: List[Any], depth: int):
        for obj in objs:
            # Parallel branches are represented as a list on the top level.
            if isinstance(obj, list):  # list[Branch]
                yield depth, obj  # pseudo-node for the parallel group
                for br in obj:  # type: Branch
                    yield depth + 1, br
                    _walk(br.steps, depth + 2)
            elif isinstance(obj, Branch):
                yield depth, obj
                _walk(obj.steps, depth + 1)
            else:  # Step, ApplyNode, JoinBranch, etc.
                yield depth, obj

    yield from _walk(chain.steps, 0)


# --------------------------------------------------------------------------- #
# Rich-aware tree builder (import lazily to avoid hard dep at import time)
# --------------------------------------------------------------------------- #

def build_rich_tree(chain: Chain):  # noqa: D401 – return type is Tree but avoid import
    """Return a *rich.tree.Tree* visualisation of *chain* (side-effect-free)."""
    from rich.tree import Tree  # local import keeps this module lightweight

    tree = Tree("[bold]Execution DAG[/]")

    def _add(parent: "Tree", objs):
        for obj in objs:
            if isinstance(obj, list):  # parallel branches wrapper
                p = parent.add("[dim]parallel ⨉ %d[/]" % len(obj))
                for br in obj:
                    bnode = p.add(f"[magenta]{br.name}[/]")
                    _add(bnode, br.steps)
            elif isinstance(obj, Branch):
                bnode = parent.add(f"[magenta]{obj.name}[/]")
                _add(bnode, obj.steps)
            else:  # Step / Apply / JoinBranch
                emoji = getattr(obj, "emoji", "")
                label = f"{emoji} [cyan]{obj.id}[/]" if emoji else f"[cyan]{obj.id}[/]"
                parent.add(label)

    _add(tree, chain.steps)
    return tree 