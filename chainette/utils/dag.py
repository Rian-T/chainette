from __future__ import annotations

"""DAG helpers (no side-effects, ≤ 70 LOC).

iter_nodes(chain) yields (depth, obj) depth-first.
build_rich_tree(chain) returns a Rich *Tree* ready for printing.
"""
from typing import Iterator, List, Tuple, Any
from dataclasses import dataclass

from chainette import Chain  # public import exposes core types via __all__
from chainette.core.branch import Branch
from chainette.core.node import Node

__all__ = [
    "iter_nodes",
    "build_rich_tree",
    "RenderOptions",
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
# Render options
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class RenderOptions:
    icons_on: bool = True
    max_branches: int | None = None  # None = unlimited


# --------------------------------------------------------------------------- #
# Rich-aware tree builder (import lazily to avoid hard dep at import time)
# --------------------------------------------------------------------------- #

def build_rich_tree(chain: Chain, **kwargs):  # noqa: D401
    """Return a *rich.tree.Tree* visualisation of *chain* (side-effect-free)."""
    from rich.tree import Tree  # local import keeps this module lightweight

    opts = RenderOptions(
        icons_on=not kwargs.get("no_icons", False),
        max_branches=kwargs.get("max_branches"),
    )

    tree = Tree("[bold]Execution DAG[/]")

    def _add(parent: "Tree", objs):
        for obj in objs:
            if isinstance(obj, list):  # parallel branches wrapper
                caption = "[dim]parallel ⨉ %d[/]" % len(obj)
                p = parent.add(caption)

                branches_to_show = obj
                if opts.max_branches is not None and len(obj) > opts.max_branches:
                    branches_to_show = obj[: opts.max_branches]

                for br in branches_to_show:
                    bnode = p.add(f"[magenta]{br.name}[/]")
                    _add(bnode, br.steps)

                if branches_to_show is not obj:
                    p.add(f"[dim]+{len(obj) - len(branches_to_show)} more…[/]")
            elif isinstance(obj, Branch):
                bnode = parent.add(f"[magenta]{obj.name}[/]")
                _add(bnode, obj.steps)
            else:  # Step / Apply / JoinBranch
                # Compose label – step id plus backend/engine name in dimmed style
                emoji = getattr(obj, "emoji", "") if opts.icons_on else ""
                try:
                    from chainette.engine.registry import get_engine_config  # local import to avoid hard dep when printing only

                    if hasattr(obj, "engine_name"):
                        cfg = get_engine_config(obj.engine_name)
                        backend = cfg.backend
                        eng_lbl = f" [dim]({backend})[/]"
                    else:
                        eng_lbl = ""
                except Exception:  # pragma: no cover – printing continues even if something goes wrong
                    eng_lbl = ""

                label_main = f"{emoji} [cyan]{obj.id}[/]" if emoji else f"[cyan]{obj.id}[/]"
                label = label_main + eng_lbl
                parent.add(label)

    _add(tree, chain.steps)
    return tree 