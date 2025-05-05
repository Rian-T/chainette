from __future__ import annotations

"""chainette.core.apply
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Utility to inject pure‑Python post‑processing functions into a Chain.

`apply(fn, **kw)` returns a lightweight wrapper carrying metadata so the
Chain runner can treat it like a Step.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Type

from pydantic import BaseModel

__all__ = ["ApplyNode", "apply"]


@dataclass
class ApplyNode:  # noqa: D401
    """Container holding the callable and metadata."""

    fn: Callable[..., Any]
    kwargs: Dict[str, Any]
    id: str
    name: str

    def run(self, inputs: Sequence[BaseModel]) -> List[Any]:  # noqa: D401
        return self.fn(inputs, **self.kwargs)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def apply(fn: Callable[..., Any], **kwargs: Any):  # noqa: D401, ANN001
    """Return an :class:`ApplyNode` ready to insert in a Chain."""

    return ApplyNode(fn=fn, kwargs=kwargs, id=fn.__name__, name=fn.__name__)
