from __future__ import annotations
"""Ultra-lightweight pub/sub **EventBus** used by the redesigned Runner.

This stays below 40 LOC as mandated by `RUNNER_PLAN.md`.

Example
-------
```python
from chainette.utils.events import subscribe, publish, BatchStarted

@subscribe(BatchStarted)
def _on_batch(evt: BatchStarted):
    print(f"processing {evt.count} items in step {evt.step_id}")

publish(BatchStarted(step_id="step1", count=128, batch_no=0))
```
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Type, TypeVar

__all__ = [
    "Event",
    "BatchStarted",
    "BatchFinished",
    "StepFinished",
    "subscribe",
    "publish",
]

T = TypeVar("T", bound="Event")
_Handler = Callable[[Any], None]
_REGISTRY: Dict[Type["Event"], List[_Handler]] = {}


@dataclass(slots=True, kw_only=True)
class Event:  # noqa: D101 – base event
    ts: datetime = field(default_factory=datetime.utcnow)


# --------------------------------------------------------------------------- #
# Concrete events (extend as needed)
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class BatchStarted(Event):
    step_id: str
    batch_no: int
    count: int  # number of items in the batch


@dataclass(slots=True)
class BatchFinished(Event):
    step_id: str
    batch_no: int
    count: int


@dataclass(slots=True)
class StepFinished(Event):
    step_id: str
    total_items: int


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #

def subscribe(event_type: Type[T]):  # noqa: D401
    """Decorator: register *func* to receive *event_type* events."""

    def _decorator(func: _Handler) -> _Handler:
        _REGISTRY.setdefault(event_type, []).append(func)
        return func

    return _decorator


def publish(evt: Event) -> None:  # noqa: D401
    """Publish an event to all registered subscribers."""
    for func in _REGISTRY.get(type(evt), []):
        try:
            func(evt)
        except Exception as e:  # noqa: BLE001
            # Failure to handle an event must never crash the main program.
            from chainette.utils.logging import log

            log.warning("event handler %s failed: %s", func.__name__, e) 