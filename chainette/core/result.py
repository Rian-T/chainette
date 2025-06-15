from __future__ import annotations
"""Minimal Result dataclass capturing success or failure.

Designed to propagate structured errors through the Chainette runtime
without throwing exceptions at the first failure.  LOC ≤30 by design.
"""
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

T = TypeVar("T")

__all__ = ["Result"]


@dataclass(slots=True)
class Result(Generic[T]):  # noqa: D101
    value: Optional[T] = None
    error: Optional[Exception] = None

    # ------------------------------------------------------------------ #
    @property
    def ok(self) -> bool:  # noqa: D401
        """Return True when *error* is None."""
        return self.error is None

    # Convenience constructors ----------------------------------------- #
    @staticmethod
    def success(val: T) -> "Result[T]":  # noqa: D401
        return Result(value=val)

    @staticmethod
    def failure(err: Exception) -> "Result[None]":  # noqa: D401
        return Result(error=err)

    # ------------------------------------------------------------------ #
    def unwrap(self) -> T:  # noqa: D401
        """Return *value* or raise *error* if present (Rust-like)."""
        if self.error is not None:
            raise self.error
        return self.value  # type: ignore[return-value] 