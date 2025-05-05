from __future__ import annotations

"""chainette.utils.ids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tiny helper for consistent identifier formatting.

Currently exposes only :func:`snake_case`, used when writing filenames and
metadata.  Kept separate so we don’t repeat the regex in multiple files.
"""

import re

__all__ = ["snake_case"]

_PATTERN = re.compile(r"[^a-zA-Z0-9]+")


def snake_case(text: str) -> str:  # noqa: D401
    """Return *text* converted to ``snake_case``.

    * non‑alphanumeric chars become ``_``
    * multiple underscores are squeezed
    * leading/trailing underscores are stripped
    * everything lower‑cased
    """

    s = _PATTERN.sub("_", text)
    s = re.sub(r"_+", "_", s)
    return s.strip("_").lower()
