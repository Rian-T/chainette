from __future__ import annotations

"""chainette.utils.ids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tiny helper for consistent identifier formatting.

Currently exposes only :func:`snake_case`, used when writing filenames and
metadata.  Kept separate so we don’t repeat the regex in multiple files.
"""

import re
import uuid
from datetime import datetime

__all__ = ["snake_case", "new_run_id"]

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


def new_run_id() -> str:
    """Generate a unique run ID combining timestamp and UUID.
    
    Returns:
        A string in format 'YYYYMMDD-HHMMSS-[first 8 chars of UUID]'
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # Just use first 8 chars for brevity
    return f"{timestamp}-{unique_id}"
