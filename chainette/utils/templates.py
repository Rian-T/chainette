from __future__ import annotations

"""chainette.utils.templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Extremely lightweight replacements for double‑curly placeholders.

Usage
-----
>>> from chainette.utils.templates import render
>>> render("Hello {{name}}", {"name": "Alice"})
'Hello Alice'

No loops, no conditionals, only dotted‑path replacement.  To include a
literal ``{{`` write ``{{{{``.
"""

import re
from typing import Any, Mapping

__all__ = ["render"]

_PATTERN = re.compile(r"\{\{\s*([\w\.]+)\s*\}\}")


def render(template: str, data: Mapping[str, Any]) -> str:  # noqa: D401
    """Replace ``{{path}}`` occurrences in *template* with *data* values.

    Nested paths (``a.b.c``) are resolved by successive dict lookups.
    Missing keys raise :class:`KeyError` – this is deliberate for early
    failure rather than silent mistakes.
    """

    def _resolve_path(data_obj: Any, path_keys: list[str]) -> Any:
        """Resolve nested path in dict or object attributes."""
        current = data_obj
        for key in path_keys:
            if isinstance(current, Mapping):
                if key not in current:
                    raise KeyError(f"Missing key '{key}' in dict")
                current = current[key]
            elif hasattr(current, key):
                current = getattr(current, key)
            else:
                raise AttributeError(f"Object has no attribute '{key}'")
        return current

    def _replace(match: re.Match[str]) -> str:  # noqa: D401
        path_str = match.group(1)
        path_keys = path_str.split(".")
        try:
            value = _resolve_path(data, path_keys)
            return str(value)
        except (KeyError, AttributeError) as exc:
            raise KeyError(f"Missing key or attribute while resolving '{{{{{path_str}}}}}': {exc}") from exc
        except Exception as exc: # Catch other potential errors during resolution
             raise RuntimeError(f"Error resolving '{{{{{path_str}}}}}': {exc}") from exc

    # Handle escaped braces first
    processed_template = template.replace("{{{{", "__DOUBLE_LEFT_BRACE__").replace("}}}}", "__DOUBLE_RIGHT_BRACE__")
    # Perform replacements
    processed_template = _PATTERN.sub(_replace, processed_template)
    # Restore escaped braces
    processed_template = processed_template.replace("__DOUBLE_LEFT_BRACE__", "{{").replace("__DOUBLE_RIGHT_BRACE__", "}}")

    return processed_template
