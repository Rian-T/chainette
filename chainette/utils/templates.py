from __future__ import annotations

"""chainette.utils.templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Uses Jinja2 for flexible and powerful templating.

Usage
-----
>>> from chainette.utils.templates import render
>>> render("Hello {{name}}", {"name": "Alice"})
'Hello Alice'
>>> render("Items: {% for item in items %}- {{item}} {% endfor %}", {"items": ["A", "B"]})
'Items: - A - B '
"""

from typing import Any, Mapping
from jinja2 import Environment, FileSystemLoader, StrictUndefined

__all__ = ["render"]

# Initialize Jinja2 environment
# Using StrictUndefined to raise errors for undefined variables in templates
env = Environment(
    loader=FileSystemLoader("."),  # Dummy loader, not used for string templates
    autoescape=False, # Disable auto-escaping
    undefined=StrictUndefined
)


def render(template_string: str, data: Mapping[str, Any]) -> str:  # noqa: D401
    """Render the *template_string* with *data* using Jinja2.

    Args:
        template_string: The template string with Jinja2 syntax.
        data: A mapping of keys to values for template rendering.

    Returns:
        The rendered string.

    Raises:
        jinja2.exceptions.UndefinedError: If a variable in the template is not in data.
        jinja2.exceptions.TemplateSyntaxError: For errors in template syntax.
    """
    try:
        template = env.from_string(template_string)
        return template.render(data)
    except Exception as exc:
        # Add more context to the exception
        raise RuntimeError(
            f"Error rendering template: {exc}\n"
            f"Template: \"{template_string[:100]}{'...' if len(template_string) > 100 else ''}\"\n"
            f"Data keys: {list(data.keys())}"
        ) from exc
