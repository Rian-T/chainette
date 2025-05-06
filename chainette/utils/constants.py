"""
Centralized UI constants for consistent styling across Chainette.

This module defines standard symbols, colors, and styles used in Rich
console output throughout the application.
"""

# Colorblind-friendly symbols and styles
SYMBOLS = {
    "chain": "⛓️ ",
    "success": "[bold green]✓[/bold green] ", 
    "error": "[bold red]![/bold red] ",
    "warning": "[bold yellow]⚠[/bold yellow] ",
    "step": "→ ",
    "apply": "⚙️ ",
    "branch": "🌿 ",
    "info": "[bold blue]i[/bold blue] ",
    "running": "🔄 ",
    "complete": "✅ ",
}

STYLE = {
    "header": "bold cyan",
    "dim": "dim",
    "info": "blue",
    "warning": "yellow",
    "error": "red",
    "success": "green",
    "node_step": "bold",
    "node_apply": "italic",
    "node_branch": "underline",
    "engine": "magenta",
    # UI elements
    "progress": "cyan",
    "progress_bar": "cyan",
    "progress_text": "blue",
    # Banner colors removed - will be randomized or use ChainetteBanner defaults
}
