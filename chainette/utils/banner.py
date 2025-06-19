from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box


class ChainetteBanner:
    """
    Banner based on user-provided ASCII art (ascii-art(2).txt).
    """

    def __init__(
        self,
        *,
        border_color: str = "bright_cyan",
        color_palette: List[str] | None = None,
        console: Optional[Console] = None,
    ) -> None:
        self.border_color = border_color
        self.palette = color_palette or [
            "red",
            "bright_magenta",
            "yellow",
            "bright_green",
            "bright_cyan",
            "bright_blue",
        ]
        self.console = console or Console()

        # ASCII art from ascii-art(2).txt
        self.raw_art: List[str] = [
            "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            "XX                                                                            XX",
            "X XX                                                                        XX X",
            "X   X                                                                      X   X",
            "X    X                                                                    X    X",
            "X     X                  _           _            _   _                  X     X",
            "X      XX            ___| |__   __ _(_)_ __   ___| |_| |_ ___          XX      X",
            "X        X          / __| '_ \ / _` | | '_ \ / _ \ __| __/ _ \        X        X",
            "X         X        | (__| | | | (_| | | | | |  __/ |_| ||  __/       X         X",
            "X          X        \___|_| |_|\__,_|_|_| |_|\___|\__|\__\___|      X          X",
            "X           XX                                                    XX           X",
            "X             X                                                  X             X",
            "X              X                                                X              X",
            "X               X                                              X               X",
            "X                XX                                          XX                X",
            "X                  X                                        X                  X",
            "X                   X                                      X                   X",
            "X                    X                                    X                    X",
            "X                     XX                                XX                     X",
            "X                       XXXX                        XXXX                       X",
            "X                           XXXXXXXX        XXXXXXXX                           X",
            "X                                   XXXXXXXX                                   X",
            "X                                                                              X",
            "X                                                                              X",
            "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
        ]

    def _style_art(self) -> Text:
        """Return Rich Text with colourful styling applied to the ASCII art."""
        styled_lines: List[str] = []
        for y, line in enumerate(self.raw_art):
            parts: List[str] = []
            for x, ch in enumerate(line):
                if ch == "X":
                    parts.append(f"[{self.border_color}]{ch}[/]")
                elif ch == " ":
                    parts.append(" ")
                else:
                    # Rainbow gradient based on x-coordinate
                    color = self.palette[x % len(self.palette)]
                    parts.append(f"[{color}]{ch}[/]")
            styled_lines.append("".join(parts))
        return Text.from_markup("\n".join(styled_lines), justify="center")

    def render(self) -> Panel:
        """Construct the panel containing the styled ASCII art."""
        final_renderable = self._style_art()

        return Panel(
            final_renderable,
            box=box.SQUARE,
            border_style=self.border_color,
            padding=(0, 1),
            expand=False,
        )

    def display(self) -> None:
        """Render and print the banner to the console."""
        panel = self.render()
        self.console.print(panel, justify="center")


def main() -> None:
    """Entry point for rendering the Chainette banner."""
    banner = ChainetteBanner()
    banner.display()


if __name__ == "__main__":
    main()
