from typing import List, Optional
from rich.console import Console
from rich.panel import Panel
from rich import box


class ChainetteBanner:
    """
    Rich-based ASCII art banner for 'CHAINETTE'.

    Attributes:
        raw (List[str]): The ASCII art lines.
        border_color (str): Color for border characters (~) and panel.
        links_color (str): Color for link characters (o).
        accent_color (str): Color for accent characters (O) and letters.
        console (Console): Rich console for rendering.
    """
    DEFAULT_RAW: List[str] = [
        "              o~~O~~o~~O~~o              ",
        "           o~~O~~o      o~~O~~o           ",
        "        o~~O~~o            o~~O~~o        ",
        "          o~~O~~o   C H A I N E T T E   o~~O~~o          ",
        "        o~~O~~o            o~~O~~o        ",
        "           o~~O~~o      o~~O~~o           ",
        "              o~~O~~o~~O~~o              ",
    ]

    def __init__(
        self,
        raw: Optional[List[str]] = None,
        border_color: str = "white",
        links_color: str = "bright_magenta",
        accent_color: str = "yellow1",
        console: Optional[Console] = None,
    ) -> None:
        self.raw = raw or self.DEFAULT_RAW
        self.border_color = border_color
        self.links_color = links_color
        self.accent_color = accent_color
        self.console = console or Console()

    def _style_char(self, ch: str, letter_idx: Optional[int] = None) -> str:
        """
        Apply rich style tags to a character based on its type.

        Args:
            ch (str): The character to style.
            letter_idx (Optional[int]): Index of the letter in the text sequence.

        Returns:
            str: Styled character with rich markup.
        """
        if ch == 'o':
            return f"[{self.links_color}]{ch}[/]"
        if ch == 'O':
            return f"[{self.accent_color}]{ch}[/]"
        if ch == '~':
            return f"[{self.border_color}]{ch}[/]"
        if ch.isalpha() and letter_idx is not None:
            # Alternate letter colors based on position
            color = self.accent_color if letter_idx % 2 == 0 else self.links_color
            return f"[{color}]{ch}[/]"
        return ch  # Return the character itself if no specific style applies

    def render(self) -> Panel:
        """
        Construct the rich Panel containing the styled ASCII art.

        Returns:
            Panel: A rich Panel ready to be printed.
        """
        styled_lines: List[str] = []
        for line in self.raw:
            styled_line_parts: List[str] = []
            letter_counter = 0
            for ch in line:
                if ch.isalpha():
                    styled_line_parts.append(self._style_char(ch, letter_counter))
                    letter_counter += 1
                else:
                    styled_line_parts.append(self._style_char(ch))
            styled_lines.append("".join(styled_line_parts))

        ascii_art = "\n".join(styled_lines)
        # The Panel itself will handle the border, so we don't need the top/bottom border lines from the original raw art
        # if they were part of self.raw. The DEFAULT_RAW doesn't include them.
        return Panel(
            ascii_art,
            box=box.SQUARE, # Using a box style
            border_style=self.border_color,
            padding=(1, 2), # Add some padding for aesthetics
            expand=False,
        )

    def display(self) -> None:
        """
        Render and print the banner to the console.
        """
        panel = self.render()
        self.console.print(panel, justify="center")


def main() -> None:
    """
    Entry point for rendering the Chainette banner.
    """
    # Example with default colors
    banner_default = ChainetteBanner()
    banner_default.console.print("\n[b]Default Banner:[/b]")
    banner_default.display()

    # Example with custom colors
    banner_custom = ChainetteBanner(
        border_color="blue",
        links_color="green",
        accent_color="red"
    )
    banner_custom.console.print("\n[b]Custom Colors Banner:[/b]")
    banner_custom.display()


if __name__ == "__main__":
    main()
