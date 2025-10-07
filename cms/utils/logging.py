import logging
from typing import Optional

from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
from rich.markup import escape

# ANSI escape codes for colors (kept for backward compatibility)
BLUE = "\033[0;34m"
RED = "\033[0;31m"
GREEN = "\033[0;32m"
RESET = "\033[0m"


# =============================================================================
# Console Management
# =============================================================================

_console = None

def get_console() -> Console:
    """Get the global Rich console instance for direct Rich output."""
    global _console
    if _console is None:
        custom_theme = Theme({
            "repr.path": "default",   # no color for paths
            "repr.filename": "default",
            "log.message": "default",
        })
        _console = Console(theme=custom_theme)
    return _console


# =============================================================================
# Specialized Logging Functions
# =============================================================================

def log_banner(text: str) -> str:
    """
    Returns a magenta-colored banner string for use with logger.

    This function creates a formatted banner with Rich markup that will be
    properly rendered by the RichHandler when logged.

    Parameters
    ----------
    text : str
        The text to display in the banner.

    Returns
    -------
    str
        Formatted banner string with Rich markup.
    """
    # Escape the text to prevent Rich from interpreting it as markup
    upper_text = text.upper()
    escaped_text = escape(upper_text)

    # Use original text length for centering calculation
    banner_text = (f"{'=' * 80}\n"
                   f"{ ' ' * ((80 - len(upper_text)) // 2)}{escaped_text}\n"
                   f"{ '=' * 80}"
                  )
    return f"[magenta]{banner_text}[/magenta]"


# =============================================================================
# Logger Setup
# =============================================================================

def setup_logging(level: str = "INFO") -> None:
    """
    Sets up logging with RichHandler configured for this project.

    The RichHandler is configured with markup enabled to support colored
    banners and tables, but regular log messages should avoid using markup
    unless specifically intended.

    Parameters
    ----------
    level : str, optional
        The logging level, by default "INFO"
    """
    log = logging.getLogger()

    # Check if handlers already exist to avoid duplicate logging
    if log.handlers:
        return

    # Use the global console instance for consistency
    console = get_console()

    # Configure RichHandler with markup enabled for banners and tables
    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=True,
        markup=True,  # Enable markup for banners and tables
        log_time_format="%H:%M:%S",
    )
    handler.setFormatter(
        logging.Formatter("%(message)s")
    )
    log.addHandler(handler)
    log.setLevel(level)
