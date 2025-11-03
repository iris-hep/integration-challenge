import logging
from typing import Optional, Dict, Any, Union
import copy

from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
from rich.markup import escape
from rich.table import Table

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
# Configuration Display & Logging
# =============================================================================

class ConfigLogger:
    """
    A specialized logger for displaying and comparing configuration settings
    using Rich tables with advanced formatting and comparison features.
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the ConfigLogger.

        Args:
            console: Optional Rich Console instance. If None, uses the global console.
        """
        self.console = console or get_console()

    def _format_value_recursive(self, value: Any, depth: int = 0, max_depth: int = 4) -> str:
        """
        Recursively format a value for display, handling nested structures.

        Args:
            value: The value to format
            depth: Current recursion depth
            max_depth: Maximum recursion depth to prevent infinite loops

        Returns:
            Formatted string representation
        """
        indent = "  " * depth

        if depth > max_depth:
            return f"{type(value).__name__}(...)"

        if isinstance(value, dict):
            if len(value) == 0:
                return "{}"

            items = []
            for k, v in value.items():
                formatted_v = self._format_value_recursive(v, depth + 1, max_depth)
                # Keep simple values on same line
                if isinstance(v, (str, int, float, bool)):
                    if isinstance(v, str):
                        items.append(f"{indent}  {k}: '{v}'")
                    else:
                        items.append(f"{indent}  {k}: {formatted_v}")
                elif callable(v):
                    # Functions on same line
                    items.append(f"{indent}  {k}: {formatted_v}")
                elif isinstance(v, (list, dict)):
                    # Check if list is compact (single line) or needs expansion
                    if isinstance(v, list) and depth < max_depth:
                        # Try compact formatting first
                        compact_v = self._format_value_recursive(v, depth + 1, max_depth)
                        # If it's a single line (no newlines), keep it on same line
                        if '\n' not in compact_v:
                            items.append(f"{indent}  {k}: {compact_v}")
                        else:
                            items.append(f"{indent}  {k}:\n{compact_v}")
                    elif isinstance(v, dict) and depth < max_depth:
                        items.append(f"{indent}  {k}:\n{formatted_v}")
                    else:
                        items.append(f"{indent}  {k}: {type(v).__name__}(length: {len(v)})")
                else:
                    items.append(f"{indent}  {k}: {formatted_v}")

            return f"{{\n" + "\n".join(items) + f"\n{indent}}}"

        elif isinstance(value, list):
            if len(value) == 0:
                return "[]"

            # Special compact formatting for lists of simple items (including tuples)
            if len(value) <= 5 and all(isinstance(item, (str, int, float, bool, tuple)) for item in value):
                # Additional check: if all tuples are simple (max 2 elements, strings/numbers)
                all_simple = True
                for item in value:
                    if isinstance(item, tuple):
                        if len(item) > 3 or not all(isinstance(x, (str, int, float, bool, type(None))) for x in item):
                            all_simple = False
                            break

                if all_simple:
                    formatted_items = []
                    for item in value:
                        if isinstance(item, str):
                            formatted_items.append(f"'{item}'")
                        elif isinstance(item, tuple):
                            tuple_items = []
                            for x in item:
                                if isinstance(x, str):
                                    tuple_items.append(f"'{x}'")
                                elif x is None:
                                    tuple_items.append('None')
                                else:
                                    tuple_items.append(str(x))
                            formatted_items.append(f"({', '.join(tuple_items)})")
                        else:
                            formatted_items.append(str(item))
                    return f"[{', '.join(formatted_items)}]"

            items = []
            for i, item in enumerate(value):
                formatted_item = self._format_value_recursive(item, depth + 1, max_depth)
                # Keep simple values on same line
                if isinstance(item, (str, int, float, bool)):
                    if isinstance(item, str):
                        items.append(f"{indent}  [{i}]: '{item}'")
                    else:
                        items.append(f"{indent}  [{i}]: {formatted_item}")
                elif isinstance(item, tuple):
                    tuple_items = [f"'{x}'" if isinstance(x, str) else str(x) for x in item]
                    items.append(f"{indent}  [{i}]: ({', '.join(tuple_items)})")
                elif isinstance(item, (list, dict)):
                    # Always expand nested structures when depth allows
                    if depth < max_depth:
                        items.append(f"{indent}  [{i}]:\n{formatted_item}")
                    else:
                        items.append(f"{indent}  [{i}]: {type(item).__name__}(length: {len(item)})")
                else:
                    items.append(f"{indent}  [{i}]: {formatted_item}")

            return "[\n" + "\n".join(items) + f"\n{indent}]"

        elif isinstance(value, str):
            # Truncate very long strings
            if len(value) > 100:
                return f"'{value[:100]}...'"
            return f"'{value}'"

        elif callable(value):
            # More compact function display
            func_name = getattr(value, '__name__', '<lambda>')
            return f"Function: {func_name}"

        else:
            return str(value)

    def _compare_values(self, old_val: Any, new_val: Any) -> bool:
        """
        Compare two values and determine if they're different.

        Args:
            old_val: Original value
            new_val: New value

        Returns:
            True if different, False if same.
        """
        # Handle callable functions specially
        if callable(old_val) and callable(new_val):
            return old_val.__name__ != new_val.__name__
        elif callable(old_val) or callable(new_val):
            return True

        # For other types, use direct comparison
        try:
            return old_val != new_val
        except:
            # If comparison fails, consider them different
            return True

    def _find_differences(self, old_config: Dict[str, Any], new_config: Dict[str, Any], prefix: str = "") -> Dict[str, Dict[str, Any]]:
        """
        Recursively find differences between two configurations.

        Args:
            old_config: Original configuration
            new_config: New configuration
            prefix: Path prefix for nested keys

        Returns:
            Dictionary of changed paths and their values.
        """
        differences = {}

        # Check for changes in existing keys
        if isinstance(old_config, dict) and isinstance(new_config, dict):
            all_keys = set(old_config.keys()) | set(new_config.keys())

            for key in all_keys:
                current_path = f"{prefix}.{key}" if prefix else key

                if key not in old_config:
                    # New key added
                    differences[current_path] = {
                        'type': 'added',
                        'old_value': None,
                        'new_value': new_config[key]
                    }
                elif key not in new_config:
                    # Key removed
                    differences[current_path] = {
                        'type': 'removed',
                        'old_value': old_config[key],
                        'new_value': None
                    }
                elif isinstance(old_config[key], dict) and isinstance(new_config[key], dict):
                    # Recursively check nested dictionaries
                    nested_diffs = self._find_differences(old_config[key], new_config[key], current_path)
                    differences.update(nested_diffs)
                elif isinstance(old_config[key], list) and isinstance(new_config[key], list):
                    # Check list differences
                    if old_config[key] != new_config[key]:
                        differences[current_path] = {
                            'type': 'modified',
                            'old_value': old_config[key],
                            'new_value': new_config[key]
                        }
                elif self._compare_values(old_config[key], new_config[key]):
                    # Value changed
                    differences[current_path] = {
                        'type': 'modified',
                        'old_value': old_config[key],
                        'new_value': new_config[key]
                    }

        return differences

    def display_config_table(self, config: Dict[str, Any], expand: bool = False,
                           compare_with: Optional[Dict[str, Any]] = None,
                           show_only_changes: bool = False,
                           table_width: Optional[int] = None) -> None:
        """
        Display configuration settings in separate tables per section using Rich.

        Args:
            config: A dictionary containing configuration settings.
            expand: If True, recursively expand lists and dicts to show their contents.
            compare_with: Optional previous configuration to compare against.
            show_only_changes: If True and compare_with is provided, only show changed values.
            table_width: Optional fixed width for all tables. If None, uses auto-sizing.
        """
        # Find differences if comparison is requested
        differences = {}
        if compare_with is not None:
            differences = self._find_differences(compare_with, config)

        # If only showing changes and no differences found
        if show_only_changes and compare_with is not None and not differences:
            self.console.print("[green]No changes detected between configurations.[/green]")
            return

        # Set default table width if not specified
        if table_width is None:
            table_width = 120  # Default consistent width

        for section, settings in config.items():
            # Check if this section has any changes when showing only changes
            if show_only_changes and compare_with is not None:
                section_has_changes = any(path.startswith(section) for path in differences.keys())
                if not section_has_changes:
                    continue

            # Create a separate table for each section with consistent width
            table_title = f"Configuration: {section.upper()}"
            if compare_with is not None and differences:
                section_changes = sum(1 for path in differences.keys() if path.startswith(section))
                if section_changes > 0:
                    table_title += f" [yellow]({section_changes} changes)[/yellow]"

            table = Table(title=table_title, width=table_width, expand=False)

            # Configure columns with flexible widths but minimum constraints
            if compare_with is not None:
                # With status column: flexible but with min/max constraints
                table.add_column("Key", style="bold cyan", justify="left", min_width=15, max_width=40)
                table.add_column("Value", justify="left", min_width=40, no_wrap=False)
                table.add_column("Status", style="bold", justify="center", min_width=10, max_width=20)
            else:
                # Without status column: flexible but with min/max constraints
                table.add_column("Key", style="bold cyan", justify="left", min_width=20, max_width=50)
                table.add_column("Value", justify="left", min_width=50, no_wrap=False)

            if isinstance(settings, dict):
                for key, value in settings.items():
                    current_path = f"{section}.{key}"

                    # Check if this key has changes when showing only changes
                    if show_only_changes and compare_with is not None:
                        key_has_changes = any(path.startswith(current_path) for path in differences.keys())
                        if not key_has_changes:
                            continue

                    # Format the value using the improved recursive formatter
                    if expand:
                        value_str = self._format_value_recursive(value)
                    elif isinstance(value, (list, dict)):
                        value_str = f"{type(value).__name__} (length: {len(value)})"
                    elif callable(value):
                        func_name = getattr(value, '__name__', '<lambda>')
                        value_str = f"Function: {func_name}"
                    else:
                        value_str = str(value)

                    # Add status column if comparing
                    if compare_with is not None:
                        status = ""
                        if current_path in differences:
                            diff = differences[current_path]
                            if diff['type'] == 'added':
                                status = "[green]NEW[/green]"
                            elif diff['type'] == 'modified':
                                status = "[yellow]CHANGED[/yellow]"
                            elif diff['type'] == 'removed':
                                status = "[red]REMOVED[/red]"
                        else:
                            # Check if any nested path has changes
                            has_nested_changes = any(path.startswith(current_path + ".") for path in differences.keys())
                            if has_nested_changes:
                                status = "[blue]NESTED CHANGES[/blue]"
                            else:
                                status = "[dim]unchanged[/dim]" if not show_only_changes else ""

                        table.add_row(key, value_str, status)
                    else:
                        table.add_row(key, value_str)
            else:
                # Handle non-dict values at the section level
                if isinstance(settings, dict) and expand:
                    for k, v in settings.items():
                        if isinstance(v, str):
                            table.add_row(k, f"'{v}'")
                        elif callable(v):
                            table.add_row(k, f"Function: {v.__name__}")
                        elif isinstance(v, (dict, list)) and expand:
                            formatted_v = self._format_value_recursive(v)
                            table.add_row(k, formatted_v)
                        else:
                            table.add_row(k, str(v))
                elif isinstance(settings, list) and expand:
                    formatted_list = self._format_value_recursive(settings)
                    table.add_row("Items", formatted_list)
                elif isinstance(settings, (list, dict)):
                    value_str = f"{type(settings).__name__} (length: {len(settings)})"
                    table.add_row("Value", value_str)
                elif callable(settings):
                    value_str = f"Function: {settings.__name__}"
                    table.add_row("Value", value_str)
                else:
                    value_str = str(settings)
                    table.add_row("Value", value_str)

            # Only print the table if it has rows
            if table.row_count > 0:
                self.console.print(table)

    def compare_configs(self, old_config: Dict[str, Any], new_config: Dict[str, Any],
                       title: str = "Configuration Comparison",
                       table_width: Optional[int] = None) -> None:
        """
        Compare two configurations and display only the changes.

        Args:
            old_config: Original configuration
            new_config: New configuration
            title: Title for the comparison display
            table_width: Optional fixed width for all tables. If None, uses default.
        """
        self.console.print(f"\n[bold magenta]{title}[/bold magenta]")
        self.display_config_table(new_config, expand=True, compare_with=old_config,
                                 show_only_changes=True, table_width=table_width)

    def display_config_summary(self, config: Dict[str, Any], sections: Optional[list] = None,
                              table_width: Optional[int] = None) -> None:
        """
        Display a summary of configuration sections.

        Args:
            config: Configuration dictionary
            sections: Optional list of sections to display. If None, displays all.
            table_width: Optional fixed width for all tables. If None, uses default.
        """
        if sections:
            filtered_config = {k: v for k, v in config.items() if k in sections}
        else:
            filtered_config = config

        self.display_config_table(filtered_config, expand=False, table_width=table_width)


# =============================================================================
# Global Configuration Logger Instance
# =============================================================================

_config_logger = None

def get_config_logger() -> ConfigLogger:
    """Get the global ConfigLogger instance."""
    global _config_logger
    if _config_logger is None:
        _config_logger = ConfigLogger()
    return _config_logger


# Convenience functions for backward compatibility
def display_config_table(config: Dict[str, Any], expand: bool = False,
                        compare_with: Optional[Dict[str, Any]] = None,
                        show_only_changes: bool = False,
                        table_width: Optional[int] = None) -> None:
    """
    Convenience function to display configuration table using the global ConfigLogger.

    Args:
        config: A dictionary containing configuration settings.
        expand: If True, recursively expand lists and dicts to show their contents.
        compare_with: Optional previous configuration to compare against.
        show_only_changes: If True and compare_with is provided, only show changed values.
        table_width: Optional fixed width for all tables. If None, uses default (120).
    """
    config_logger = get_config_logger()
    config_logger.display_config_table(config, expand, compare_with, show_only_changes, table_width)


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

