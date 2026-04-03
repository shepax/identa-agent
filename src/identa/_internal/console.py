import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

# Global state for output suppression
QUIET_MODE = False

# Primary stdout console.
console = Console()
# Primary stderr console (errors and logs).
err_console = Console(stderr=True)

# Patch console.print to respect QUIET_MODE natively without requiring every caller
# to manually check the flag.
_original_print = console.print
def _quiet_print(*args, **kwargs):
    if not QUIET_MODE:
        _original_print(*args, **kwargs)
console.print = _quiet_print


def setup_logging(level: int = logging.INFO, verbose: bool = False, quiet: bool = False) -> None:
    """Configure python logging and global quiet state."""
    global QUIET_MODE
    QUIET_MODE = quiet
    
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
        
    # We pipe logs to err_console so they don't corrupt stdout formats like --json
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=err_console, rich_tracebacks=True, markup=True)]
    )

def print_error(title: str, context: str = None, fix: str = None) -> None:
    """Renders the ✗ error template with suggested fix to stderr.
    
    Format:
      ✗  <Title>
         <Context sentence>
         → <Concrete fix / next command>
    """
    err_console.print(f"  [bold red]✗[/bold red]  [bold red]{title}[/bold red]")
    if context:
        err_console.print(f"     [white]{context}[/white]")
    if fix:
        err_console.print(f"     [dim italic]→ {fix}[/dim italic]")

def print_success(msg: str) -> None:
    """Renders a ✓ success line."""
    console.print(f"  [bold green]✓[/bold green]  [green]{msg}[/green]")

def print_warning(msg: str) -> None:
    """Renders a ⚠ warning line."""
    console.print(f"  [bold yellow]⚠[/bold yellow]  [bold yellow]{msg}[/bold yellow]")

def print_info(msg: str) -> None:
    """Renders a standard info line."""
    console.print(f"  [bold cyan]i[/bold cyan]  {msg}")

def make_progress() -> Progress:
    """Returns a standardized rich.Progress instance."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        disable=QUIET_MODE
    )
