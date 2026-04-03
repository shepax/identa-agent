import typer
import logging
import importlib.metadata
from identa._internal.console import (
    console, err_console, setup_logging, print_error
)

app = typer.Typer(
    help="Identa v2.0 — Cross-Model Prompt Migration",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

@app.callback()
def main_callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose debug logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress all output except fatal errors"),
    config: str = typer.Option(None, "--config", help="Path to identa.toml"),
    version: bool = typer.Option(False, "--version", help="Print version and exit", is_eager=True),
):
    """Global configuration for the Identa CLI."""
    if version:
        try:
            ver = importlib.metadata.version("identa")
        except importlib.metadata.PackageNotFoundError:
            ver = "unknown (not installed)"
        console.print(f"Identa version: [bold cyan]{ver}[/bold cyan]")
        raise typer.Exit()

    setup_logging(level=logging.INFO, verbose=verbose, quiet=quiet)


from identa.cli import migrate, drift, calibrate, init_cmd, tasks, store, config_cmd

app.command(name="migrate")(migrate.migrate)
app.command(name="drift")(drift.drift)
app.add_typer(calibrate.app, name="calibrate")
app.command(name="init")(init_cmd.init)
app.add_typer(tasks.app, name="tasks")
app.add_typer(store.app, name="store")
app.add_typer(config_cmd.app, name="config")

def main():
    try:
        app()
    except SystemExit as e:
        # Typer exits with code 2 for usage errors
        if e.code == 2:
            err_console.print(
                "\n[dim italic]→ Run [bold]identa --help[/bold] to see all commands "
                "and required flags.[/dim italic]"
            )
        raise
