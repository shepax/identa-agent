import typer
import asyncio
import json
import logging
from pathlib import Path
from rich.table import Table
from identa._internal.console import (
    console, err_console, make_progress, print_error, print_warning, print_success, QUIET_MODE
)
from identa.config.loader import load_config
from identa.providers.registry import ProviderRegistry
from identa.calibration.scorer import BehavioralScorer
from identa.calibration.drift import DriftEstimator
from identa.tasks.loader import load_builtin_task
from identa import ProviderError, ProviderAuthError, ProviderRateLimitError

app = typer.Typer()

@app.command()
def drift(
    source_model: str = typer.Option(..., "--from", "-f", help="Source model for reference baseline"),
    target_model: str = typer.Option(..., "--to", "-t", help="Target model to test prompt against"),
    input_path: Path = typer.Option(
        ..., "--input", "-i",
        help="Prompt file to analyze"
    ),
    task_id: str = typer.Option(
        "synthetic_code", "--task",
        help="Specific alignment task to test against"
    ),
    samples: int = typer.Option(
        10, "--samples", "-n",
        help="Number of test samples for gap estimation"
    ),
    json_output: bool = typer.Option(
        False, "--json",
        help="Output results as JSON exclusively to stdout"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose debug logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress all non-error output"),
) -> None:
    """Measure the transfer gap (model drift) for a prompt."""
    from identa._internal.console import setup_logging
    if quiet:
        setup_logging(level=logging.ERROR, quiet=True)
    elif verbose:
        setup_logging(level=logging.DEBUG, verbose=True)
    if not input_path.exists():
        if json_output:
            console.print(json.dumps({"status": "error", "errors": ["Prompt file not found"]}))
            raise typer.Exit(1)
        print_error("Input not found", fix=f"Check the path with: ls -la {input_path}")
        raise typer.Exit(1)

    prompt_text = input_path.read_text()
    config = load_config()
    registry = ProviderRegistry(config)
    
    try:
        source_provider, src_ident = registry.resolve(source_model)
        target_provider, targ_ident = registry.resolve(target_model)
    except Exception as e:
        if json_output:
            console.print(json.dumps({"status": "error", "errors": [str(e)]}))
            raise typer.Exit(1)
        print_error("Configuration error", str(e), fix="Check the provider and model names.")
        raise typer.Exit(1)

    scorer = BehavioralScorer(config.scorer)
    estimator = DriftEstimator(
        config=config,
        source_provider=source_provider,
        source_model_id=src_ident.model_id,
        target_provider=target_provider,
        target_model_id=targ_ident.model_id,
        scorer=scorer
    )

    task = load_builtin_task(task_id)
    
    if not json_output and not QUIET_MODE:
        console.print(f"\n[bold]Identa v2.0[/bold] — Drift Analysis\n")
        console.print(f"  [dim]From:[/dim]    [cyan]{source_model}[/cyan]")
        console.print(f"  [dim]To:[/dim]      [cyan]{target_model}[/cyan]")
        console.print(f"  [dim]Samples:[/dim] [cyan]{samples}[/cyan]\n")
        
    try:
        if not json_output and not QUIET_MODE:
            with make_progress() as progress:
                dr = progress.add_task(f"  └─ [cyan]Measuring[/cyan] transfer gap…", total=1)
                result = asyncio.run(estimator.estimate(prompt_text, task, samples=samples))
                progress.update(dr, advance=1, description=f"  └─ [green]✓ Drift measured[/green]")
        else:
            result = asyncio.run(estimator.estimate(prompt_text, task, samples=samples))

        if json_output:
            console.print(json.dumps({
                "status": "success",
                "command": "drift",
                "source_perf": result.source_perf,
                "target_perf": result.target_perf,
                "performance_gap": result.performance_gap,
                "source_behavior": result.source_behavior,
                "target_behavior": result.target_behavior,
                "behavior_gap": result.behavior_gap,
                "total_gap": result.total_gap,
                "errors": []
            }))
            raise typer.Exit(0)

        if not QUIET_MODE:
            console.print("\n  ─────────────────────────────────────")
            table = Table(box=None, header_style="dim", show_edge=False)
            table.add_column("Metric", style="white")
            table.add_column(f"Source ({source_model})", justify="right", style="cyan")
            table.add_column(f"Target ({target_model})", justify="right", style="cyan")
            table.add_column("Gap", justify="right", style="bold red")

            table.add_row("Performance", f"{result.source_perf:.3f}", f"{result.target_perf:.3f}", f"{result.performance_gap:.3f}")
            table.add_row("Behavioral", f"{result.source_behavior:.3f}", f"{result.target_behavior:.3f}", f"{result.behavior_gap:.3f}")
            table.add_row("[bold white]Combined[/bold white]", "-", "-", f"[bold blue]{result.total_gap:.3f}[/bold blue]")

            console.print(table)
            
            if result.total_gap > 0.1:
                print_warning("Major Drift Detected! Significant performance degradation on target model.")
            else:
                print_success("Low Drift. Prompt transfers well to target model.")

    except ProviderAuthError as e:
        if json_output: console.print(json.dumps({"status": "error", "errors": [str(e)]})); raise typer.Exit(1)
        print_error("Auth failed", context=str(e), fix="Check your configuration or environment variables.")
        raise typer.Exit(1)
    except ProviderRateLimitError as e:
        if json_output: console.print(json.dumps({"status": "error", "errors": [str(e)]})); raise typer.Exit(1)
        print_warning(f"Rate limited. Retry in {e.retry_after_seconds}s")
        raise typer.Exit(1)
    except Exception as e:
        if json_output: console.print(json.dumps({"status": "error", "errors": [str(e)]})); raise typer.Exit(1)
        import traceback
        logging.debug(traceback.format_exc())
        print_error("Drift analysis failed", str(e), fix="Run with -v for full debug traceback.")
        raise typer.Exit(1)
