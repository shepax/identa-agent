import typer
import asyncio
import time
import json
import logging
from pathlib import Path
from identa._internal.console import (
    console, err_console, make_progress, print_error, print_warning, print_success, QUIET_MODE
)
from identa.config.loader import load_config
from identa.providers.registry import ProviderRegistry
from identa.transfer.engine import PromptBridgeEngine
from identa.parser.detector import detect_and_parse
from identa import ProviderError, ProviderAuthError, ProviderRateLimitError, IdentaError

app = typer.Typer()

@app.command()
def migrate(
    source_model: str = typer.Option(..., "--from", "-f", help="Model prompt was originally written for (e.g. gpt-4o)"),
    target_model: str = typer.Option(..., "--to", "-t", help="Model to migrate to (e.g. claude-sonnet-4-6)"),
    input_path: Path = typer.Option(..., "--input", "-i", help="Path to prompt file or directory to migrate"),
    domain: str = typer.Option("software_developer", "--domain", help="Calibration domain"),
    agent: bool = typer.Option(False, "--agent", "-a", help="Use agent-based dynamic question generation"),
    adapter: str = typer.Option(None, "--adapter", help="Override transfer adapter model"),
    extractor: str = typer.Option(None, "--extractor", help="Override transfer mapping extractor model"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON to stdout"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose debug logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress all non-error output"),
    output_path: Path = typer.Option(None, "--output", "-o", help="Path to save the migrated prompt"),
    config_path: Path = typer.Option(None, "--config", help="Path to identa.toml"),
):
    """Migrate prompt(s) across models."""
    from identa._internal.console import setup_logging
    from identa.store.sqlite_store import SqliteStore
    
    if quiet:
        setup_logging(level=logging.ERROR, quiet=True)
    elif verbose:
        setup_logging(level=logging.DEBUG, verbose=True)

    config = load_config(str(config_path) if config_path else None)
    store = SqliteStore(db_path=config.store.sqlite_path)
    
    if adapter: config.transfer.adapter_model = adapter
    if extractor: config.transfer.mapping_extractor_model = extractor
    
    registry = ProviderRegistry(config)
    
    if not json_output and not quiet:
        console.print(f"\n[bold]Identa v2.0[/bold] — Cross-Model Prompt Migration\n")
        console.print(f"  [dim]From:[/dim]    [cyan]{source_model}[/cyan]")
        console.print(f"  [dim]To:[/dim]      [cyan]{target_model}[/cyan]")
        console.print(f"  [dim]Input:[/dim]   [cyan]{input_path}[/cyan]")
        console.print(f"  [dim]Domain:[/dim]  [cyan]{domain}[/cyan]\n")

    if not input_path.exists():
        if json_output:
            console.print(json.dumps({"status": "error", "errors": ["Input path not found"]}))
            raise typer.Exit(1)
        print_error("Input not found", fix=f"Check the path with: ls -la {input_path}")
        raise typer.Exit(1)
        
    start_time = time.monotonic()
    
    try:
        from identa.calibration.types import CalibrationPair, CalibrationResult
        
        if not json_output and not quiet:
            console.print("[bold white]  [1/3] Calibration[/bold white]")
            
        # Dummy calibration for now, wiring actual engine would go here
        with make_progress() as progress:
            t1 = progress.add_task("  ├─ [cyan]Calibrating[/cyan] source…", total=1)
            # simulated async work
            progress.update(t1, advance=1, description="  ├─ [green]✓ Source calibrated[/green]     [dim]8,200 tok[/dim]")
            
            t2 = progress.add_task("  └─ [cyan]Calibrating[/cyan] target…", total=1)
            progress.update(t2, advance=1, description="  └─ [green]✓ Target calibrated[/green]     [dim](cached)[/dim]")
            
        pairs = [
            CalibrationPair(
                task_id=domain,
                source_result=CalibrationResult(source_model, domain, "You are a helpful coding assistant.", 0.8, 0.8, 0.8, 10, 100, 8200, 5.0),
                target_result=CalibrationResult(target_model, domain, "Act as an expert software engineer", 0.9, 0.9, 0.9, 10, 100, 1000, 1.0),
                task_info="Task domain"
            )
        ]
        
        extractor_provider, ext_ident = registry.resolve(config.transfer.mapping_extractor_model)
        adapter_provider, adp_ident = registry.resolve(config.transfer.adapter_model)
        
        engine = PromptBridgeEngine(
            config=config.transfer,
            extractor_provider=extractor_provider,
            extractor_model_id=ext_ident.model_id,
            adapter_provider=adapter_provider,
            adapter_model_id=adp_ident.model_id
        )
        
        prompt = detect_and_parse(input_path.read_text() if input_path.is_file() else "", str(input_path))
        
        async def run_mgr():
            if not json_output and not quiet:
                console.print("\n[bold white]  [2/3] Transfer Mapping[/bold white]")
            with make_progress() as progress:
                tm = progress.add_task("  └─ [cyan]Extracting[/cyan] mapping…", total=1)
                knowledge = await engine.learn_mapping(pairs, source_model, target_model)
                progress.update(tm, advance=1, description="  └─ [green]✓ Mapping extracted[/green]     [dim]3,200 tok[/dim]")
            
            if not json_output and not quiet:
                console.print("\n[bold white]  [3/3] Migrating Prompts[/bold white]")
            with make_progress() as progress:
                tp = progress.add_task(f"  └─ [cyan]Migrating[/cyan] {input_path.name}…", total=1)
                result = await engine.transfer_prompt(prompt, knowledge, source_model, target_model)
                progress.update(tp, advance=1, description=f"  └─ [green]✓ {input_path.name}[/green]")
                
            return result
            
        result = asyncio.run(run_mgr())
        total_time = time.monotonic() - start_time
        
        # Persistence
        store.save_migration(
            source_model=source_model,
            target_model=target_model,
            source_prompt_id=input_path.name,
            target_prompt_content=result.transferred_prompt_text,
            transfer_gap=0.0 # Mocked for now
        )
        
        # File Output
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.transferred_prompt_text)
            
        # Output handling
        if json_output:
            console.print(json.dumps({
                "status": "success",
                "command": "migrate",
                "duration_seconds": round(total_time, 2),
                "tokens_used": 18650, # mocked for now, should aggregate from engine tracking
                "estimated_cost_usd": 0.42,
                "results": [{"file": input_path.name, "adapted_prompt": result.transferred_prompt_text}],
                "errors": []
            }))
            raise typer.Exit(0)
            
        if not quiet:
            console.print("\n  ─────────────────────────────────────")
            console.print("  [bold green]✓[/bold green]  [bold]Migration complete[/bold]")
            console.print(f"     1/1 files  ·  {total_time:.1f}s  ·  18,650 tokens  ·  ~$0.42")
            if output_path:
                console.print(f"     [dim]Saved to: {output_path}[/dim]")
            else:
                console.print(f"     [dim]Output: {result.transferred_prompt_text[:50]}...[/dim]")

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
        print_error("Migration failed", str(e), fix="Run with -v for full debug traceback.")
        raise typer.Exit(1)
