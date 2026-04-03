import typer
import asyncio
import json
import logging
from rich.panel import Panel
from rich.table import Table
from identa._internal.console import (
    console, make_progress, print_error, print_warning, print_success, QUIET_MODE
)
from identa.config.loader import load_config
from identa.calibration.engine import MAPRPEEngine
from identa.calibration.scorer import BehavioralScorer
from identa.providers.registry import ProviderRegistry
from identa.store.sqlite_store import SqliteStore
from identa import ProviderError, ProviderAuthError, ProviderRateLimitError, CalibrationError

app = typer.Typer()

@app.command(name="run")
def run(
    target_model: str = typer.Option(None, "--target", "-t", help="Model to calibrate for (overrides config)"),
    source_model: str = typer.Option(None, "--source", "-s", help="Source model (optional, for reflection)"),
    task: str = typer.Option(None, "--task", help="Alignment task ID to use (e.g. synthetic_code)"),
    source_prompt: str = typer.Option(None, "--source-prompt", "-p", help="Starting prompt text"),
    domain: str = typer.Option(None, "--domain", "-d", help="Calibration domain (e.g. software_developer)"),
    agent: bool = typer.Option(False, "--agent", "-a", help="Use agent-based dynamic question generation"),
    questions: int = typer.Option(None, "--questions", "-q", help="Override calibration questions per step"),
    iterations: int = typer.Option(None, "--iterations", "-i", help="Override global iterations"),
    steps: int = typer.Option(None, "--steps", help="Override local evolution steps"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON only"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose debug logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress all non-error output"),
    config_path: str = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Run MAP-RPE calibration for a specific model and task."""
    from identa._internal.console import setup_logging
    if quiet:
        setup_logging(level=logging.ERROR, quiet=True)
    elif verbose:
        setup_logging(level=logging.DEBUG, verbose=True)
    config = load_config(config_path)

    resolved_target = target_model or config.calibration.target_model
    if not resolved_target:
        if json_output:
            console.print(json.dumps({"status": "error", "errors": ["Target model not provided in CLI or config."]})); raise typer.Exit(1)
        print_error("Missing target model", "No target model specified.", fix="Run with --target <model> or set target_model in identa.toml.")
        raise typer.Exit(1)

    resolved_source = source_model or config.calibration.reflection_model or resolved_target

    if not json_output and not QUIET_MODE:
        console.print(f"\n[bold]Identa v2.0[/bold] — Calibration RUN\n")
        console.print(f"  [dim]Target:[/dim]  [cyan]{resolved_target}[/cyan]")
        console.print(f"  [dim]Domain:[/dim]  [cyan]{domain or task or 'none'}[/cyan]\n")
        
    if source_prompt:
        import pathlib
        path = pathlib.Path(source_prompt)
        if path.is_file():
            source_prompt = path.read_text()
            
    if iterations: config.calibration.global_iterations = iterations
    if steps: config.calibration.local_evolution_steps = steps
    if questions: config.calibration.calibration_questions = questions
    
    registry = ProviderRegistry(config)
    try:
        target_provider, targ_ident = registry.resolve(resolved_target)
        refl_provider, refl_ident = registry.resolve(resolved_source)
    except Exception as e:
        if json_output: console.print(json.dumps({"status": "error", "errors": [str(e)]})); raise typer.Exit(1)
        print_error("Configuration error", str(e), fix="Check the provider and model names.")
        raise typer.Exit(1)
        
    scorer = BehavioralScorer(config.scorer)
    store = SqliteStore(config.store.sqlite_path)
    
    engine = MAPRPEEngine(
        config=config.calibration,
        target_provider=target_provider,
        target_model_id=targ_ident.model_id,
        reflection_provider=refl_provider,
        reflection_model_id=refl_ident.model_id,
        scorer=scorer,
        cache=store
    )

    try:
        def on_progress(g, t, score):
            if not json_output and not QUIET_MODE:
                console.print(f"  [dim]├─ Gen {g}/{t} best combined score:[/dim] [cyan]{score:.3f}[/cyan]")

        if domain:
            initial = source_prompt or "You are a helpful assistant."
            result = asyncio.run(engine.calibrate_from_domain(
                domain_name=domain,
                initial_prompt=initial,
                use_agent_generation=agent,
                num_questions=questions or config.calibration.calibration_questions,
                progress_callback=on_progress
            ))
        else:
            if not task:
                raise CalibrationError("Either a task ID or a domain must be provided.")
                
            from identa.tasks.schema import AlignmentTask, TaskInstance
            import pathlib
            
            builtin_path = pathlib.Path(__file__).parent.parent / "tasks" / "builtin" / f"{task}.json"
            if not builtin_path.exists():
                raise CalibrationError(f"Task {task} not found in builtin tasks.")
                
            data = json.loads(builtin_path.read_text())
            instances = [TaskInstance(**d) for d in data]
            
            align_task = AlignmentTask(
                task_id=task,
                name=task,
                domain="coding",
                description=f"Calibration using {task}",
                instances=instances,
                evaluation_metric="text_similarity",
                source="builtin",
                default_prompt=source_prompt or "You are a helpful coding assistant."
            )
            result = asyncio.run(engine.calibrate(align_task, align_task.default_prompt, on_progress))
            
        store.save_calibration(result)
        
        if json_output:
            console.print(json.dumps({
                "status": "success",
                "command": "calibrate run",
                "duration_seconds": result.duration_seconds,
                "tokens_used": result.total_tokens,
                "estimated_cost_usd": 0.0,
                "optimal_prompt": result.optimal_prompt,
                "score": result.combined_score,
                "results": [],
                "errors": []
            }))
            raise typer.Exit(0)
            
        if not QUIET_MODE:
            console.print("\n  ─────────────────────────────────────")
            console.print("  [bold green]✓[/bold green]  [bold]Calibration complete[/bold]")
            console.print(f"     Score: {result.combined_score:.3f}  ·  {result.duration_seconds:.1f}s  ·  {result.total_tokens} tokens")
            console.print(f"     [dim]Prompt: {result.optimal_prompt[:80]}...[/dim]")

    except ProviderAuthError as e:
        if json_output: console.print(json.dumps({"status": "error", "errors": [str(e)]})); raise typer.Exit(1)
        print_error("Auth failed", context=str(e), fix="Check your configuration or environment variables.")
        raise typer.Exit(1)
    except ProviderRateLimitError as e:
        if json_output: console.print(json.dumps({"status": "error", "errors": [str(e)]})); raise typer.Exit(1)
        print_warning(f"Rate limited. Retry in {e.retry_after_seconds}s")
        raise typer.Exit(1)
    except CalibrationError as e:
        if json_output: console.print(json.dumps({"status": "error", "errors": [str(e)]})); raise typer.Exit(1)
        print_error("Calibration setup failed", str(e), fix="Check task or domain arguments.")
        raise typer.Exit(1)
    except Exception as e:
        if json_output: console.print(json.dumps({"status": "error", "errors": [str(e)]})); raise typer.Exit(1)
        import traceback
        logging.debug(traceback.format_exc())
        print_error("Calibration execution failed", str(e), fix="Run with -v for full debug traceback.")
        raise typer.Exit(1)

@app.command()
def domains():
    """List available calibration domains."""
    from identa.tasks.domains import list_domains
    
    table = Table(title="Available Calibration Domains")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Questions", justify="right")
    table.add_column("Agent Mode")
    
    for d in list_domains():
        table.add_row(
            d.domain_id, 
            d.name, 
            str(len(d.static_questions)), 
            "[green]Yes[/green]" if d.agent_generation_prompt else "[red]No[/red]"
        )
        
    console.print(table)

@app.command()
def inspect(domain_id: str):
    """View details for a domain."""
    from identa.tasks.domains import get_domain
    
    try:
        domain = get_domain(domain_id)
        console.print(Panel(f"[bold]{domain.name}[/bold]\n{domain.description}"))
        
        console.print("\n[bold]Static Question Library:[/bold]")
        for i, inst in enumerate(domain.static_questions):
            console.print(f"  {i+1}. [dim]{inst.question}[/dim]")
            
        if domain.agent_generation_prompt:
            console.print("\n[bold]Agent Generation Prompt:[/bold]")
            console.print(f"  [dim]{domain.agent_generation_prompt}[/dim]")
    except Exception as e:
        print_error(f"Domain lookup failed: {domain_id}", str(e), fix="Use 'identa calibrate domains' to see valid names.")
