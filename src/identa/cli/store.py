import typer
from rich.console import Console
from rich.table import Table
from identa.store.sqlite_store import SqliteStore
from identa.config.loader import load_config

app = typer.Typer()
console = Console()

@app.command("list")
def list_prompts(
    config_path: str = typer.Option(None, "--config", "-c", help="Path to identa.toml"),
):
    """List all stored prompts and their versions."""
    config = load_config(config_path)
    store = SqliteStore(config.store.sqlite_path)
    
    prompts = store.list_prompts()
    
    if not prompts:
        console.print("[yellow]No prompts found in store.[/yellow]")
        return

    table = Table(title="Identa Prompt Store")
    table.add_column("ID", style="cyan")
    table.add_column("Model/Format", style="magenta")
    table.add_column("Created", style="green")
    table.add_column("Preview", style="white")

    for p in prompts:
        table.add_row(
            p.id,
            f"{p.source_model} ({p.format})",
            p.created_at.strftime("%Y-%m-%%d %H:%M"),
            (p.content[:50] + "...") if len(p.content) > 50 else p.content
        )

    console.print(table)

@app.command()
def show(
    prompt_id: str = typer.Argument(..., help="Prompt ID to show"),
    config_path: str = typer.Option(None, "--config", "-c"),
):
    """Show the full content of a specific prompt version."""
    config = load_config(config_path)
    store = SqliteStore(config.store.sqlite_path)
    
    content = store.get_prompt(prompt_id)
    if not content:
        console.print(f"[bold red]Prompt {prompt_id} not found.[/bold red]")
        raise typer.Exit(1)
        
    console.print(f"[bold cyan]Prompt {prompt_id}:[/bold cyan]")
    console.print("-" * 40)
    console.print(content)
