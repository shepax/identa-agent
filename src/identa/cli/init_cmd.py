import typer
import asyncio
from pathlib import Path
from rich.prompt import Prompt
from identa._internal.console import (
    console, print_error, print_success, print_warning, make_progress
)
from identa.config.settings import ProviderConfig
from identa.providers.registry import ProviderRegistry
from identa.tasks.domains import list_domains

app = typer.Typer()

@app.command()
def init():
    """Initialize project config with an interactive wizard."""
    config_path = Path.cwd() / "identa.toml"
    
    if config_path.exists():
        print_warning(f"Configuration file already exists at {config_path}")
        if not typer.confirm("Do you want to overwrite it?"):
            console.print("Aborted.")
            raise typer.Exit()

    console.print("\n[bold cyan]Identa Quick Setup[/bold cyan]")
    console.print("Let's configure your primary AI provider for calibration and transfer.\n")
    
    # 1. Select provider
    providers = ["openai", "anthropic", "openrouter", "google", "ollama"]
    provider = Prompt.ask(
        "Which provider do you want to use primarily?",
        choices=providers,
        default="openai"
    )
    
    # 2. Collect API Key
    api_key = ""
    if provider != "ollama":
        api_key_prompt = f"Enter your {provider.capitalize()} API Key"
        api_key = Prompt.ask(api_key_prompt, password=True)
    
    # 3. Select default domain
    domains = [d.domain_id for d in list_domains()]
    domain = Prompt.ask(
        "Select your primary calibration domain",
        choices=domains,
        default="software_developer" if "software_developer" in domains else domains[0]
    )

    # Validate before writing
    try:
        if api_key:
            _validate_key(provider, api_key)
            print_success("API Key validated")
    except Exception as e:
        print_error(
            title=f"Authentication failed for {provider}",
            context=f"The provided API key is invalid ({e})",
            fix="Please double-check the key and re-run identa init."
        )
        raise typer.Exit(1)

    # Write config
    _write_toml(config_path, provider, api_key)

    print_success(f"Config written to {config_path.absolute()}")
    console.print(
        "\n   [white]Next steps:[/white]\n"
        "   [dim italic]→ List calibration domains:[/dim italic]  [cyan]identa calibrate domains[/cyan]\n"
        f"   [dim italic]→ Run your first calibration:[/dim italic] [cyan]identa calibrate run --target {provider}/example-model --domain {domain}[/cyan]\n"
    )

def _validate_key(provider_name: str, api_key: str):
    """Attempt a 1-token completion to validate the key."""
    import logging
    from identa.providers.base import CompletionRequest
    
    # Minimal config to test the provider
    class MockConfig:
        providers = ProviderConfig()
    
    cfg = MockConfig()
    if provider_name == "openai":
        cfg.providers.openai_api_key = api_key
        test_model = "openai/gpt-4o-mini"
    elif provider_name == "anthropic":
        cfg.providers.anthropic_api_key = api_key
        test_model = "anthropic/claude-3-haiku-20240307"
    elif provider_name == "openrouter":
        cfg.providers.openrouter_api_key = api_key
        test_model = "openrouter/openai/gpt-4o-mini"
    elif provider_name == "google":
        cfg.providers.google_api_key = api_key
        test_model = "google/gemini-2.5-flash"
    else:
        return # Skip validation for others
    
    # Use proper logging control to hide inner tracebacks during validation
    with make_progress() as progress:
        task = progress.add_task(f"Validating {provider_name} API key...", total=1)
        
        registry = ProviderRegistry(cfg)
        provider, ident = registry.resolve(test_model)
        
        async def ping():
            try:
                await provider.complete(CompletionRequest(
                    messages=[{"role": "user", "content": "Hi"}],
                    model=ident.model_id,
                    max_tokens=1
                ))
            except Exception as e:
                # We expect auth errors to be raised
                from identa import ProviderError
                raise Exception(str(e))
                
        asyncio.run(ping())
        progress.update(task, advance=1)

def _write_toml(config_path: Path, provider: str, api_key: str):
    key_field = f"{provider}_api_key"
    content = f'''[providers]
{key_field} = "{api_key}"

[store]
backend = "sqlite"
sqlite_path = "~/.identa/store.db"
'''
    config_path.write_text(content)
