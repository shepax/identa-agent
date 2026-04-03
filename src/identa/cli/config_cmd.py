import typer

app = typer.Typer()

@app.command()
def show():
    """Display current config"""
    pass

@app.command()
def set():
    """Update config value"""
    pass
