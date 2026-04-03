import typer

app = typer.Typer()

@app.command("list")
def list_tasks():
    """List available tasks"""
    pass

@app.command()
def add():
    """Add custom task"""
    pass

@app.command()
def inspect():
    """Show task details"""
    pass
