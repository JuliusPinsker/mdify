"""Ollama detection, installation guidance, and model management."""

from __future__ import annotations

import shutil
import subprocess
import sys

from rich.console import Console
from rich.prompt import Confirm

console = Console(stderr=True)


def find_ollama() -> str | None:
    """Return the path to the ``ollama`` binary, or *None* if not found."""
    return shutil.which("ollama")


def ollama_is_running(ollama_bin: str) -> bool:
    """Return *True* if the Ollama server is reachable."""
    try:
        result = subprocess.run(
            [ollama_bin, "list"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def ensure_ollama() -> str:
    """Ensure Ollama is installed and running.  Returns the binary path.

    If Ollama is not found on ``$PATH`` the user is prompted with
    installation instructions. If they decline, the process exits.
    """
    ollama_bin = find_ollama()

    if ollama_bin is None:
        console.print(
            "\n[bold red]Ollama is not installed.[/bold red]\n"
            "\nmdify requires a local Ollama server to run vision models.\n"
        )
        console.print("[bold]Install Ollama:[/bold]")
        console.print("  • Linux / WSL : [cyan]curl -fsSL https://ollama.com/install.sh | sh[/cyan]")
        console.print("  • macOS       : [cyan]brew install ollama[/cyan]  or download from [link=https://ollama.com]ollama.com[/link]")
        console.print("  • Windows     : Download from [link=https://ollama.com]ollama.com[/link]\n")

        if Confirm.ask("Would you like to attempt automatic installation? (Linux/macOS only)"):
            _auto_install_ollama()
            ollama_bin = find_ollama()
            if ollama_bin is None:
                console.print("[red]Installation did not place ollama on PATH. Please install manually.[/red]")
                sys.exit(1)
        else:
            console.print("[yellow]Please install Ollama and re-run mdify.[/yellow]")
            sys.exit(1)

    if not ollama_is_running(ollama_bin):
        console.print(
            "[yellow]Ollama is installed but the server does not appear to be running.[/yellow]\n"
            "Start it with: [cyan]ollama serve[/cyan]  (in another terminal)\n"
        )
        # Still continue — the server may start before the first request.

    return ollama_bin


def _auto_install_ollama() -> None:
    """Attempt the curl-pipe-sh install on Linux / macOS."""
    console.print("[dim]Running: curl -fsSL https://ollama.com/install.sh | sh[/dim]")
    try:
        subprocess.run(
            ["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
            check=True,
        )
        console.print("[green]Ollama installed successfully.[/green]")
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]Installation failed (exit {exc.returncode}).[/red]")
    except FileNotFoundError:
        console.print("[red]bash not found — cannot auto-install. Please install Ollama manually.[/red]")


def pull_model(ollama_bin: str, model: str) -> None:
    """Pull *model* if it is not already available locally."""
    # Check if model is already pulled
    try:
        result = subprocess.run(
            [ollama_bin, "list"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0 and model in result.stdout:
            console.print(f"[green]Model [bold]{model}[/bold] is already available.[/green]")
            return
    except Exception:
        pass

    console.print(f"[cyan]Pulling model [bold]{model}[/bold] … this may take a while.[/cyan]")
    try:
        subprocess.run(
            [ollama_bin, "pull", model],
            check=True,
        )
        console.print(f"[green]Model [bold]{model}[/bold] pulled successfully.[/green]")
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]Failed to pull model {model} (exit {exc.returncode}).[/red]")
        sys.exit(1)
