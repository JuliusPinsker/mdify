"""mdify CLI — PDF → Markdown converter powered by local Ollama vision models."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from mdify.converter import convert_pdf
from mdify.ollama import ensure_ollama, pull_model

console = Console(stderr=True)

DEFAULT_MODEL = "qwen2.5vl:3b"
DEFAULT_DPI = 200
DEFAULT_WORKERS = 1
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1/chat/completions"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mdify",
        description="Convert PDF files to Markdown using local Ollama vision models.",
        epilog="Examples:\n"
        "  mdify document.pdf\n"
        "  mdify ./papers/ -o ./markdown/\n"
        "  mdify report.pdf --model qwen2.5vl:7b --dpi 300\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input PDF file or directory containing PDFs.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory for markdown files (default: same as input).",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Ollama model tag (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Render resolution in DPI (default: {DEFAULT_DPI}).",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel PDF workers (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama API endpoint (default: {DEFAULT_OLLAMA_URL}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing markdown files.",
    )
    parser.add_argument(
        "--skip-ollama-check",
        action="store_true",
        help="Skip Ollama installation/model check (useful in Docker).",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )
    return parser


def _get_version() -> str:
    from mdify import __version__

    return __version__


def collect_pdfs(input_path: Path) -> list[Path]:
    """Collect PDF files from the given path (file or directory)."""
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            console.print(f"[red]Not a PDF file: {input_path}[/red]")
            sys.exit(1)
        return [input_path]
    elif input_path.is_dir():
        pdfs = sorted(input_path.glob("**/*.pdf"))
        if not pdfs:
            console.print(f"[yellow]No PDF files found in {input_path}[/yellow]")
            sys.exit(0)
        return pdfs
    else:
        console.print(f"[red]Path does not exist: {input_path}[/red]")
        sys.exit(1)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the mdify CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path: Path = args.input.resolve()
    output_dir: Path = (args.output or (input_path if input_path.is_dir() else input_path.parent)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Ollama setup ────────────────────────────────────────────────────
    if not args.skip_ollama_check:
        ollama_bin = ensure_ollama()
        pull_model(ollama_bin, args.model)
    else:
        console.print("[dim]Skipping Ollama check (--skip-ollama-check)[/dim]")

    # ── Collect PDFs ────────────────────────────────────────────────────
    pdfs = collect_pdfs(input_path)
    total = len(pdfs)

    console.print(f"\n[bold]mdify[/bold] — converting {total} PDF{'s' if total != 1 else ''}")
    console.print(f"  Model   : [cyan]{args.model}[/cyan]")
    console.print(f"  DPI     : {args.dpi}")
    console.print(f"  Workers : {args.workers}")
    console.print(f"  Output  : {output_dir}\n")

    # ── Convert ─────────────────────────────────────────────────────────
    done = 0
    failed = 0
    t0 = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Converting", total=total)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    convert_pdf,
                    pdf,
                    output_dir,
                    ollama_url=args.ollama_url,
                    model=args.model,
                    dpi=args.dpi,
                    overwrite=args.overwrite,
                ): pdf
                for pdf in pdfs
            }

            for fut in as_completed(futures):
                out_path, ok, msg = fut.result()
                done += 1
                if not ok:
                    failed += 1
                    progress.console.print(f"  [red]FAIL[/red] {out_path.name}: {msg}")
                else:
                    progress.console.print(f"  [green]OK[/green]   {out_path.name}: {msg}")
                progress.advance(task)

    elapsed = time.time() - t0
    console.print(
        f"\n[bold]Done.[/bold] {done - failed}/{total} succeeded, "
        f"{failed} failed in {elapsed:.1f}s."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
