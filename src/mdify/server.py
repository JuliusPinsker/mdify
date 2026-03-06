"""MCP server exposing mdify file → Markdown conversion via VLM."""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from mdify.converter import SUPPORTED_EXTENSIONS

mcp = FastMCP("mdify")


@mcp.tool()
def convert(
    path: str,
    model: str = "qwen3.5:4b",
    dpi: int = 200,
    ollama_url: str = "http://localhost:11434/v1/chat/completions",
) -> str:
    """Convert any file to Markdown via a local Ollama vision model.

    Supports PDF, DOCX, PPTX, XLSX, ODT, JPG, PNG, TIFF, and more.
    Every page/image is sent to the VLM for high-fidelity OCR — tables,
    diagrams, and drawings are captured as structured Markdown.

    Requires Ollama running locally. Office docs also require LibreOffice.

    Args:
        path: Absolute or relative path to the file.
        model: Ollama model tag (default: qwen3.5:4b).
        dpi: Render resolution in DPI for PDF/office docs (default: 200).
        ollama_url: Ollama API endpoint.
    """
    from mdify.converter import convert_file

    file_path = Path(path).expanduser().resolve()
    ext = file_path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        return (
            f"Unsupported file type: {ext}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    out_path, success, message = convert_file(
        file_path,
        file_path.parent,
        ollama_url=ollama_url,
        model=model,
        dpi=dpi,
        overwrite=True,
    )
    if not success:
        return f"Conversion failed: {message}"
    return out_path.read_text(encoding="utf-8")


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
