<p align="center">
  <h1 align="center">mdify</h1>
  <p align="center">
    File → Markdown converter powered by local Ollama vision models
  </p>
</p>

<p align="center">
  <a href="https://pypi.org/project/mdify/"><img src="https://img.shields.io/pypi/v/mdify?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/mdify/"><img src="https://img.shields.io/pypi/pyversions/mdify" alt="Python"></a>
  <a href="https://github.com/jupinsker/mdify/blob/main/LICENSE"><img src="https://img.shields.io/github/license/jupinsker/mdify" alt="License"></a>
  <a href="https://github.com/jupinsker/mdify/actions"><img src="https://img.shields.io/github/actions/workflow/status/jupinsker/mdify/ci.yml?label=CI" alt="CI"></a>
</p>

---

**mdify** converts documents and images into clean Markdown using a local [Ollama](https://ollama.com) vision-language model (Qwen3.5). Every page is rendered as an image and fed to the VLM for precise OCR — tables, diagrams, pinouts, and dimension drawings are all captured as structured Markdown.

### Supported formats

| Category | Extensions |
|----------|-----------|
| Documents | `.pdf` `.docx` `.doc` `.pptx` `.ppt` `.xlsx` `.xls` `.odt` `.ods` `.odp` `.rtf` `.csv` |
| Images | `.jpg` `.jpeg` `.png` `.tiff` `.tif` `.bmp` `.gif` `.webp` |

Office documents (DOCX, PPTX, XLSX, ODT, etc.) are converted to PDF via LibreOffice, then rendered to images for VLM processing.

## Features

- **Fully local** — no cloud APIs, no data leaves your machine
- **Any file format** — PDF, DOCX, PPTX, XLSX, ODT, JPG, PNG, TIFF, and more
- **Always uses VLM** — most reliable way to extract information from any document
- **Automatic Ollama setup** — detects Ollama, offers to install it, and pulls the model for you
- **High-fidelity OCR** — tables, technical drawings, block diagrams, and pinout diagrams are transcribed as structured Markdown
- **Batch processing** — point at a directory and convert hundreds of files in parallel
- **MCP server** — expose `convert` as a tool for AI agents via Model Context Protocol
- **Programmatic API** — use `mdify` as a library in your own Python projects

## Installation

```bash
pip install mdify
```

### Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- [LibreOffice](https://www.libreoffice.org/) (only for office document formats — `sudo apt install libreoffice-core`)

mdify will check for Ollama on startup and guide you through installation if needed.

## Quickstart

### Convert a single file

```bash
mdify document.pdf
mdify presentation.pptx
mdify photo.jpg
```

### Convert a directory of files

```bash
mdify ./documents/ -o ./markdown/
```

### Use a larger model for better accuracy

```bash
mdify document.pdf --model qwen3.5:9b
```

## Usage

```
usage: mdify [-h] [-o OUTPUT] [-m MODEL] [--dpi DPI] [-w WORKERS]
             [--ollama-url OLLAMA_URL] [--overwrite] [--skip-ollama-check] [-V]
             input

Convert files to Markdown using local Ollama vision models.

positional arguments:
  input                 Input file or directory containing supported files.

options:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   Output directory for markdown files (default: same as input).
  -m, --model MODEL     Ollama model tag (default: qwen3.5:4b).
  --dpi DPI             Render resolution in DPI (default: 200).
  -w, --workers WORKERS Parallel workers (default: 1).
  --ollama-url URL      Ollama API endpoint (default: http://localhost:11434/v1/chat/completions).
  --overwrite           Overwrite existing markdown files.
  --skip-ollama-check   Skip Ollama installation/model check.
  -V, --version         show program's version number and exit
```

## MCP Server

mdify includes a built-in [Model Context Protocol](https://modelcontextprotocol.io/) server so AI agents can convert files directly.

### VS Code setup

Add to your `~/.config/Code/User/mcp.json` (or `%APPDATA%/Code/User/mcp.json` on Windows):

```json
{
  "servers": {
    "mdify": {
      "type": "stdio",
      "command": "mdify-mcp"
    }
  }
}
```

The server exposes a single `convert` tool that accepts any supported file path.

## Python API

```python
from mdify.api import convert_file, ensure_ollama, pull_model

# Ensure Ollama is ready
ollama_bin = ensure_ollama()
pull_model(ollama_bin, "qwen3.5:4b")

# Convert any file
from pathlib import Path

out_path, success, message = convert_file(
    Path("presentation.pptx"),
    Path("./output/"),
    ollama_url="http://localhost:11434/v1/chat/completions",
    model="qwen3.5:4b",
)
print(f"{out_path}: {message}")
```

## How it works

1. **Ollama check** — verifies Ollama is installed and the requested model is pulled
2. **File → images** — PDFs are rendered page-by-page via PyMuPDF; images load directly; office docs are converted to PDF via LibreOffice first
3. **Smart resize** — images are resized to satisfy Qwen3.5’s vision encoder constraints (dimensions must be multiples of 32, total pixels ≤ 1,003,520)
4. **VLM inference** — each image is sent to Ollama's OpenAI-compatible endpoint with a detailed system prompt tuned for document OCR
5. **Retry handling** — GGML crashes trigger image resizing and retry; timeouts are logged
6. **Output** — per-page Markdown is joined with `---` separators and written to disk

## Model recommendations

| Model | VRAM | Speed | Accuracy |
|-------|------|-------|----------|
| `qwen3.5:4b` | ~4 GB | Fast | Good for most documents |
| `qwen3.5:9b` | ~8 GB | Medium | Better for complex tables/diagrams |

## Development

```bash
git clone https://github.com/jupinsker/mdify.git
cd mdify
pip install -e ".[dev]"
pytest
```

## License

[Apache License 2.0](LICENSE) — see [LICENSE](LICENSE) for details.
