<p align="center">
  <h1 align="center">mdify</h1>
  <p align="center">
    PDF → Markdown converter powered by local Ollama vision models
  </p>
</p>

<p align="center">
  <a href="https://pypi.org/project/mdify/"><img src="https://img.shields.io/pypi/v/mdify?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/mdify/"><img src="https://img.shields.io/pypi/pyversions/mdify" alt="Python"></a>
  <a href="https://github.com/jupinsker/mdify/blob/main/LICENSE"><img src="https://img.shields.io/github/license/jupinsker/mdify" alt="License"></a>
  <a href="https://github.com/jupinsker/mdify/actions"><img src="https://img.shields.io/github/actions/workflow/status/jupinsker/mdify/ci.yml?label=CI" alt="CI"></a>
</p>

---

**mdify** converts PDF documents into clean Markdown using a local [Ollama](https://ollama.com) vision-language model (Qwen2.5-VL). Every page is rendered as an image and fed to the VLM for precise OCR — tables, diagrams, pinouts, and dimension drawings are all captured as structured Markdown.

## Features

- **Fully local** — no cloud APIs, no data leaves your machine
- **Automatic Ollama setup** — detects Ollama, offers to install it, and pulls the model for you
- **High-fidelity OCR** — tables, technical drawings, block diagrams, and pinout diagrams are transcribed as structured Markdown
- **Batch processing** — point at a directory and convert hundreds of PDFs in parallel
- **Smart image sizing** — automatically resizes pages to Qwen2.5-VL-safe dimensions (multiples of 28, ≤ 1M pixels)
- **Retry logic** — handles GGML crashes and timeouts with automatic retry + image resizing
- **Programmatic API** — use `mdify` as a library in your own Python projects

## Installation

```bash
pip install mdify
```

### Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally

mdify will check for Ollama on startup and guide you through installation if needed.

## Quickstart

### Convert a single PDF

```bash
mdify document.pdf
```

### Convert a directory of PDFs

```bash
mdify ./papers/ -o ./markdown/
```

### Use a larger model for better accuracy

```bash
mdify document.pdf --model qwen2.5vl:7b
```

## Usage

```
usage: mdify [-h] [-o OUTPUT] [-m MODEL] [--dpi DPI] [-w WORKERS]
             [--ollama-url OLLAMA_URL] [--overwrite] [--skip-ollama-check] [-V]
             input

Convert PDF files to Markdown using local Ollama vision models.

positional arguments:
  input                 Input PDF file or directory containing PDFs.

options:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   Output directory for markdown files (default: same as input).
  -m, --model MODEL     Ollama model tag (default: qwen2.5vl:3b).
  --dpi DPI             Render resolution in DPI (default: 200).
  -w, --workers WORKERS Parallel PDF workers (default: 1).
  --ollama-url URL      Ollama API endpoint (default: http://localhost:11434/v1/chat/completions).
  --overwrite           Overwrite existing markdown files.
  --skip-ollama-check   Skip Ollama installation/model check.
  -V, --version         show program's version number and exit
```

## Python API

```python
from mdify.api import convert_pdf, ensure_ollama, pull_model

# Ensure Ollama is ready
ollama_bin = ensure_ollama()
pull_model(ollama_bin, "qwen2.5vl:3b")

# Convert a PDF
from pathlib import Path

out_path, success, message = convert_pdf(
    Path("document.pdf"),
    Path("./output/"),
    ollama_url="http://localhost:11434/v1/chat/completions",
    model="qwen2.5vl:3b",
)
print(f"{out_path}: {message}")
```

## How it works

1. **Ollama check** — verifies Ollama is installed and the requested model is pulled
2. **PDF rendering** — each page is rendered at the configured DPI using PyMuPDF
3. **Smart resize** — images are resized to satisfy Qwen2.5-VL's vision encoder constraints (dimensions must be multiples of 28, total pixels ≤ 1,003,520)
4. **VLM inference** — each page image is sent to Ollama's OpenAI-compatible endpoint with a detailed system prompt tuned for document OCR
5. **Retry handling** — GGML crashes trigger image resizing and retry; timeouts are logged
6. **Output** — per-page Markdown is joined with `---` separators and written to disk

## Model recommendations

| Model | VRAM | Speed | Accuracy |
|-------|------|-------|----------|
| `qwen2.5vl:3b` | ~4 GB | Fast | Good for most documents |
| `qwen2.5vl:7b` | ~6 GB | Medium | Better for complex tables/diagrams |

## Development

```bash
git clone https://github.com/jupinsker/mdify.git
cd mdify
pip install -e ".[dev]"
pytest
```

## License

[Apache License 2.0](LICENSE) — see [LICENSE](LICENSE) for details.
