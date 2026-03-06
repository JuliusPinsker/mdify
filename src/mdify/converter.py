"""Core file → Markdown conversion engine.

Converts any supported file to images, then calls an Ollama VLM for
high-fidelity OCR. Supports PDF, DOCX, PPTX, XLSX, ODT, images, and more.
"""

from __future__ import annotations

import base64
import io
import math
import subprocess
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable

import fitz  # PyMuPDF
import requests
from PIL import Image

# ── Qwen3.5 vision encoder constraints ─────────────────────────────────
FACTOR = 32  # patch_size(16) × spatial_merge_size(2)
MIN_PIXELS = 64 * 64
MAX_PIXELS = 1_003_520

MAX_RETRIES = 2
RETRY_BACKOFF = 10  # seconds

GGML_ERRORS = ("ggml_assert", "runner process no longer running")

# ── Supported file types ────────────────────────────────────────────────
IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif", ".webp",
})
PDF_EXTENSIONS = frozenset({".pdf"})
OFFICE_EXTENSIONS = frozenset({
    ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".odt", ".ods", ".odp", ".rtf", ".csv",
})
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS | OFFICE_EXTENSIONS

SYSTEM_PROMPT = """\
You are a precise document OCR engine.

ABSOLUTE RULES:
- Transcribe EVERY word, number, symbol, label, and annotation visible on the page.
- NEVER omit, summarize, or skip any readable text, no matter how small or repetitive.
- NEVER say "image not shown" or "see figure" — describe what the figure contains instead.
- Output ONLY clean markdown — no commentary, no preamble, no meta-explanations.
- Do NOT wrap your output in code fences.

FORMATTING RULES:
- Use # headings for section titles.
- Use markdown tables for tabular data. Preserve every row and column.
- Preserve units, tolerances, part numbers, and footnotes exactly as printed.

HANDLING IMAGES AND DIAGRAMS:
- DIMENSION DRAWINGS → table of parameters and values with units.
- BLOCK DIAGRAMS → describe topology: central unit, connected modules, signal types.
- PINOUT/CONNECTOR DIAGRAMS → table of ALL pin numbers and signal names.
- PRODUCT PHOTOS WITH LABELS → describe annotations, capacities, labels.
- NOMENCLATURE DIAGRAMS → structured breakdown of all fields and options.

Transcribe EVERYTHING visible — every label, number, annotation.\
"""


def clean_markdown(text: str) -> str:
    """Remove markdown code fence artifacts wrapping the content."""
    text = text.strip()
    if text.startswith("```markdown"):
        text = text[11:].lstrip()
    if text.startswith("```"):
        text = text[3:].lstrip()
    if text.endswith("```"):
        text = text[:-3].rstrip()
    return text.strip()


# ── Qwen3.5 smart resize ───────────────────────────────────────────────
def smart_resize(height: int, width: int) -> tuple[int, int]:
    """Resize dimensions so both are multiples of ``FACTOR`` and total ≤ ``MAX_PIXELS``."""
    if height < FACTOR or width < FACTOR:
        raise ValueError(f"height:{height} or width:{width} must be >= {FACTOR}")

    h_bar = round(height / FACTOR) * FACTOR
    w_bar = round(width / FACTOR) * FACTOR

    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt((height * width) / MAX_PIXELS)
        h_bar = math.floor(height / beta / FACTOR) * FACTOR
        w_bar = math.floor(width / beta / FACTOR) * FACTOR

    if h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / FACTOR) * FACTOR
        w_bar = math.ceil(width * beta / FACTOR) * FACTOR

    return h_bar, w_bar


def page_to_base64(page: fitz.Page, dpi: int = 200, shrink_step: int = 0) -> str:
    """Render a PDF page → base64 PNG, resized to Qwen-safe dimensions.

    *shrink_step*: on GGML retries, pass 1, 2, … to reduce both dims by
    64 px per step, shifting the tile grid to dodge the Ollama bug.
    """
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    new_h, new_w = smart_resize(img.height, img.width)

    if shrink_step > 0:
        delta = FACTOR * 2 * shrink_step
        new_h = max(FACTOR, new_h - delta)
        new_w = max(FACTOR, new_w - delta)
        new_h = round(new_h / FACTOR) * FACTOR
        new_w = round(new_w / FACTOR) * FACTOR

    if (new_h, new_w) != (img.height, img.width):
        img = img.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def image_to_base64(img: Image.Image, shrink_step: int = 0) -> str:
    """Resize a PIL Image to VLM-safe dimensions and return base64 PNG."""
    if img.mode != "RGB":
        img = img.convert("RGB")

    new_h, new_w = smart_resize(img.height, img.width)

    if shrink_step > 0:
        delta = FACTOR * 2 * shrink_step
        new_h = max(FACTOR, new_h - delta)
        new_w = max(FACTOR, new_w - delta)
        new_h = round(new_h / FACTOR) * FACTOR
        new_w = round(new_w / FACTOR) * FACTOR

    if (new_h, new_w) != (img.height, img.width):
        img = img.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Office → PDF conversion ─────────────────────────────────────────────
def _find_libreoffice() -> str | None:
    """Return path to ``libreoffice`` / ``soffice`` if available."""
    for name in ("libreoffice", "soffice"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _office_to_pdf(src: Path) -> Path:
    """Convert an office document to PDF via LibreOffice headless."""
    lo = _find_libreoffice()
    if lo is None:
        raise RuntimeError(
            "LibreOffice is required for office file conversion. "
            "Install it: sudo apt install libreoffice-core"
        )
    tmp_dir = tempfile.mkdtemp(prefix="mdify_")
    subprocess.run(
        [lo, "--headless", "--convert-to", "pdf", "--outdir", tmp_dir, str(src)],
        capture_output=True,
        check=True,
        timeout=120,
    )
    pdf_path = Path(tmp_dir) / (src.stem + ".pdf")
    if not pdf_path.exists():
        raise RuntimeError(f"LibreOffice conversion produced no output for {src.name}")
    return pdf_path


def _pdf_to_images(pdf_path: Path, dpi: int = 200) -> list[Image.Image]:
    """Render every page of a PDF to a list of PIL Images."""
    doc = fitz.open(str(pdf_path))
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)
    doc.close()
    return images


def file_to_images(path: Path, dpi: int = 200) -> list[Image.Image]:
    """Convert any supported file to a list of PIL Images for VLM processing.

    - PDF: render each page
    - Images: load directly (single-element list)
    - Office docs: convert to PDF via LibreOffice, then render
    """
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if ext in IMAGE_EXTENSIONS:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return [img]

    if ext in PDF_EXTENSIONS:
        return _pdf_to_images(path, dpi=dpi)

    # Office documents → PDF → images
    pdf_path = _office_to_pdf(path)
    try:
        return _pdf_to_images(pdf_path, dpi=dpi)
    finally:
        # Clean up temp PDF
        pdf_path.unlink(missing_ok=True)
        pdf_path.parent.rmdir()


def call_vlm(
    image_b64: str,
    *,
    ollama_url: str,
    model: str,
    system_prompt: str = SYSTEM_PROMPT,
    timeout: float = 600.0,
) -> str:
    """Send a single image to Ollama and return the markdown text."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {
                        "type": "text",
                        "text": "Transcribe this page to markdown.",
                    },
                ],
            },
        ],
        "temperature": 0.0,
        "stream": False,
    }
    resp = requests.post(ollama_url, json=payload, timeout=timeout)
    data = resp.json()

    if "error" in data:
        err = data["error"]
        raise RuntimeError(err.get("message", str(err)) if isinstance(err, dict) else str(err))

    content = data["choices"][0]["message"]["content"]
    return clean_markdown(content)


def convert_file(
    file_path: Path,
    output_dir: Path,
    *,
    ollama_url: str,
    model: str,
    dpi: int = 200,
    system_prompt: str = SYSTEM_PROMPT,
    overwrite: bool = False,
    on_page: Callable[[int, int, str], None] | None = None,
) -> tuple[Path, bool, str]:
    """Convert any supported file to Markdown via VLM.

    Parameters
    ----------
    file_path:
        Path to the source file (PDF, image, or office document).
    output_dir:
        Directory for the output ``.md`` file.
    ollama_url:
        Ollama OpenAI-compatible chat/completions endpoint.
    model:
        Ollama model tag (e.g. ``qwen3.5:4b``).
    dpi:
        Render resolution (for PDF/office docs).
    system_prompt:
        System prompt for the VLM.
    overwrite:
        If *False*, skip files that already have non-empty output.
    on_page:
        Optional callback ``(page_idx, total_pages, status)`` for progress.

    Returns
    -------
    tuple of ``(output_path, success, message)``
    """
    out_path = output_dir / (file_path.stem + ".md")

    if not overwrite and out_path.exists() and out_path.stat().st_size > 0:
        return out_path, True, "skipped (already exists)"

    try:
        images = file_to_images(file_path, dpi=dpi)
    except Exception as exc:
        return out_path, False, f"cannot process file: {exc}"

    total_pages = len(images)
    pages_md: list[str] = []

    for page_idx, img in enumerate(images):
        last_err = ""
        ggml_retries = 0

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                b64 = image_to_base64(img, shrink_step=ggml_retries)
                md = call_vlm(
                    b64,
                    ollama_url=ollama_url,
                    model=model,
                    system_prompt=system_prompt,
                )
                if md.strip():
                    pages_md.append(md)
                    if on_page:
                        on_page(page_idx, total_pages, "ok")
                    break
                last_err = "empty VLM response"
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF)
            except Exception as exc:
                last_err = str(exc)
                if any(e in last_err.lower() for e in GGML_ERRORS):
                    ggml_retries += 1
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_BACKOFF)
                        continue
                elif "timeout" in last_err.lower() or "timed out" in last_err.lower():
                    if on_page:
                        on_page(page_idx, total_pages, f"timeout: {last_err}")
                    break
                else:
                    if on_page:
                        on_page(page_idx, total_pages, f"error: {last_err}")
                    break
        else:
            if on_page:
                on_page(page_idx, total_pages, f"failed: {last_err}")

    if not pages_md:
        return out_path, False, "no pages extracted"

    full_md = "\n\n---\n\n".join(pages_md)
    out_path.write_text(full_md, encoding="utf-8")
    return out_path, True, f"{len(full_md):,} chars, {len(pages_md)}/{total_pages} pages"


# Backward-compatible alias
convert_pdf = convert_file
