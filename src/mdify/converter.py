"""Core PDF → Markdown conversion engine.

Renders each PDF page to an image, resizes to Qwen2.5-VL-safe dimensions,
and calls the Ollama OpenAI-compatible chat/completions endpoint.
"""

from __future__ import annotations

import base64
import io
import math
import time
from pathlib import Path
from typing import Callable

import fitz  # PyMuPDF
import requests
from PIL import Image

# ── Qwen2.5-VL vision encoder constraints ──────────────────────────────
FACTOR = 28  # patch_size(14) × spatial_merge_size(2)
MIN_PIXELS = 56 * 56
MAX_PIXELS = 1_003_520  # 14 × 14 × 4 × 1280

MAX_RETRIES = 3
RETRY_BACKOFF = 20  # seconds

GGML_ERRORS = ("ggml_assert", "runner process no longer running")

SYSTEM_PROMPT = """\
You are a precise document OCR engine.

ABSOLUTE RULES:
- Transcribe EVERY word, number, symbol, label, and annotation visible on the page.
- NEVER omit, summarize, or skip any readable text, no matter how small or repetitive.
- NEVER say "image not shown" or "see figure" — describe what the figure contains instead.
- Output ONLY clean markdown — no commentary, no preamble, no meta-explanations.

FORMATTING RULES:
- Use # headings for section titles.
- Use markdown tables for tabular data. Preserve every row and column.
- Preserve units, tolerances, part numbers, and footnotes exactly as printed.

HANDLING IMAGES AND DIAGRAMS:
When the page contains technical drawings, diagrams, or schematics, capture all
visible information as structured markdown:
- Dimension drawings → table of parameters and values.
- Block diagrams → describe topology (nodes, connections, labels).
- Pinout diagrams → table of pin numbers and signal names.
- Photos with labels → list every annotation.
- Part number breakdowns → structured description of each field.

Remember: transcribe EVERYTHING visible. Every label, every number, every annotation.\
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


# ── Qwen2.5-VL smart resize (from official Qwen2VLImageProcessor) ──────
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
    56 px per step, shifting the tile grid to dodge the Ollama bug.
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


def convert_pdf(
    pdf_path: Path,
    output_dir: Path,
    *,
    ollama_url: str,
    model: str,
    dpi: int = 200,
    system_prompt: str = SYSTEM_PROMPT,
    overwrite: bool = False,
    on_page: Callable[[int, int, str], None] | None = None,
) -> tuple[Path, bool, str]:
    """Convert a single PDF to markdown.

    Parameters
    ----------
    pdf_path:
        Path to the source PDF.
    output_dir:
        Directory for the output ``.md`` file.
    ollama_url:
        Ollama OpenAI-compatible chat/completions endpoint.
    model:
        Ollama model tag (e.g. ``qwen2.5vl:3b``).
    dpi:
        Render resolution for PDF pages.
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
    out_path = output_dir / (pdf_path.stem + ".md")

    if not overwrite and out_path.exists() and out_path.stat().st_size > 0:
        return out_path, True, "skipped (already exists)"

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        return out_path, False, f"cannot open PDF: {exc}"

    total_pages = len(doc)
    pages_md: list[str] = []

    for page_idx in range(total_pages):
        page = doc[page_idx]
        last_err = ""
        ggml_retries = 0

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                b64 = page_to_base64(page, dpi=dpi, shrink_step=ggml_retries)
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

    doc.close()

    if not pages_md:
        return out_path, False, "no pages extracted"

    full_md = "\n\n---\n\n".join(pages_md)
    out_path.write_text(full_md, encoding="utf-8")
    return out_path, True, f"{len(full_md):,} chars, {len(pages_md)}/{total_pages} pages"
