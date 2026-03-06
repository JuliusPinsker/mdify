"""Public API for programmatic use of mdify."""

from mdify.converter import (
    SUPPORTED_EXTENSIONS,
    convert_file,
    convert_pdf,
    SYSTEM_PROMPT,
    clean_markdown,
    smart_resize,
    file_to_images,
    image_to_base64,
)
from mdify.ollama import ensure_ollama, find_ollama, pull_model

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "convert_file",
    "convert_pdf",
    "clean_markdown",
    "smart_resize",
    "file_to_images",
    "image_to_base64",
    "ensure_ollama",
    "find_ollama",
    "pull_model",
    "SYSTEM_PROMPT",
]
