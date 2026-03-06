"""Tests for mdify.converter — smart_resize, clean_markdown, file_to_images."""

import pytest
from pathlib import Path

from mdify.converter import (
    clean_markdown,
    smart_resize,
    file_to_images,
    image_to_base64,
    FACTOR,
    MIN_PIXELS,
    MAX_PIXELS,
    SUPPORTED_EXTENSIONS,
    IMAGE_EXTENSIONS,
    OFFICE_EXTENSIONS,
)

SAMPLES_DIR = Path(__file__).parent / "samples"


class TestSmartResize:
    """Validate the Qwen3.5 smart resize logic."""

    def test_both_dimensions_are_factor_multiples(self):
        h, w = smart_resize(100, 200)
        assert h % FACTOR == 0
        assert w % FACTOR == 0

    def test_total_pixels_within_max(self):
        h, w = smart_resize(2000, 3000)
        assert h * w <= MAX_PIXELS

    def test_total_pixels_above_min(self):
        h, w = smart_resize(32, 32)
        assert h * w >= MIN_PIXELS

    def test_raises_on_too_small(self):
        with pytest.raises(ValueError):
            smart_resize(10, 10)

    def test_identity_for_aligned_dims(self):
        """Dimensions already aligned and within bounds should stay close."""
        h, w = smart_resize(320, 640)
        assert h == 320
        assert w == 640

    def test_large_image_downscaled(self):
        h, w = smart_resize(4000, 6000)
        assert h * w <= MAX_PIXELS
        assert h % FACTOR == 0
        assert w % FACTOR == 0


class TestCleanMarkdown:
    """Validate markdown fence stripping."""

    def test_removes_markdown_fence(self):
        text = "```markdown\n# Hello\n```"
        assert clean_markdown(text) == "# Hello"

    def test_removes_plain_fence(self):
        text = "```\n# Hello\n```"
        assert clean_markdown(text) == "# Hello"

    def test_no_fence_passthrough(self):
        text = "# Hello\nWorld"
        assert clean_markdown(text) == "# Hello\nWorld"

    def test_strips_whitespace(self):
        text = "  \n  # Hello  \n  "
        assert clean_markdown(text) == "# Hello"

    def test_empty_string(self):
        assert clean_markdown("") == ""


class TestSupportedExtensions:
    """Verify format categorization."""

    def test_pdf_is_supported(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS

    def test_images_are_supported(self):
        for ext in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"):
            assert ext in IMAGE_EXTENSIONS

    def test_office_docs_are_supported(self):
        for ext in (".docx", ".pptx", ".xlsx", ".odt"):
            assert ext in OFFICE_EXTENSIONS


class TestFileToImages:
    """Test file_to_images for different formats."""

    def test_pdf_produces_images(self):
        images = file_to_images(SAMPLES_DIR / "sample.pdf")
        assert len(images) > 0
        assert images[0].mode == "RGB"

    def test_jpg_produces_single_image(self):
        images = file_to_images(SAMPLES_DIR / "sample.jpg")
        assert len(images) == 1
        assert images[0].mode == "RGB"

    def test_png_produces_single_image(self):
        images = file_to_images(SAMPLES_DIR / "sample.png")
        assert len(images) == 1

    def test_tiff_produces_single_image(self):
        images = file_to_images(SAMPLES_DIR / "sample.tiff")
        assert len(images) == 1

    def test_unsupported_raises(self, tmp_path: Path):
        bad = tmp_path / "file.xyz"
        bad.touch()
        with pytest.raises(ValueError, match="Unsupported"):
            file_to_images(bad)


class TestImageToBase64:
    """Test VLM-safe image encoding."""

    def test_returns_base64_string(self):
        images = file_to_images(SAMPLES_DIR / "sample.jpg")
        b64 = image_to_base64(images[0])
        assert isinstance(b64, str)
        assert len(b64) > 100
