"""Tests for mdify.converter — smart_resize and clean_markdown."""

import pytest

from mdify.converter import clean_markdown, smart_resize, FACTOR, MIN_PIXELS, MAX_PIXELS


class TestSmartResize:
    """Validate the Qwen2.5-VL smart resize logic."""

    def test_both_dimensions_are_factor_multiples(self):
        h, w = smart_resize(100, 200)
        assert h % FACTOR == 0
        assert w % FACTOR == 0

    def test_total_pixels_within_max(self):
        h, w = smart_resize(2000, 3000)
        assert h * w <= MAX_PIXELS

    def test_total_pixels_above_min(self):
        h, w = smart_resize(30, 30)
        assert h * w >= MIN_PIXELS

    def test_raises_on_too_small(self):
        with pytest.raises(ValueError):
            smart_resize(10, 10)

    def test_identity_for_aligned_dims(self):
        """Dimensions already aligned and within bounds should stay close."""
        h, w = smart_resize(280, 560)
        assert h == 280
        assert w == 560

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
