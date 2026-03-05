"""Tests for mdify.ollama — Ollama detection utilities."""

from unittest.mock import patch

from mdify.ollama import find_ollama


class TestFindOllama:
    def test_returns_path_when_found(self):
        with patch("shutil.which", return_value="/usr/bin/ollama"):
            assert find_ollama() == "/usr/bin/ollama"

    def test_returns_none_when_missing(self):
        with patch("shutil.which", return_value=None):
            assert find_ollama() is None
