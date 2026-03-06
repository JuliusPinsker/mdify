"""Tests for mdify.cli — argument parsing and file collection."""

import pytest
from pathlib import Path
from unittest.mock import patch

from mdify.cli import build_parser, collect_files


class TestBuildParser:
    def test_default_model(self):
        parser = build_parser()
        args = parser.parse_args(["test.pdf"])
        assert args.model == "qwen3.5:4b"

    def test_custom_model(self):
        parser = build_parser()
        args = parser.parse_args(["test.pdf", "-m", "qwen3.5:9b"])
        assert args.model == "qwen3.5:9b"

    def test_output_dir(self):
        parser = build_parser()
        args = parser.parse_args(["test.pdf", "-o", "/tmp/out"])
        assert args.output == Path("/tmp/out")

    def test_overwrite_flag(self):
        parser = build_parser()
        args = parser.parse_args(["test.pdf", "--overwrite"])
        assert args.overwrite is True

    def test_workers(self):
        parser = build_parser()
        args = parser.parse_args(["test.pdf", "-w", "4"])
        assert args.workers == 4

    def test_dpi(self):
        parser = build_parser()
        args = parser.parse_args(["test.pdf", "--dpi", "300"])
        assert args.dpi == 300


class TestCollectFiles:
    def test_single_file(self, tmp_path: Path):
        pdf = tmp_path / "doc.pdf"
        pdf.touch()
        result = collect_files(pdf)
        assert result == [pdf]

    def test_directory(self, tmp_path: Path):
        (tmp_path / "a.pdf").touch()
        (tmp_path / "b.pdf").touch()
        (tmp_path / "c.txt").touch()  # not supported
        result = collect_files(tmp_path)
        assert len(result) == 2
        assert all(p.suffix == ".pdf" for p in result)

    def test_supported_formats(self, tmp_path: Path):
        for ext in (".pdf", ".docx", ".xlsx", ".pptx", ".jpg", ".png"):
            (tmp_path / f"test{ext}").touch()
        (tmp_path / "ignore.txt").touch()
        result = collect_files(tmp_path)
        assert len(result) == 6

    def test_unsupported_file_exits(self, tmp_path: Path):
        txt = tmp_path / "doc.txt"
        txt.touch()
        with pytest.raises(SystemExit):
            collect_files(txt)

    def test_nonexistent_path_exits(self, tmp_path: Path):
        with pytest.raises(SystemExit):
            collect_files(tmp_path / "nope")
