"""Tests for the CLI surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaicraft.cli import build_parser, main


def test_help_runs() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_version_flag() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_presets_command(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["presets"])
    assert code == 0
    captured = capsys.readouterr()
    assert "ultra" in captured.out
    assert "vivid" in captured.out


def test_generate_requires_input(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["generate", "--output", str(tmp_path / "out.jpg")])
    assert exc.value.code != 0


def test_generate_smoke(synthetic_tile_dir: Path, synthetic_target: Path, tmp_path: Path) -> None:
    out = tmp_path / "cli_out.png"
    code = main(
        [
            "generate",
            str(synthetic_target),
            "--tiles",
            str(synthetic_tile_dir),
            "--output",
            str(out),
            "--preset",
            "fast",
            "--target-tiles",
            "49",
            "--tile-size",
            "32",
        ]
    )
    assert code == 0
    assert out.exists()


def test_cache_command(synthetic_tile_dir: Path, tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    code = main(
        [
            "cache",
            "--tiles",
            str(synthetic_tile_dir),
            "--cache-dir",
            str(cache_dir),
            "--sizes",
            "32",
            "--thumb-size",
            "48",
        ]
    )
    assert code == 0
    assert (cache_dir / "features_32.npz").exists()
