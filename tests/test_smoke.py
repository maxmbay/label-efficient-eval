"""Smoke test that the package imports."""

from lee import __version__


def test_version() -> None:
    assert __version__ == "0.1.0"
