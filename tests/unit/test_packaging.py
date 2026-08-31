"""Contract tests for the package itself. If these fail, the build is broken."""

from importlib.metadata import version

import zepto


def test_package_is_importable() -> None:
    assert zepto is not None


def test_version_matches_installed_metadata() -> None:
    assert zepto.__version__ == version("zepto-platform")
    assert zepto.__version__ != "0.0.0.dev0"
