"""Pengujian awal untuk memastikan paket dapat diimpor."""

from fraudshield import __version__


def test_package_version_is_defined() -> None:
    """Memastikan versi paket telah ditentukan."""
    assert __version__ == "0.1.0"
