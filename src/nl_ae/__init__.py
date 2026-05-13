"""nl-ae — minimal recreation of Wang et al. 2024 'My Answer is C'."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nl-ae")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
