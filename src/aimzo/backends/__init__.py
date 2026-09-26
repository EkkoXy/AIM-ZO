from .base import Backend
from .mock import MockBackend

__all__ = ["Backend", "HFBackend", "MockBackend"]


def __getattr__(name: str):
    if name == "HFBackend":
        from .hf import HFBackend

        return HFBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
