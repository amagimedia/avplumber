"""Python side of avplumber: the engine API (``AVPlumber``, ``pyplumber.node``) and the
application libraries built on it (``pyplumber.mixer``).

``AVPlumber`` needs the native ``_avplumber`` module and is imported on first use, so the
pure-Python parts (mixer configuration, control client, color contracts) load without it.
"""


def __getattr__(name):
    if name == "AVPlumber":
        from .core import AVPlumber
        return AVPlumber
    raise AttributeError(name)


__all__ = ["AVPlumber"]
