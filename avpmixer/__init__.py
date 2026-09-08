"""Mixer graph construction and control, separate from generic pyplumber bindings.

``MixerGraphBuilder`` needs the native ``pyplumber`` module; it is imported lazily
so that ``avpmixer.control`` and other pure-Python helpers work without it.
"""

from .models import MixerScene, MixerSource

__all__ = ["MixerGraphBuilder", "MixerScene", "MixerSource"]


def __getattr__(name):
    if name == "MixerGraphBuilder":
        from .graph import MixerGraphBuilder
        return MixerGraphBuilder
    raise AttributeError(name)
