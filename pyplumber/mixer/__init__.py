"""The mixer library: graph construction, show configuration, control client, color contracts
and Janus output, on top of the pyplumber engine API.

``MixerGraphBuilder`` needs the native ``_avplumber`` module; it is imported lazily so that
``pyplumber.mixer.control`` and the other pure-Python helpers work without it.
"""

from .models import MixerScene, MixerSource

__all__ = ["MixerGraphBuilder", "MixerScene", "MixerSource"]


def __getattr__(name):
    if name == "MixerGraphBuilder":
        from .graph import MixerGraphBuilder
        return MixerGraphBuilder
    raise AttributeError(name)
