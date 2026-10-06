"""The mixer library: graph construction, show configuration, control client, color contracts
and Janus output, on top of the pyplumber engine API.

``MixerGraphBuilder`` needs the native ``_avplumber`` module; it is imported lazily so that
``pyplumber.mixer.control`` and the other pure-Python helpers work without it.
"""

from .models import MixerScene, MixerSource

__all__ = ["MixerGraphBuilder", "MixerScene", "MixerSource", "MixerOptions", "MixerApplication",
           "SourceContext", "build_application", "run_application"]


def __getattr__(name):
    if name == "MixerGraphBuilder":
        from .graph import MixerGraphBuilder
        return MixerGraphBuilder
    if name in {"MixerOptions", "MixerApplication", "SourceContext", "build_application", "run_application"}:
        from . import application
        return getattr(application, name)
    raise AttributeError(name)
