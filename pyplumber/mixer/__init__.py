"""The mixer library: graph construction, show configuration, control client, color contracts
and Janus output.

The show runs in a separate ``avplumber`` process. This package connects to it
and sends the control protocol; it does not embed the engine.
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
