"""Native composition construction with explicit source or shared-clock timing."""

MAX_INPUTS = 193  # src/mixer/primitives/source_mask.hpp


def source_mask_param(mask: int):
    """Native pad masks use a number up to 64 bits, then an LSB-first bit string."""
    return mask if mask < 1 << 64 else f"{mask:b}"[::-1]


def compositor(params, *, timing="source", keyer=False, api=None):
    """Keep frame pacing, device ownership and native parameters caller-owned.

    ``source`` consumes corresponding input frames and preserves their PTS/EOF.
    ``clock`` uses the mixer's shared playout clock for independently paced inputs.
    Both use the native compositor primitives; no pixel transfer or Python render.
    """
    if timing not in ("source", "clock"):
        raise ValueError("composition timing must be 'source' or 'clock'")
    if keyer and timing != "clock":
        raise ValueError("the keyer requires clock timing")
    if api is None:
        from pyplumber import node as api
    node = (
        api.CudaRectOverlay
        if timing == "source"
        else (api.MixerKeyer if keyer else api.MixerCompositor)
    )
    return node(params)


def source_compositor(params, *, api=None):
    """Compose source-paced frames without introducing an output clock."""
    return compositor(params, timing="source", api=api)


def transition(params, *, threads=1, api=None):
    """Construct the shared CUDA transition filter; callers own take timing."""
    if api is None:
        from pyplumber import node as api
    return api.FilterVideo(
        {
            "graph": "transition_cuda=alpha='0':eval=frame:metadata_prefix=avp.source.",
            "threads": threads,
            **params,
        }
    )
