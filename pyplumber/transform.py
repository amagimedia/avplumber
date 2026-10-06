"""Parameters of the ``cuda_transform`` node, the geometry engine shared by the applications.

``cuda_transform`` takes one CUDA frame and draws it onto one canvas per output: crop, scale,
placement and, with ``fps``, the choice of frames. It converts no colour and never repeats a
frame. An application (mixer, reframer, recorder) builds the node's parameters with the two
functions below instead of writing the JSON by hand, so a change of the engine is made here.

Pure Python: nothing in this module loads the native engine.
"""
from fractions import Fraction

FITS = ("stretch", "contain")


def _rect(name, value, smallest):
    """*value* as four integers ``(x, y, w, h)`` whose width and height are at least *smallest*."""
    rect = tuple(value)
    if len(rect) != 4 or any(type(v) is not int for v in rect) or min(rect[2:]) < smallest:
        raise ValueError(f"{name} must be four integers (x, y, w, h) with w and h of at least {smallest}")
    return rect


def _edges(dst):
    return [dst] if isinstance(dst, str) else list(dst)


def _rate(fps):
    """A number, an ``"n/d"`` string or an ``(n, d)`` pair as the ``"n/d"`` string the node parses."""
    try:
        # str(): Fraction(29.97) is the binary value of the float, not 2997/100.
        rate = Fraction(*fps) if isinstance(fps, (tuple, list)) else Fraction(str(fps))
    except ZeroDivisionError:
        raise ValueError("fps must be positive") from None
    if rate <= 0:
        raise ValueError("fps must be positive")
    return f"{rate.numerator}/{rate.denominator}"


def transform_output(dst, width, height, *, sw_format="nv12", fit="stretch", box=None, crop=None,
                     fps=None, drop=False, pass_arrays=False):
    """One output canvas of *width* x *height* in *sw_format*, sent to the edge or edges *dst*.

    The input frame, or its *crop* ``(x, y, w, h)``, is scaled into *box* ``(x, y, w, h)`` of the
    canvas; the box is the whole canvas by default. A crop width or height of 0 means the rest of
    the frame from ``x, y``. ``fit="contain"`` keeps the aspect ratio and centres the picture in
    the box. Where nothing is drawn the canvas stays black.

    With *fps* the output takes the first frame of each 1/fps slot of the input's timestamps. It
    never repeats a frame, so a consumer that needs a constant rate still needs ``force_fps``.
    *drop* discards a frame when an edge of this output is full; without it the node waits, which
    also holds back its other outputs.

    An output that would only repeat the input (the whole frame at 1:1 in the canvas's own
    *sw_format*) gets the input frame itself, undrawn. A CUarray input is passed on that way only
    with *pass_arrays*, for consumers that read arrays; otherwise it is drawn into a CUDA frame.
    """
    if fit not in FITS:
        raise ValueError(f"fit must be one of {FITS}")
    if not _edges(dst):
        raise ValueError("an output needs a dst edge")
    x, y, w, h = (0, 0, width, height) if box is None else _rect("box", box, 1)
    layer = {"dst_x": x, "dst_y": y, "dst_w": w, "dst_h": h}
    if fit != "stretch":
        layer["fit"] = fit
    if crop is not None:
        layer["crop"] = dict(zip("xywh", _rect("crop", crop, 0)))
    output = {"dst": dst if isinstance(dst, str) else list(dst), "width": width, "height": height,
              "sw_format": sw_format, "layers": [layer]}
    if fps is not None:
        output["fps"] = _rate(fps)
    if drop:
        output["drop"] = True
    if pass_arrays:
        output["pass_arrays"] = True
    return output


def transform_params(src, outputs, *, hwaccel, **node):
    """Parameters of one node that reads the edge *src* on the CUDA device named *hwaccel*.

    *outputs* are :func:`transform_output` results; all of them are drawn from the same input
    frame. *node* carries the parameters every node accepts (``name``, ``group``,
    ``auto_restart``, ``on_error``).
    """
    outputs = list(outputs)
    if not outputs:
        raise ValueError("cuda_transform needs at least one output")
    # The node reads its edges from the outputs. The top-level dst repeats them for the graph
    # manager, which orders the nodes of a group by the top-level src and dst alone
    # (NodeGroupUtils::offers in src/graph_mgmt.cpp).
    return {"src": src, "dst": [edge for output in outputs for edge in _edges(output["dst"])],
            "hwaccel": hwaccel, "outputs": outputs, **node}
