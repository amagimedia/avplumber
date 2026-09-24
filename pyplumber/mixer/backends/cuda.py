"""CUDA construction policy; routing and timing belong to the mixer builders."""

from ..color import Color, SEMIPLANAR_FORMATS, validate_conversion


class CudaMixerBackend:
    name = "cuda"
    hardware_format = "cuda"

    def compositor(self, params, *, api=None):
        if api is None:
            from pyplumber import node as api
        return api.CudaRectOverlay(params)

    def transition(self, params):
        from pyplumber.node import FilterVideo
        return FilterVideo({"graph": "transition_cuda=alpha='0':eval=frame", **params})

    def scale(self, *, width=None, height=None, pixel_format=None, interpolation=None):
        values = {"w": width, "h": height, "interp_algo": interpolation, "format": pixel_format}
        return "scale_cuda=" + ":".join(f"{key}={value}" for key, value in values.items() if value is not None)

    def conversion(self, target, pixel_format, **options):
        return conversion_graph(target, pixel_format, **options)

    def wipe_upload(self, color):
        # Alpha clips decode on CPU; upload at native size to the mixer device.
        return (color.setparams + "," if color else "") + "format=rgba,hwupload"


def conversion_graph(target, pixel_format, *, source=None, source_format=None,
                     tonemap="clip", sdr_white=203.0, hdr_peak=1000.0, desat=0.0, param=0.0):
    """Validate and normalize CUDA YUV frames, keeping identity frames zero-copy.

    NVDEC emits NV12/P010. Callers with other CUDA storage must declare it so
    the required chroma conversion precedes tone mapping. Custom source filters
    run before this graph: their output frame metadata is authoritative.
    With pixel_format=None, tonemap_cuda preserves input chroma subsampling and
    selects depth from the target transfer. Renditions need an explicit format.
    """
    target = validate_conversion(target, pixel_format, source_format=source_format, tonemap=tonemap,
                                 sdr_white=sdr_white, hdr_peak=hdr_peak, desat=desat, param=param)
    if pixel_format is not None and source is not None and Color.parse(source) == target:
        # Same contract: stamp it and change storage only when necessary.
        parts = [target.setparams]
        return ",".join(parts if source_format == pixel_format else parts + [f"scale_cuda=format={pixel_format}"])
    parts = [Color.parse(source).setparams] if source is not None else []
    if source_format and source_format not in SEMIPLANAR_FORMATS:
        # tonemap_cuda works on semiplanar storage; planar sources are re-laid out at 10 bits.
        parts.append("scale_cuda=format=p210le" if "422" in source_format else "scale_cuda=format=p010le")
    # tonemap_cuda converts color and storage in one pass, 4:2:0 or 4:2:2 in and out.
    # param is the operator knee in reference-white units (mobius/reinhard; 0 keeps the
    # filter default 0.3). mobius at 0.9 keeps 0..90% of SDR white linear and folds
    # everything brighter into the top 10% of the SDR range; 1.0 would be a plain clip.
    storage = f":format={pixel_format}" if pixel_format is not None else ""
    parts.append(f"tonemap_cuda=transfer_in=auto:transfer_out={target.transfer}{storage}"
                 f":tonemap={tonemap}:sdr_white={sdr_white:g}:hdr_peak={hdr_peak:g}:desat={desat:g}"
                 + (f":param={param:g}" if param else ""))
    return ",".join(parts)
