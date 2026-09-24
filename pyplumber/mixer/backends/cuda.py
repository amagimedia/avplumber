"""CUDA construction policy; routing and timing belong to the mixer builders."""


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
        from ..color import conversion_graph
        return conversion_graph(target, pixel_format, **options)

    def wipe_upload(self, color):
        # Alpha clips decode on CPU; upload at native size to the mixer device.
        return (color.setparams + "," if color else "") + "format=rgba,hwupload"
