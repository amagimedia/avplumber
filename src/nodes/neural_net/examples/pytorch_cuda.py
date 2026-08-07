#!/usr/bin/env python3
# Simple node that just prints the input data

import time
import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

import pyplumber
from pyplumber.node import (
    PythonNode,
    InputRec,
    Demux,
    DecVideo,
    RealtimeVideoFrame,
    FilterVideo,
    AssumeVideoFormat,
    EncVideo,
    Mux,
    Output,
)

INPUT_URL = os.environ.get("AVP_INPUT", "input.mp4")
OUTPUT_URL = os.environ.get("AVP_OUTPUT", "output.mp4")
OUTPUT_FORMAT = os.environ.get("AVP_OUTPUT_FORMAT", "mpegts")
USE_REALTIME = os.environ.get("AVP_USE_REALTIME", "1") == "1"

FLOAT_TYPE = torch.float16

class _CudaPlaneView:
    def __init__(self, ptr: int, shape: tuple[int, int], strides: tuple[int, int], typestr: str = "|u1"):
        self._iface = {
            "shape": shape,
            "strides": strides,
            "typestr": typestr,
            "data": (ptr, False),
            "version": 3,
        }

    @property
    def __cuda_array_interface__(self):
        return self._iface


avp = pyplumber.AVPlumber()

avp.executeCommandsFromString('hwaccel.init { "name": "@gpu", "type": "cuda" }')
avp.edges.planCapacity("*", 7)

input_rec = InputRec({
    "url": INPUT_URL,
    "dst": "in10",
    "group": "in",
    "auto_restart": "off",
    "name": "input",
    "timeout": -1,
    "preseek": 0,
    "loop": True
})
demux = Demux({
    "src": "in10",
    "routing": {"v:0": "v_in"},
    "group": "in",
    "auto_restart": "off",
})
dec_video = DecVideo({
    "auto_restart": "off",
    "name": "Video_Decode",
    "group": "in",
    "src": "v_in",
    "dst": "Video_Decode_out",
    "pixel_format": "?cuda",
    "hwaccel": "@gpu",
    "codec_map": {"h264": "h264_cuvid", "hevc": "hevc_cuvid"},
    "hwaccel_only_for_codecs": ["h264", "hevc"],
})
if USE_REALTIME:
    realtime_vf = RealtimeVideoFrame({
        "team": "sync-team",
        "negative_time_tolerance": 0.007,
        "jitter_margin": 0,
        "discontinuity_threshold": 3,
        "src": "Video_Decode_out",
        "dst": "v_rt",
        "group": "g1",
        "name": "rtsync",
        "set_pts": True,
    })
else:
    realtime_vf = None
rescale = FilterVideo({
    "graph": "scale_cuda=w=1280:h=720",
    "src": "v_rt" if USE_REALTIME else "Video_Decode_out",
    "dst": "scaled_to_python",
    "group": "g1",
    "name": "video_scaler",
    "auto_restart": "group",
    "dst_width": 1280,
    "dst_height": 720,
    "dst_pixel_format": "cuda",
    "hwaccel": "@gpu",
})
assume_video_format = AssumeVideoFormat({
    "width": 1280,
    "height": 720,
    "pixel_format": "cuda",
    "real_pixel_format": "nv12",
    "group": "g1",
    "src": "scaled_from_python",
    "dst": "v_to_encoder_assumed",
    "auto_restart": "group",
})
enc_video = EncVideo({
    "auto_restart": "group",
    "name": "Video_Encode",
    "group": "g1",
    "src": "v_to_encoder_assumed",
    "dst": "v_outenc",
    "codec": "h264_nvenc",
    "hwaccel": "@gpu",
    "options": {
        "b": "16000k",
        "maxrate": "16000k",
        "bufsize": "32000k",
        "preset": "p7",
        "profile": "high",
        "spatial_aq": 1,
        "temporal_aq": 1
    },
})
mux = Mux({
    "src": ["v_outenc"],
    "dst": "mux0-1",
    "group": "g1",
})
out = Output({
    "format": OUTPUT_FORMAT,
    "url": OUTPUT_URL,
    "src": "mux0-1",
    "group": "g1",
    "auto_restart": "group",
})
        

for node in (
    input_rec,
    demux,
    dec_video,
    realtime_vf,
    rescale,
    assume_video_format,
    enc_video,
    mux,
    out,
):
    if node is not None:
        avp.addNode(node)


class PytorchCUDANode(PythonNode):
    def __init__(self, args):
        super().__init__(args)
        self._frame_index = 0
        self._sample_every_n = max(1, int(args.get("sample_every_n", 1)))
        self._log_every_n = max(1, int(args.get("log_every_n", 30)))
        self._torch_warning_logged = False
        self._device = None
        self._blur_kernel = None
        self._sobel_x = None
        self._sobel_y = None

        if torch is not None and F is not None:
            try:
                if torch.cuda.is_available():
                    self._device = torch.device("cuda:0")
                    blur = torch.tensor(
                        [
                            [1.0, 2.0, 1.0],
                            [2.0, 4.0, 2.0],
                            [1.0, 2.0, 1.0],
                        ],
                        dtype=FLOAT_TYPE,
                        device=self._device,
                    )
                    sobel_x = torch.tensor(
                        [
                            [-1.0, 0.0, 1.0],
                            [-2.0, 0.0, 2.0],
                            [-1.0, 0.0, 1.0],
                        ],
                        dtype=FLOAT_TYPE,
                        device=self._device,
                    )
                    sobel_y = torch.tensor(
                        [
                            [1.0, 2.0, 1.0],
                            [0.0, 0.0, 0.0],
                            [-1.0, -2.0, -1.0],
                        ],
                        dtype=FLOAT_TYPE,
                        device=self._device,
                    )
                    self._blur_kernel = (blur / 16.0).view(1, 1, 3, 3)
                    self._sobel_x = sobel_x.view(1, 1, 3, 3)
                    self._sobel_y = sobel_y.view(1, 1, 3, 3)
            except Exception:
                self._device = None
                self._blur_kernel = None
                self._sobel_x = None
                self._sobel_y = None

    def _get_luma_plane(self, frame):
        width = int(frame.width)
        height = int(frame.height)
        linesize = frame.linesize
        data_ptr = frame.data_ptr
        stride = int(linesize[0]) if linesize else 0
        ptr = int(data_ptr[0]) if data_ptr else 0
        #print(f"width: {width}, height: {height}, linesize: {linesize}, data_ptr: {data_ptr}, stride: {stride}, ptr: {ptr}")
        if width <= 0 or height <= 0 or stride <= 0 or ptr <= 0:
            return None

        # Zero-copy view over CUDA frame plane memory.
        view = _CudaPlaneView(
            ptr=ptr,
            shape=(height, stride),
            strides=(stride, 1),
        )
        plane = torch.as_tensor(view, device=self._device)
        return plane[:, :width]

    def _apply_filter(self, frame) -> str:
        y_plane_u8 = self._get_luma_plane(frame)
        if y_plane_u8 is None:
            return "skipped_invalid_frame"

        y_plane_f32 = y_plane_u8.to(FLOAT_TYPE).mul_(1.0 / 255.0).unsqueeze(0).unsqueeze(0)
        blur = F.conv2d(
            y_plane_f32,
            self._blur_kernel,
            padding=1,
        )
        grad_x = F.conv2d(blur, self._sobel_x, padding=1)
        grad_y = F.conv2d(blur, self._sobel_y, padding=1)
        edge = torch.sqrt(grad_x * grad_x + grad_y * grad_y)
        edge = edge / (edge.amax() + 1e-6)

        # High-visibility stylization: solarized tones + bright edge emphasis.
        solarized = torch.where(blur > 0.5, 1.0 - blur, blur) * 2.0
        stylized = (solarized * 0.7) + (edge * 0.85)
        stylized = stylized.clamp_(0.0, 1.0).squeeze(0).squeeze(0)
        y_plane_u8.copy_(stylized.mul_(255.0).to(torch.uint8))
        torch.cuda.current_stream(device=self._device).synchronize()
        return "applied_cuda_stylized_edges"

    def process(self):
        p = self._src.get()
        if not p:
            return

        #self._dst.enqueue(p)
        #return

        self._frame_index += 1
        p.metadata["pytorch_filter_frame"] = self._frame_index

        if (
            self._device is None
            or self._blur_kernel is None
            or self._sobel_x is None
            or self._sobel_y is None
        ):
            if not self._torch_warning_logged:
                print("PytorchCUDANode: CUDA torch is unavailable, forwarding frames unchanged.")
                self._torch_warning_logged = True
            p.metadata["pytorch_filter_status"] = "skipped_no_torch_cuda"
            self._dst.enqueue(p)
            return

        if self._frame_index % self._sample_every_n != 0:
            p.metadata["pytorch_filter_status"] = "skipped_not_sampled"
            self._dst.enqueue(p)
            return

        pixel_format = ""
        try:
            pixel_format = p.format.name.lower()
        except Exception:
            pass

        if pixel_format != "cuda":
            p.metadata["pytorch_filter_status"] = "skipped_non_cuda_frame"
            self._dst.enqueue(p)
            return

        t0 = time.perf_counter()
        try:
            p.metadata["pytorch_filter_status"] = self._apply_filter(p)
        except Exception as exc:
            p.metadata["pytorch_filter_status"] = f"error_{type(exc).__name__}"
        processing_ms = (time.perf_counter() - t0) * 1000.0
        p.metadata["pytorch_filter_ms"] = f"{processing_ms:.3f}"

        if (self._frame_index % self._log_every_n) == 0:
            print(
                "PytorchCUDANode: "
                f"frame={self._frame_index} "
                f"status={p.metadata['pytorch_filter_status']} "
                f"proc_ms={processing_ms:.3f}"
            )

        self._dst.enqueue(p)


node = PytorchCUDANode({
    "src": "scaled_to_python",
    "dst": "scaled_from_python",
    "group": "g1",
    "name": "python-test-node",
    "sample_every_n": 1,
    "log_every_n": 1,
})

avp.addNode(node)

print("Starting groups")
avp.group("in").startNodes()
avp.group("g1").startNodes()

while True:
    time.sleep(1)
    avp.heartbeat()
