#!/usr/bin/env python3
# Example node that runs a torchvision pre-trained detector on CUDA video frames.

import time
import sys
import os
sys.path.append("../../")

try:
    import torch
    import torchvision
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )
except Exception:
    torch = None
    torchvision = None
    FasterRCNN_ResNet50_FPN_V2_Weights = None
    fasterrcnn_resnet50_fpn_v2 = None

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

MODEL_DTYPE = torch.float32 if torch is not None else None

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


class TorchvisionDetectionNode(PythonNode):
    def __init__(self, args):
        super().__init__(args)
        self._frame_index = 0
        self._sample_every_n = max(1, int(args.get("sample_every_n", 15)))
        self._log_every_n = max(1, int(args.get("log_every_n", 30)))
        self._score_threshold = float(args.get("score_threshold", 0.65))
        self._max_detections = max(1, int(args.get("max_detections", 10)))
        self._torch_warning_logged = False
        self._device = None
        self._model = None
        self._categories = []
        self._last_boxes = None
        self._last_labels = None
        self._last_scores = None

        if torch is not None and torchvision is not None:
            try:
                if torch.cuda.is_available():
                    self._device = torch.device("cuda:0")
                    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                    self._categories = weights.meta.get("categories", [])
                    self._model = fasterrcnn_resnet50_fpn_v2(weights=weights)
                    self._model.to(device=self._device, dtype=MODEL_DTYPE)
                    self._model.eval()
            except Exception:
                self._device = None
                self._model = None
                self._categories = []

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

    def _label_name(self, label: int) -> str:
        if 0 <= label < len(self._categories):
            return self._categories[label]
        return str(label)

    def _run_detection(self, y_plane_u8):
        # COCO pre-trained detectors expect a 3-channel image tensor in 0..1.
        image = y_plane_u8.to(MODEL_DTYPE).mul(1.0 / 255.0)
        image = image.unsqueeze(0).repeat(3, 1, 1)

        with torch.inference_mode():
            output = self._model([image])[0]

        scores = output["scores"]
        keep = torch.nonzero(scores >= self._score_threshold, as_tuple=False).flatten()
        keep = keep[: self._max_detections]
        self._last_boxes = output["boxes"][keep].detach()
        self._last_labels = output["labels"][keep].detach()
        self._last_scores = scores[keep].detach()
        return int(keep.numel())

    def _draw_detections(self, y_plane_u8) -> int:
        if self._last_boxes is None:
            return 0

        height, width = y_plane_u8.shape
        thickness = 3
        boxes = self._last_boxes.round().to(torch.int64)
        drawn = 0

        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width - 1, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            t = min(thickness, y2 - y1 + 1, x2 - x1 + 1)
            y_plane_u8[y1 : y1 + t, x1 : x2 + 1] = 255
            y_plane_u8[y2 - t + 1 : y2 + 1, x1 : x2 + 1] = 255
            y_plane_u8[y1 : y2 + 1, x1 : x1 + t] = 255
            y_plane_u8[y1 : y2 + 1, x2 - t + 1 : x2 + 1] = 255
            drawn += 1

        return drawn

    def _apply_detection(self, frame, run_model: bool) -> tuple[str, int]:
        y_plane_u8 = self._get_luma_plane(frame)
        if y_plane_u8 is None:
            return "skipped_invalid_frame", 0

        detected = self._run_detection(y_plane_u8) if run_model else 0
        drawn = self._draw_detections(y_plane_u8)
        torch.cuda.current_stream(device=self._device).synchronize()
        status = "detected_with_torchvision" if run_model else "drew_cached_detections"
        return status, detected if run_model else drawn

    def process(self):
        p = self._src.get()
        if not p:
            return

        #self._dst.enqueue(p)
        #return

        self._frame_index += 1
        p.metadata["torchvision_detection_frame"] = self._frame_index

        if self._device is None or self._model is None:
            if not self._torch_warning_logged:
                print("TorchvisionDetectionNode: CUDA torch/torchvision is unavailable, forwarding frames unchanged.")
                self._torch_warning_logged = True
            p.metadata["torchvision_detection_status"] = "skipped_no_torchvision_cuda"
            self._dst.enqueue(p)
            return

        pixel_format = ""
        try:
            pixel_format = p.format.name.lower()
        except Exception:
            pass

        if pixel_format != "cuda":
            p.metadata["torchvision_detection_status"] = "skipped_non_cuda_frame"
            self._dst.enqueue(p)
            return

        run_model = (self._frame_index % self._sample_every_n) == 0
        t0 = time.perf_counter()
        detection_count = 0
        try:
            status, detection_count = self._apply_detection(p, run_model=run_model)
            p.metadata["torchvision_detection_status"] = status
            p.metadata["torchvision_detection_count"] = str(detection_count)
        except Exception as exc:
            p.metadata["torchvision_detection_status"] = f"error_{type(exc).__name__}"
        processing_ms = (time.perf_counter() - t0) * 1000.0
        p.metadata["torchvision_detection_ms"] = f"{processing_ms:.3f}"

        if (self._frame_index % self._log_every_n) == 0:
            label_preview = ""
            if self._last_labels is not None and self._last_scores is not None:
                labels = self._last_labels[:3].tolist()
                scores = self._last_scores[:3].tolist()
                label_preview = " labels=" + ",".join(
                    f"{self._label_name(int(label))}:{float(score):.2f}"
                    for label, score in zip(labels, scores)
                )
            print(
                "TorchvisionDetectionNode: "
                f"frame={self._frame_index} "
                f"status={p.metadata['torchvision_detection_status']} "
                f"detections={detection_count} "
                f"proc_ms={processing_ms:.3f}"
                f"{label_preview}"
            )

        self._dst.enqueue(p)


node = TorchvisionDetectionNode({
    "src": "scaled_to_python",
    "dst": "scaled_from_python",
    "group": "g1",
    "name": "torchvision-detection-node",
    "sample_every_n": 1,
    "log_every_n": 1,
    "score_threshold": 0.65,
    "max_detections": 10,
})

avp.addNode(node)

print("Starting groups")
avp.group("in").startNodes()
avp.group("g1").startNodes()

while True:
    time.sleep(1)
    avp.heartbeat()

