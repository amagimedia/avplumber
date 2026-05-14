#!/usr/bin/env python3
# Simple node that use pytorch for person detection
# It uses the fasterrcnn_mobilenet_v3_large_320_fpn model
# It detects persons in the frame and draws bounding boxes around them
# It writes the number of persons detected and the detection time to the metadata
# It writes the bounding boxes to the frame
# It writes the frame to the output


import time
import sys
import os
import ctypes
sys.path.append("../..")

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None
try:
    from torchvision.models.detection import (
        fasterrcnn_mobilenet_v3_large_320_fpn,
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    )
except Exception:
    fasterrcnn_mobilenet_v3_large_320_fpn = None
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights = None

import pyplumber
from pyplumber.node import (
    PythonNode,
    InputRec,
    Demux,
    DecVideo,
    RescaleVideo,
    FilterVideo,
    AssumeVideoFormat,
    EncVideo,
    Mux,
    Output,
)


INPUT_URL = os.environ.get("AVP_INPUT", "input.mp4")
OUTPUT_URL = os.environ.get("AVP_OUTPUT", "output.mp4")
OUTPUT_FORMAT = os.environ.get("AVP_OUTPUT_FORMAT", "mpegts")

_FILTER_GRAPH = (
    "drawtext=text='MSG\\: %{metadata\\:msg}\n"
    "FRAME\\: %{metadata\\:frame_no}\n"
    "VIDEO\\: %{metadata\\:video_ts}\n"
    "PTS\\: %{metadata\\:video_pts}\n"
    "INPUT\\: %{metadata\\:input_ts}\n"
    "OUTPUT\\: %{metadata\\:output_ts}\n"
    "WALLCLOCK\\: %{metadata\\:wallclock_ts}': x=20: y=100: box=1: boxborderw=2"
)

avp = pyplumber.AVPlumber()

input_rec = InputRec({
    "url": INPUT_URL,
    "dst": "in10",
    "group": "in",
    "auto_restart": "off",
    "name": "input",
    "timeout": -1,
    "preseek": 0,
    "seek_table": "",
    "ts_offsets": "",
    "timestamp_source": "wallclock",
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
})
rescale = RescaleVideo({
    "dst_width": 1280,
    "dst_height": 720,
    "flags": ["SWS_LANCZOS"],
    "dst_pixel_format": "yuv420p",
    "auto_restart": "panic",
    "name": "scale",
    "group": "g1",
    "src": "Video_Decode_out",
    "dst": "scaled_to_python",
})
filter_video = FilterVideo({
    "graph": _FILTER_GRAPH,
    "auto_restart": "group",
    "name": "Video_Filter",
    "group": "g1",
    "src": "scaled",
    "dst": "v_to_encoder",
})
assume_video_format = AssumeVideoFormat({
    "width": 1280,
    "height": 720,
    "pixel_format": "yuv420p",
    "group": "g1",
    "src": "v_to_encoder",
    "dst": "v_to_encoder_assumed",
    "auto_restart": "group",
})
enc_video = EncVideo({
    "auto_restart": "group",
    "name": "Video_Encode",
    "group": "g1",
    "src": "v_to_encoder_assumed",
    "dst": "v_outenc",
    "codec": "libx264",
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
    rescale,
    filter_video,
    assume_video_format,
    enc_video,
    mux,
    out,
):
    avp.addNode(node)


class PyTorchNode(PythonNode):
    def __init__(self, args):
        super().__init__(args)
        self._frame_count = 0
        self._detect_every_n = max(1, int(args.get("detect_every_n", 5)))
        self._person_score_threshold = float(args.get("person_score_threshold", 0.60))
        self._max_person_boxes = max(1, int(args.get("max_person_boxes", 8)))
        self._box_thickness = max(1, int(args.get("box_thickness", 4)))
        self._detector = None
        self._detector_ready = False
        self._detector_error = None
        self._last_person_boxes = []
        self._last_detection_time_ms = 0.0
        self._last_detection_frame = 0

    def _plane_u8(self, ptr: int, stride: int, rows: int):
        if ptr <= 0 or stride <= 0 or rows <= 0:
            return None
        size = int(stride) * int(rows)
        c_buf = (ctypes.c_uint8 * size).from_address(int(ptr))
        return torch.frombuffer(memoryview(c_buf), dtype=torch.uint8, count=size).view(rows, stride)

    def _ensure_detector(self) -> bool:
        if self._detector_ready:
            return True
        if torch is None or F is None or fasterrcnn_mobilenet_v3_large_320_fpn is None:
            self._detector_error = "missing_torch_or_torchvision"
            return False
        try:
            weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
            self._detector = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
            self._detector.eval()
            self._detector_ready = True
            self._detector_error = None
            return True
        except Exception as exc:
            self._detector_error = type(exc).__name__
            return False

    def _frame_to_rgb(self, frame):
        height = int(frame.height)
        width = int(frame.width)
        linesize = frame.linesize
        data_ptr = frame.data_ptr
        y_stride = int(linesize[0]) if linesize else 0
        y_ptr = int(data_ptr[0]) if data_ptr else 0

        y_plane = self._plane_u8(y_ptr, y_stride, height)
        if y_plane is None:
            raise RuntimeError("invalid_luma_plane")

        uv_h = (height + 1) // 2
        uv_w = (width + 1) // 2
        u_stride = int(linesize[1]) if len(linesize) > 1 else 0
        v_stride = int(linesize[2]) if len(linesize) > 2 else 0
        u_ptr = int(data_ptr[1]) if len(data_ptr) > 1 else 0
        v_ptr = int(data_ptr[2]) if len(data_ptr) > 2 else 0
        u_plane = self._plane_u8(u_ptr, u_stride, uv_h)
        v_plane = self._plane_u8(v_ptr, v_stride, uv_h)
        if u_plane is None or v_plane is None:
            raise RuntimeError("invalid_chroma_plane")

        y = y_plane[:, :width].to(torch.float32) / 255.0
        u = u_plane[:, :uv_w].to(torch.float32) / 255.0
        v = v_plane[:, :uv_w].to(torch.float32) / 255.0
        u = u.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)[:height, :width]
        v = v.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)[:height, :width]
        r = torch.clamp(y + 1.4020 * (v - 0.5), 0.0, 1.0)
        g = torch.clamp(y - 0.3441 * (u - 0.5) - 0.7141 * (v - 0.5), 0.0, 1.0)
        b = torch.clamp(y + 1.7720 * (u - 0.5), 0.0, 1.0)
        return torch.stack((r, g, b), dim=0)

    def _detect_person_boxes(self, frame):
        detect_t0 = time.perf_counter()
        rgb = self._frame_to_rgb(frame)
        _, height, width = rgb.shape
        scale = min(640.0 / float(max(height, width)), 1.0)
        if scale < 1.0:
            infer_h = max(192, int(height * scale))
            infer_w = max(192, int(width * scale))
            infer_rgb = F.interpolate(
                rgb.unsqueeze(0),
                size=(infer_h, infer_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            infer_rgb = rgb
            infer_h = height
            infer_w = width

        with torch.inference_mode():
            outputs = self._detector([infer_rgb])[0]

        labels = outputs["labels"]
        scores = outputs["scores"]
        boxes = outputs["boxes"]
        mask = (labels == 1) & (scores >= self._person_score_threshold)
        boxes = boxes[mask][: self._max_person_boxes].cpu()
        if boxes.numel() == 0:
            return [], (time.perf_counter() - detect_t0) * 1000.0

        scale_x = float(width) / float(infer_w)
        scale_y = float(height) / float(infer_h)
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        return boxes.tolist(), (time.perf_counter() - detect_t0) * 1000.0

    def _draw_person_boxes(self, frame, boxes):
        height = int(frame.height)
        width = int(frame.width)
        linesize = frame.linesize
        data_ptr = frame.data_ptr
        y_plane = self._plane_u8(int(data_ptr[0]), int(linesize[0]), height)
        if y_plane is None:
            return

        uv_h = (height + 1) // 2
        uv_w = (width + 1) // 2
        u_plane = self._plane_u8(int(data_ptr[1]), int(linesize[1]), uv_h) if len(data_ptr) > 1 and len(linesize) > 1 else None
        v_plane = self._plane_u8(int(data_ptr[2]), int(linesize[2]), uv_h) if len(data_ptr) > 2 and len(linesize) > 2 else None

        t = self._box_thickness
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width, x2))
            y2 = max(0, min(height, y2))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue

            y_plane[y1:min(y1 + t, y2), x1:x2] = 235
            y_plane[max(y2 - t, y1):y2, x1:x2] = 235
            y_plane[y1:y2, x1:min(x1 + t, x2)] = 235
            y_plane[y1:y2, max(x2 - t, x1):x2] = 235

            if u_plane is None or v_plane is None:
                continue

            ux1, ux2 = max(0, x1 // 2), min(uv_w, (x2 + 1) // 2)
            uy1, uy2 = max(0, y1 // 2), min(uv_h, (y2 + 1) // 2)
            ut = max(1, t // 2)
            u_plane[uy1:min(uy1 + ut, uy2), ux1:ux2] = 84
            u_plane[max(uy2 - ut, uy1):uy2, ux1:ux2] = 84
            u_plane[uy1:uy2, ux1:min(ux1 + ut, ux2)] = 84
            u_plane[uy1:uy2, max(ux2 - ut, ux1):ux2] = 84
            v_plane[uy1:min(uy1 + ut, uy2), ux1:ux2] = 255
            v_plane[max(uy2 - ut, uy1):uy2, ux1:ux2] = 255
            v_plane[uy1:uy2, ux1:min(ux1 + ut, ux2)] = 255
            v_plane[uy1:uy2, max(ux2 - ut, ux1):ux2] = 255

    def process(self):
        p = self._src.get()
        if not p:
            return

        self._frame_count += 1
        p.metadata["pytorch_frame"] = self._frame_count

        if torch is None or F is None or not hasattr(torch, "frombuffer"):
            p.metadata["pytorch_status"] = "no_torch_buffer_api"
            p.metadata["msg"] = "Torch unavailable for torchvision detection"
        elif not self._ensure_detector():
            p.metadata["pytorch_status"] = "detector_unavailable"
            p.metadata["msg"] = f"Detector init failed: {self._detector_error}"
        else:
            try:
                if self._frame_count % self._detect_every_n == 0 or not self._last_person_boxes:
                    self._last_person_boxes, self._last_detection_time_ms = self._detect_person_boxes(p)
                    self._last_detection_frame = self._frame_count
                self._draw_person_boxes(p, self._last_person_boxes)
                person_count = len(self._last_person_boxes)
                p.metadata["pytorch_status"] = "torchvision_person_detector"
                p.metadata["person_count"] = person_count
                p.metadata["detection_time_ms"] = f"{self._last_detection_time_ms:.2f}"
                p.metadata["msg"] = (
                    f"Persons detected: {person_count} | "
                    f"det_time_ms={self._last_detection_time_ms:.2f} "
                    f"(frame {self._last_detection_frame})"
                )
            except Exception as exc:
                p.metadata["pytorch_status"] = f"error_{type(exc).__name__}"
                p.metadata["msg"] = "Torchvision person detection failed"

        if isinstance(self._dst, dict):
            for dst in self._dst.values():
                dst.enqueue(p)
        else:
            self._dst.enqueue(p)


node = PyTorchNode({
    "src": "scaled_to_python",
    "dst": ["scaled"],
    "group": "g1",
    "name": "python-test-node",
    "detect_every_n": 10,
    "person_score_threshold": 0.60,
    "max_person_boxes": 10,
    "box_thickness": 2,
})

avp.addNode(node)

print("Starting groups")
avp.group("in").startNodes()
avp.group("g1").startNodes()

while True:
    time.sleep(1)
    avp.heartbeat()
