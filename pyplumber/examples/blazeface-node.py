#!/usr/bin/env python3
import time
import sys
import os
import json
import urllib.request
from pathlib import Path
sys.path.append("../../")

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    np = None
    torch = None
    nn = None
    F = None

import pyplumber
from pyplumber.node import (
    PythonNode,
    InputRec,
    Demux,
    DecVideo,
    ForceFPS,
    RealtimeVideoFrame,
    Split,
    FilterVideo,
    JoinMetadata,
    DrawBBox,
    AssumeVideoFormat,
    EncVideo,
    Mux,
    Output,
)

INPUT_URL = os.environ.get("AVP_INPUT", "input.mp4")
OUTPUT_URL = os.environ.get("AVP_OUTPUT", "output.mp4")
OUTPUT_FORMAT = os.environ.get("AVP_OUTPUT_FORMAT", "mpegts")
USE_REALTIME = os.environ.get("AVP_USE_REALTIME", "1") == "1"
BLAZEFACE_DIR = Path(os.environ.get("AVP_BLAZEFACE_DIR", Path(__file__).resolve().parent / "models" / "blazeface"))
BLAZEFACE_WEIGHTS_URL = "https://github.com/hollance/BlazeFace-PyTorch/raw/master/blazeface.pth"
BLAZEFACE_ANCHORS_URL = "https://github.com/hollance/BlazeFace-PyTorch/raw/master/anchors.npy"

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


def _download_if_missing(path: Path, url: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"BlazeFaceNode: downloading {path.name} from {url}")
    urllib.request.urlretrieve(url, path)


def ensure_blazeface_assets() -> tuple[Path, Path]:
    weights_path = BLAZEFACE_DIR / "blazeface.pth"
    anchors_path = BLAZEFACE_DIR / "anchors.npy"
    _download_if_missing(weights_path, BLAZEFACE_WEIGHTS_URL)
    _download_if_missing(anchors_path, BLAZEFACE_ANCHORS_URL)
    return weights_path, anchors_path


BlazeFace = None


if nn is not None and F is not None:
    class BlazeBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
            super().__init__()
            self.stride = stride
            self.channel_pad = out_channels - in_channels

            if stride == 2:
                self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
                padding = 0
            else:
                padding = (kernel_size - 1) // 2

            self.convs = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=in_channels,
                    bias=True,
                ),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
            )
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            if self.stride == 2:
                h = F.pad(x, (0, 2, 0, 2), "constant", 0)
                x = self.max_pool(x)
            else:
                h = x

            if self.channel_pad > 0:
                x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

            return self.act(self.convs(h) + x)


    class BlazeFaceModel(nn.Module):
        """BlazeFace model architecture adapted from hollance/BlazeFace-PyTorch."""

        def __init__(self):
            super().__init__()
            self.num_classes = 1
            self.num_anchors = 896
            self.num_coords = 16
            self.score_clipping_thresh = 100.0
            self.x_scale = 128.0
            self.y_scale = 128.0
            self.h_scale = 128.0
            self.w_scale = 128.0
            self.min_score_thresh = float(os.environ.get("AVP_BLAZEFACE_MIN_SCORE", "0.75"))
            self.min_suppression_threshold = float(os.environ.get("AVP_BLAZEFACE_NMS", "0.3"))

            self.backbone1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2, padding=0, bias=True),
                nn.ReLU(inplace=True),
                BlazeBlock(24, 24),
                BlazeBlock(24, 28),
                BlazeBlock(28, 32, stride=2),
                BlazeBlock(32, 36),
                BlazeBlock(36, 42),
                BlazeBlock(42, 48, stride=2),
                BlazeBlock(48, 56),
                BlazeBlock(56, 64),
                BlazeBlock(64, 72),
                BlazeBlock(72, 80),
                BlazeBlock(80, 88),
            )
            self.backbone2 = nn.Sequential(
                BlazeBlock(88, 96, stride=2),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
            )
            self.classifier_8 = nn.Conv2d(88, 2, 1, bias=True)
            self.classifier_16 = nn.Conv2d(96, 6, 1, bias=True)
            self.regressor_8 = nn.Conv2d(88, 32, 1, bias=True)
            self.regressor_16 = nn.Conv2d(96, 96, 1, bias=True)

        def forward(self, x):
            x = F.pad(x, (1, 2, 1, 2), "constant", 0)
            batch_size = x.shape[0]

            x = self.backbone1(x)
            h = self.backbone2(x)

            c1 = self.classifier_8(x).permute(0, 2, 3, 1).reshape(batch_size, -1, 1)
            c2 = self.classifier_16(h).permute(0, 2, 3, 1).reshape(batch_size, -1, 1)
            r1 = self.regressor_8(x).permute(0, 2, 3, 1).reshape(batch_size, -1, 16)
            r2 = self.regressor_16(h).permute(0, 2, 3, 1).reshape(batch_size, -1, 16)
            return torch.cat((r1, r2), dim=1), torch.cat((c1, c2), dim=1)

        def load_weights(self, path: Path) -> None:
            self.load_state_dict(torch.load(path, map_location=self._device()))
            self.eval()

        def load_anchors(self, path: Path) -> None:
            self.anchors = torch.tensor(np.load(path), dtype=torch.float32, device=self._device())
            assert self.anchors.shape == (self.num_anchors, 4)

        def _device(self):
            return self.classifier_8.weight.device

        def predict_on_batch(self, x):
            assert x.shape[1:] == (3, 128, 128)
            x = x.to(self._device()).float() / 127.5 - 1.0

            with torch.no_grad():
                raw_boxes, raw_scores = self(x)

            detections = self._tensors_to_detections(raw_boxes, raw_scores, self.anchors)
            filtered = []
            for image_detections in detections:
                faces = self._weighted_non_max_suppression(image_detections)
                if faces:
                    filtered.append(torch.stack(faces))
                else:
                    filtered.append(torch.zeros((0, 17), dtype=torch.float32, device=self._device()))
            return filtered

        def _tensors_to_detections(self, raw_box_tensor, raw_score_tensor, anchors):
            boxes = self._decode_boxes(raw_box_tensor, anchors)
            raw_score_tensor = raw_score_tensor.clamp(-self.score_clipping_thresh, self.score_clipping_thresh)
            scores = raw_score_tensor.sigmoid().squeeze(dim=-1)
            mask = scores >= self.min_score_thresh

            output = []
            for i in range(raw_box_tensor.shape[0]):
                image_boxes = boxes[i, mask[i]]
                image_scores = scores[i, mask[i]].unsqueeze(dim=-1)
                output.append(torch.cat((image_boxes, image_scores), dim=-1))
            return output

        def _decode_boxes(self, raw_boxes, anchors):
            boxes = torch.zeros_like(raw_boxes)

            x_center = raw_boxes[..., 0] / self.x_scale * anchors[:, 2] + anchors[:, 0]
            y_center = raw_boxes[..., 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]
            w = raw_boxes[..., 2] / self.w_scale * anchors[:, 2]
            h = raw_boxes[..., 3] / self.h_scale * anchors[:, 3]

            boxes[..., 0] = y_center - h / 2.0
            boxes[..., 1] = x_center - w / 2.0
            boxes[..., 2] = y_center + h / 2.0
            boxes[..., 3] = x_center + w / 2.0

            for k in range(6):
                offset = 4 + k * 2
                keypoint_x = raw_boxes[..., offset] / self.x_scale * anchors[:, 2] + anchors[:, 0]
                keypoint_y = raw_boxes[..., offset + 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]
                boxes[..., offset] = keypoint_x
                boxes[..., offset + 1] = keypoint_y

            return boxes

        def _weighted_non_max_suppression(self, detections):
            if len(detections) == 0:
                return []

            output = []
            remaining = torch.argsort(detections[:, 16], descending=True)
            while len(remaining) > 0:
                detection = detections[remaining[0]]
                first_box = detection[:4]
                other_boxes = detections[remaining, :4]
                ious = _overlap_similarity(first_box, other_boxes)

                mask = ious > self.min_suppression_threshold
                overlapping = remaining[mask]
                remaining = remaining[~mask]

                weighted_detection = detection.clone()
                if len(overlapping) > 1:
                    coordinates = detections[overlapping, :16]
                    scores = detections[overlapping, 16:17]
                    total_score = scores.sum()
                    weighted_detection[:16] = (coordinates * scores).sum(dim=0) / total_score
                    weighted_detection[16] = total_score / len(overlapping)

                output.append(weighted_detection)
            return output


    def _intersect(box_a, box_b):
        area_count = box_a.size(0)
        box_count = box_b.size(0)
        max_xy = torch.min(
            box_a[:, 2:].unsqueeze(1).expand(area_count, box_count, 2),
            box_b[:, 2:].unsqueeze(0).expand(area_count, box_count, 2),
        )
        min_xy = torch.max(
            box_a[:, :2].unsqueeze(1).expand(area_count, box_count, 2),
            box_b[:, :2].unsqueeze(0).expand(area_count, box_count, 2),
        )
        inter = torch.clamp(max_xy - min_xy, min=0)
        return inter[:, :, 0] * inter[:, :, 1]


    def _jaccard(box_a, box_b):
        inter = _intersect(box_a, box_b)
        area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
        area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
        union = area_a + area_b - inter
        return inter / union


    def _overlap_similarity(box, other_boxes):
        return _jaccard(box.unsqueeze(0), other_boxes).squeeze(0)


    BlazeFace = BlazeFaceModel


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
    "seek_table": "",
    "ts_offsets": "",
    "team": "@input-seek-team",
    "timestamp_source": "wallclock",
    "loop": True,
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
    "dst": "scaled_720p",
    "group": "g1",
    "name": "video_scaler",
    "auto_restart": "group",
    "dst_width": 1280,
    "dst_height": 720,
    "dst_pixel_format": "cuda",
    "hwaccel": "@gpu",
})
split_for_blazeface = Split({
    "src": "scaled_720p",
    "dst": ["scaled_to_encoder", "scaled_to_blazeface"],
    "group": "g1",
    "auto_restart": "group",
})
scale_blazeface = FilterVideo({
    "graph": "scale_cuda=w=128:h=72,pad_cuda=128:128:0:28",
    "src": "scaled_to_blazeface",
    "dst": "scaled_to_blazeface_128",
    "group": "g1",
    "name": "scale_blazeface",
    "auto_restart": "group",
    "dst_width": 128,
    "dst_height": 128,
    "dst_pixel_format": "cuda",
    "hwaccel": "@gpu",
})
join_blazeface_metadata = JoinMetadata({
    "src": ["scaled_to_encoder", "blazeface_detections"],
    "dst": "scaled_from_python",
    "group": "g1",
    "auto_restart": "group",
})
draw_blazeface_boxes = DrawBBox({
    "src": "scaled_from_python",
    "dst": "blazeface_boxes_drawn",
    "group": "g1",
    "name": "Draw_BlazeFace",
    "metadata_key": "blazeface_faces",
    "bbox_thickness": 3,
    "min_conf": 0.75,
    "allowed_labels": ["Face", "FaceKeypoint"],
    "label_colors": {"Face": "green", "FaceKeypoint": "yellow"},
    "width": 1280,
    "height": 720,
    "pixel_format": "cuda",
    "real_pixel_format": "nv12",
    "debug_log_every_n": 30,
    "auto_restart": "group",
})
force_fps = ForceFPS({
    "fps": "50",
    "group": "g1",
    "src": "blazeface_boxes_drawn",
    "dst": "forcedfps",
})
assume_video_format = AssumeVideoFormat({
    "width": 1280,
    "height": 720,
    "pixel_format": "cuda",
    "real_pixel_format": "nv12",
    "group": "g1",
    "src": "forcedfps",
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
    split_for_blazeface,
    scale_blazeface,
    join_blazeface_metadata,
    draw_blazeface_boxes,
    force_fps,
    assume_video_format,
    enc_video,
    mux,
    out,
):
    if node is not None:
        avp.addNode(node)


class BlazeFaceNode(PythonNode):
    def __init__(self, args):
        super().__init__(args)
        self._frame_index = 0
        self._sample_every_n = max(1, int(args.get("sample_every_n", 1)))
        self._log_every_n = max(1, int(args.get("log_every_n", 30)))
        self._torch_warning_logged = False
        self._device = None
        self._model = None

        if torch is not None and F is not None and np is not None and BlazeFace is not None:
            try:
                if torch.cuda.is_available():
                    self._device = torch.device("cuda:0")
                    weights_path, anchors_path = ensure_blazeface_assets()
                    self._model = BlazeFace().to(self._device)
                    self._model.load_weights(weights_path)
                    self._model.load_anchors(anchors_path)
                    self._model.eval()
            except Exception as exc:
                print(f"BlazeFaceNode: failed to initialize detector: {exc}")
                self._device = None
                self._model = None

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

    def _frame_to_blazeface_input(self, y_plane_u8):
        model_input = y_plane_u8.to(torch.float32).unsqueeze(0).unsqueeze(0)
        return model_input.repeat(1, 3, 1, 1)

    def _serialize_detections(self, detections):
        model_size = 128.0
        content_width = 128.0
        content_height = 72.0
        content_offset_x = 0.0
        content_offset_y = 28.0
        target_width = 1280.0
        target_height = 720.0
        point_box_size = 4.0
        point_radius = point_box_size * 0.5

        def clamp(value, low, high):
            return max(low, min(high, value))

        def map_x(normalized_x):
            model_x = normalized_x * model_size
            content_x = clamp(model_x - content_offset_x, 0.0, content_width)
            return content_x * target_width / content_width

        def map_y(normalized_y):
            model_y = normalized_y * model_size
            content_y = clamp(model_y - content_offset_y, 0.0, content_height)
            return content_y * target_height / content_height

        serialized = {
            "coord_space": "frame",
            "frame_width": int(target_width),
            "frame_height": int(target_height),
            "detections": [],
        }
        for detection in detections.detach().cpu().tolist():
            ymin = float(detection[0])
            xmin = float(detection[1])
            ymax = float(detection[2])
            xmax = float(detection[3])
            score = float(detection[16])
            keypoints = [
                [map_x(float(detection[offset])), map_y(float(detection[offset + 1]))]
                for offset in range(4, 16, 2)
            ]
            serialized["detections"].append({
                "cls": 0,
                "conf": score,
                "xyxy": [map_x(xmin), map_y(ymin), map_x(xmax), map_y(ymax)],
                "model_index": 0,
                "label": "Face",
                "keypoints": keypoints,
            })
            for point_x, point_y in keypoints:
                serialized["detections"].append({
                    "cls": 1,
                    "conf": score,
                    "xyxy": [
                        clamp(point_x - point_radius, 0.0, target_width),
                        clamp(point_y - point_radius, 0.0, target_height),
                        clamp(point_x + point_radius, 0.0, target_width),
                        clamp(point_y + point_radius, 0.0, target_height),
                    ],
                    "model_index": 0,
                    "label": "FaceKeypoint",
                })
        return serialized

    def _detect_faces(self, frame):
        y_plane_u8 = self._get_luma_plane(frame)
        if y_plane_u8 is None:
            return "skipped_invalid_frame", torch.zeros((0, 17), dtype=torch.float32, device=self._device)

        model_input = self._frame_to_blazeface_input(y_plane_u8)
        detections = self._model.predict_on_batch(model_input)[0]

        torch.cuda.current_stream(device=self._device).synchronize()
        return "detected_faces", detections

    def process(self):
        p = self._src.get()
        if not p:
            return

        #self._dst.enqueue(p)
        #return

        self._frame_index += 1
        p.metadata["blazeface_frame"] = self._frame_index

        if self._device is None or self._model is None:
            if not self._torch_warning_logged:
                print("BlazeFaceNode: CUDA torch/BlazeFace is unavailable, forwarding frames unchanged.")
                self._torch_warning_logged = True
            p.metadata["blazeface_status"] = "skipped_no_torch_cuda_or_model"
            self._dst.enqueue(p)
            return

        if self._frame_index % self._sample_every_n != 0:
            p.metadata["blazeface_status"] = "skipped_not_sampled"
            self._dst.enqueue(p)
            return

        pixel_format = ""
        try:
            pixel_format = p.format.name.lower()
        except Exception:
            pass

        if pixel_format != "cuda":
            p.metadata["blazeface_status"] = "skipped_non_cuda_frame"
            self._dst.enqueue(p)
            return

        t0 = time.perf_counter()
        face_count = 0
        try:
            status, detections = self._detect_faces(p)
            face_count = int(detections.shape[0])
            p.metadata["blazeface_status"] = status
            p.metadata["blazeface_faces"] = json.dumps(self._serialize_detections(detections))
            p.metadata["blazeface_face_count"] = face_count
        except Exception as exc:
            p.metadata["blazeface_status"] = f"error_{type(exc).__name__}"
        processing_ms = (time.perf_counter() - t0) * 1000.0
        p.metadata["blazeface_ms"] = f"{processing_ms:.3f}"

        if (self._frame_index % self._log_every_n) == 0:
            print(
                "BlazeFaceNode: "
                f"frame={self._frame_index} "
                f"status={p.metadata['blazeface_status']} "
                f"faces={face_count} "
                f"proc_ms={processing_ms:.3f}"
            )

        self._dst.enqueue(p)


node = BlazeFaceNode({
    "src": "scaled_to_blazeface_128",
    "dst": "blazeface_detections",
    "group": "g1",
    "name": "blazeface-node",
    "sample_every_n": 1,
    "log_every_n": 30,
})

avp.addNode(node)

print("Starting groups")
avp.group("in").startNodes()
avp.group("g1").startNodes()

while True:
    time.sleep(1)
    avp.heartbeat()

