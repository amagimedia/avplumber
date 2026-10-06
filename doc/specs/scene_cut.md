# scene_cut — Scene and Camera Motion Nodes

- `luma_diff` computes a CUDA luma difference metric and attaches frame metadata.
- `hog_diff` computes a CUDA HOG difference metric.
- `cuda_camera_motion` estimates background camera motion with NVIDIA Optical Flow when the dense SDK headers and runtime library are available.

All native nodes in this module require `HAVE_CUDA=1`. `luma_diff` and
`hog_diff` additionally require `HAVE_NVCC=1`; `cuda_camera_motion` requires
`HAVE_NVOF=1` plus the dense NVOF headers.

The learned pairwise detector (`cuda_infer_scene_cut_onnx`) is not part of this
module: its model is proprietary, so the node lives in the private
amagi-avplumber-nodes distribution and is contributed through `EXTRA_NODES_MK`.

Metric nodes are deliberately separate from policy: downstream graphs decide
thresholds, confirmation windows, and whether camera motion vetoes a cut.

## Camera-motion input storage

`cuda_camera_motion` accepts linear CUDA frames with NV12, YUV420P, YUVJ420P or
GRAY8 storage, and native CUarray frames with NV12 storage. Storage is detected
from each frame; no node parameter is needed. Request `pixel_format: "cuarray"`
and decoder option `hwaccel_flags: "unsafe_output"` on a compatible NVDEC input.
CUarray support requires FFmpeg's `AV_PIX_FMT_CUARRAY` API and a CUDA driver with
`cuArrayGetPlane`; builds against older FFmpeg retain linear input support.

The node copies only luma directly into the existing NVOF GRAYSCALE8 input and
reference buffers. It does not allocate a linear copy of the source image or
copy its chroma. The source frame, storage handle and timestamps are forwarded
unchanged, with camera-motion metadata attached. This preserves the existing
GPU luma-copy cost; it does not make NVOF input zero-copy.

Copies and NVOF execution use the source device's producer stream. The node
retains that device and the in-flight frame until GPU reads finish. Producer
streams may change within one CUDA context; changing contexts is rejected.
CUarray P010/P210 is unsupported: NVOF's current input path expects 8-bit luma.
In non-strict mode unsupported formats are forwarded with
`status: "unsupported_sw_format"`; they are not analyzed.

`tests/cuda/nvdec/camera_motion.py` compares linear and CUarray input using a
finite 8-bit HEVC fixture. It checks translation, CPU IRLS and GPU IRLS backends,
exact metadata and PTS, unchanged storage handles, filtered linear input on the
decoder's private stream, alternating linear/CUarray producers, EOF and repeated
teardown. An optional Main10 fixture checks the non-strict unsupported result.

## Luma-difference input storage

`luma_diff` accepts the same linear formats and NV12 CUarray input. Its reduction
kernel samples CUarray luma directly through an integer texture; linear input
uses the existing pitched-pointer kernel. Both variants share the reduction
implementation. The existing `L + 1` history planes, scratch plane and copies
are unchanged: CUarray support adds no image buffer or full-image copy.
Lookahead, metadata, timestamps and forwarded storage handles are unchanged.

Camera motion and luma difference share version-gated array-plane lookup and
luma-copy source validation. Luma difference and the compositor share a texture
cache that retains the FFmpeg fixed array pool, rather than individual decoder
surfaces. Luma difference creates only the luma texture. GPU work uses each
frame's producer stream and retains its device and frame until reads finish.
`hog_diff` still requires linear CUDA input.

`tests/cuda/nvdec/luma_diff.py` compares every score against CPU-decoded luma at
lookahead 0, 2, 4 and 16, including ring wraparound and EOF tails. It checks exact
linear/CUarray metadata and PTS, unchanged source handles, private-stream linear
input, alternating producer streams, repeated teardown and optional Main10
rejection. Both metric test scripts support `--linear-only --report <path>` on a reference build and
`--reference <path>` on the changed build to detect linear regressions.
