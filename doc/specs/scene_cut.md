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
