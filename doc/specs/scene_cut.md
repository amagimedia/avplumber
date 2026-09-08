# scene_cut — Scene and Camera Motion Nodes

- `luma_diff` computes a CUDA luma difference metric and attaches frame metadata.
- `hog_diff` computes a CUDA HOG difference metric.
- `cuda_infer_scene_cut_onnx` runs a pairwise TensorRT scene-cut model.
- `cuda_camera_motion` estimates background camera motion with NVIDIA Optical Flow when the dense SDK headers and runtime library are available.

All native nodes in this module require `HAVE_CUDA=1`. `luma_diff` and
`hog_diff` additionally require `HAVE_NVCC=1`; the ONNX detector requires
`HAVE_NVCC=1 HAVE_TENSORRT=1 NEURAL_NET=1`; and `cuda_camera_motion` requires
`HAVE_NVOF=1` plus the dense NVOF headers.

Metric nodes are deliberately separate from policy: downstream graphs decide
thresholds, confirmation windows, and whether camera motion vetoes a cut.
