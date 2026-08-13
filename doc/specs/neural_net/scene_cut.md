# neural_net/scene_cut — Scene and Camera Motion Nodes

- `luma_diff` computes a CUDA luma difference metric and attaches frame metadata.
- `hog_diff` computes a CUDA HOG difference metric.
- `cuda_infer_scene_cut_onnx` runs a pairwise TensorRT scene-cut model.
- `cuda_camera_motion` estimates background camera motion with NVIDIA Optical Flow when the dense SDK headers and runtime library are available.

Metric nodes are deliberately separate from policy: downstream graphs decide
thresholds, confirmation windows, and whether camera motion vetoes a cut.
