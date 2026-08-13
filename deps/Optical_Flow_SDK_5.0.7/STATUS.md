# NVOF camera-motion status

The dense-CUDA NVOF integration is implemented as the `cuda_camera_motion`
node in `src/nodes/neural_net/scene_cut/cuda_camera_motion.cpp`.

The node is built when `HAVE_CUDA=1`, `NEURAL_NET=1`, and `HAVE_NVOF=1`.
Its dense optical-flow engine is provided by the NVIDIA driver through
`libnvidia-opticalflow.so`; the public headers in this directory provide the
compile-time API. Setting `HAVE_NVCC=1` additionally enables the GPU IRLS
affine estimator.

The probe programs under `probe/` remain useful for checking driver and
hardware support independently of avplumber. The separate NvOFFRUC frame-rate
interpolation integration is unrelated to this dense camera-motion path.
