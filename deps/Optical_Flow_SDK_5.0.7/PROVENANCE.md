# NVIDIA Optical Flow SDK — headers-only (dense CUDA API)

This is **not** the full Optical Flow SDK bundle. It contains only the two
public, redistributable dense-API headers needed to compile against the NVIDIA
Optical Flow **CUDA** interface:

- `NvOFInterface/nvOpticalFlowCommon.h`
- `NvOFInterface/nvOpticalFlowCuda.h`

## Source

Fetched from NVIDIA's public headers-only repository:

- Repo: https://github.com/NVIDIA/NVIDIAOpticalFlowSDK
- Commit: `edb50da3cf849840d680249aa6dbef248ebce2ca`
- Archive md5: `a73cd48b18dcc0cc8933b30796074191`
  (matches the pin used by opencv_contrib `modules/cudaoptflow/CMakeLists.txt`,
  which confirms these are the genuine, untampered NVIDIA headers.)

## Why headers-only

The dense optical-flow **runtime engine** ships with the NVIDIA driver as
`libnvidia-opticalflow.so.1` (exports `NvOFAPICreateInstanceCuda`). No SDK
binary or FRUC redistributable is required for the dense CUDA path — we compile
against these headers and link against the driver library, which the NVIDIA
container runtime exposes inside the container at runtime (`--gpus all` /
`nvidia.com/gpu`).

The FRUC node (`nvof_fruc.cpp`) is a different API that additionally needs
`libNvOFFRUC.so` at runtime; it is intentionally NOT used by the dense
camera-motion path.
