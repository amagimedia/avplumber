# Public TensorRT installation for demos

Checked 2026-09-07. Recommendation: keep the existing media demos independent of TensorRT. For inference, use a pinned NVIDIA CUDA/Ubuntu container and obtain TensorRT from NVIDIA's public package repository. To minimize deployment size, copy the required shared libraries from a build stage into a separate runtime image. This gives Fedora hosts the same application dependencies. The host still needs a compatible NVIDIA driver and NVIDIA Container Toolkit. [NVIDIA container installation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-container.html)

## Fedora 44

NVIDIA publishes a [Fedora 44 CUDA repository](https://developer.download.nvidia.com/compute/cuda/repos/fedora44/x86_64/), but its index currently contains no `nvinfer` or `tensorrt` packages. TensorRT 11.2.1's RPM instructions list RHEL/Rocky 8, 9 and 10 as supported operating systems; the introductory mention of Fedora does not establish Fedora 44 support. Do not instruct users to add a RHEL repository to Fedora. [NVIDIA RPM installation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-rpm.html)

For a native installation, NVIDIA's Linux tar distribution provides headers and libraries under a chosen directory, without root access. Select the CPU architecture and CUDA variant on the [official TensorRT download page](https://developer.nvidia.com/tensorrt/download). This is a public alternative to a custom bundle, but Fedora 44 is absent from the documented supported OS list; native compatibility needs a build and inference check. [NVIDIA tar installation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-tar.html)

## What avplumber needs

The [Makefile](../../Makefile) enables TensorRT with `HAVE_CUDA=1 NEURAL_NET=1 HAVE_TENSORRT=1 HAVE_NVCC=1`, accepts `TENSORRT_ROOT` for the SDK's `include` and `lib` directories, and links `libnvinfer` plus `libnvinfer_plugin`. The [inference runtime](../../src/nodes/neural_net/common/infer_trt_base.cpp) deserializes engines and calls `enqueueV3`; it does not build engines. The [mixer Dockerfile](../../demos/mixer/Dockerfile), [CUDA overlay Dockerfile](../../demos/cuda-overlay/Dockerfile), and [DMA-BUF CUDA Dockerfile](../../demos/dmabuf-browser/consumer/Dockerfile.cuda) currently disable TensorRT.

For a smaller inference image, keep headers, the ONNX parser, `trtexec`, and architecture-specific builder resources in the build stage. Copy the standard runtime and plugin shared libraries, preserving symlinks, plus required CUDA dependencies into the final image. Validate it by deserializing and executing an engine; `ldd` alone misses dynamically loaded dependencies.

Do not assume the `tensorrt-libs` metapackage is minimal. In NVIDIA's public Ubuntu 22.04 package index, version `10.15.1.29-1+cuda12.9` depends on the full, lean and dispatch runtimes, plugins, the ONNX parser, and Windows builder resources. `libnvinfer10` alone reports about 3.3 GiB installed. Downloading pinned official packages in a build stage and selecting deployment libraries reproduces a trimmed SDK without a privately hosted archive. The network installer can resolve CUDA dependencies without installing the entire CUDA Toolkit. [NVIDIA package index](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/Packages.gz), [NVIDIA package installation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-debian.html#installation-steps-network-repo-method)

An SM89-only builder selection targets L4-class GPUs. Engine-build environments need resources for their target GPU; a bundle trimmed for L4 is not a generic T4 build setup. Inference-only packaging and engine-building support are separate decisions. [NVIDIA builder resource loading](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/c-api-docs.html)

Python wheels are an official, convenient Python installation, but omit C++ headers and `trtexec`; installing `tensorrt` with pip does not complete this project's native build setup. [NVIDIA pip installation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html)

Lean/dispatch runtimes are smaller deployment options for suitably built engines. They are not drop-in replacements for avplumber's current standard-library linkage. A switch needs deliberate linking, engine-build and loading changes; version-compatible engines can embed a lean runtime, and loading that embedded code requires explicit runtime permission currently absent from avplumber. [NVIDIA engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)

## Version choice and verification

The versioned support-matrix data lists T4/SM 7.5 for both TensorRT 10.13.3 and 11.2.1. TensorRT 11 does not inherently exclude T4. However, 11 changes APIs and engine-building behavior, including removing weak typing. Select and test one exact SDK version; do not silently follow `latest`. [Support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html), [11.0 release notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.0.0.html)

Build demo engines on the target GPU with the selected runtime version: ordinary engines have version and GPU compatibility restrictions. Public download availability is verified here; native Fedora 44 builds, TensorRT 11 source compatibility, and inference execution remain unverified. [Engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)
