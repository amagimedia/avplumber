## Build & test workflow
* If `nvidia-smi` is unavailable locally, do not attempt CUDA, TensorRT, NVENC, Janus GPU, or neural build/run checks on localhost; use the configured SSH NVIDIA host instead.
* Don't commit instance-specific build details (hostnames, IPs, SSH keys, CUDA/TensorRT paths).
* CUDA/neural builds: preserve feature flags `CUDA`, `neural_net common`, `neural_net specific`, `NVCC/PTX`, `TensorRT`. Keep FRUC disabled unless testing frame interpolation.
* pyplumber builds must use the same neural/CUDA/TensorRT feature set as the binary build.
* DMA-BUF overlay builds need DRM/GL flags so `ipc_dmabuf_source`, `drm_prime_to_egl_image`, `drm_prime_to_cuda` are registered.
* Prefer targeted tests for Python logic; do a remote build/run check for CUDA, TensorRT, Janus, or mixer graph changes.
* When changing Python mixer code, rerun pytest covering `pyplumber/auto_mixer/`, `pyplumber/auto_switcher.py`, `pyplumber/mixer.py`, `tools/mixer_tui/` before reporting done.
* For remote mixer graph debugging, keep web UI backend running with web UI registration enabled.

## CUDA graph rules
* No `hwdownload`/`hwupload`/`hwupload_cuda` workarounds between CUDA decode, preprocessing, inference, draw nodes, and NVENC. Keep pipelines zero-copy (GPU-native filters: `scale_cuda`, `pad_cuda`, CUDA/NVENC frames end-to-end).
* If CUDA filters fail, fix runtime/library selection; don't hide the problem with CPU round trips.
* Only use `hwupload`/`hwdownload` in examples explicitly about CPU/GPU or DMA-BUF interop; make that purpose clear in the graph name or comments.
