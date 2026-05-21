
## When thinking about something non-obvious
* Do not assume that user is right, especially when they say *maybe*, *possibly* etc., it indicates uncertainity, and it's your job to point out better solution - if there is one.
* When the user wants to change the architecture and is asking for advice, don't assume that their (or your) more recent idea is always better. Be objective.


## When writing documentation and comments
* Do not write obvious things...
* ... but prefer unambiguity over brevity.
* See [README.md](README.md) for example style.


## Project shape
* avplumber is a graph-based real-time multimedia framework built around nodes connected by typed queues/edges.
* The main binary is controlled through a line-based TCP protocol. `.avplumber` scripts are the usual way to create, connect, and start graph nodes.
* Core media edges carry FFmpeg packets, video frames, audio samples, metadata frames, or platform-specific image handles depending on the node chain.


## When using local agent notes
* Treat this `AGENTS.md` file as the durable instruction source for this repo.
* If another local agent note contains useful workflow guidance, copy only the reusable rule into this file and generalize any machine-specific commands, paths, hosts, ports, URLs, or credentials first.
* Do not preserve private environment details merely because they appeared in another local note.


## When handling private environment details
* Do not commit fixed IP addresses, private hostnames, private HTTP(S) URLs, S3 bucket names, credentials, tokens, API keys, or test-instance-specific absolute paths.
* Loopback literals such as `127.0.0.1` are acceptable when the code must force local IPv4 behavior; do not replace them with environment-specific hosts.
* Prefer CLI arguments, environment variables, ignored local config, or placeholder examples such as `<host>`, `<path>`, and `<bucket>`.
* Public project URLs and package registry URLs are fine when they are not credentials or private infrastructure.


## When reporting files
* When referring to a concrete file in a response, include the full absolute path with the filename.
* If multiple files are involved, list each file separately.


## Build and test workflow
* Do not build locally on ARM when the target requires the x86/CUDA runtime. Use a user-provided remote x86/GPU build environment.
* Do not commit remote hostnames, IPs, SSH key paths, checkout paths, CUDA paths, TensorRT paths, or other instance-specific build details. Keep those in local shell config, ignored notes, or user-provided commands.
* For CUDA/neural builds, preserve the repo's intended feature flags: CUDA, neural_net common, neural_net specific, NVCC/PTX, and TensorRT where required. Keep optional FRUC disabled unless explicitly testing frame interpolation.
* Python module builds for pyplumber demos must use the same neural/CUDA/TensorRT feature set as the binary build. Do not use a plain Python-module build if it drops required node factories.
* Builds that exercise Electron DMA-BUF overlay must also include the DRM/GL feature flags so `ipc_dmabuf_source`, `drm_prime_to_egl_image`, and `drm_prime_to_cuda` are registered in the generated node factory.
* `make` regenerates the node factory registry. After adding a new node under `src/nodes/`, rebuild so the generated registry includes it.
* Prefer targeted tests for Python logic when available, plus a remote build/run check for CUDA, TensorRT, Janus, or mixer graph changes.
* When changing Python mixer code, rerun the relevant pytest coverage before reporting done. At minimum, cover the affected files under `pyplumber/auto_mixer/`, `pyplumber/auto_switcher.py`, `pyplumber/mixer.py`, and `tools/mixer_tui/`.
* When running a remote mixer for graph debugging, keep the web UI backend running and start the mixer with web UI registration enabled so the graph is visible during testing.


## Python/VAD and graph-management transplants
* Keep the current graph-management framework when transplanting Python/VAD files.
* Do not copy `src/graph_mgmt.cpp` or `src/graph_mgmt.hpp` from older feature branches unless the user explicitly asks for that framework change.


## CUDA graph rules
* For CUDA/neural example graphs, do not introduce `hwdownload`, `hwupload`, or `hwupload_cuda` as a workaround between CUDA decode, CUDA preprocessing, inference, CUDA draw nodes, and NVENC.
* Keep those pipelines zero-copy with GPU-native filters such as `scale_cuda` and `pad_cuda`, CUDA nodes, and CUDA/NVENC frames all the way through.
* If CUDA filters fail while the build appears to include them, fix the runtime/library selection first. Do not hide the problem with CPU round trips.
* On GPU hosts that install the custom FFmpeg build under `/usr/local`, verify that `_avplumber`/`avplumber` resolves `libavcodec`, `libavfilter`, `libavformat`, `libavutil`, `libswscale`, and related FFmpeg libraries from `/usr/local/lib`, not the distro FFmpeg library directory.
* When diagnosing missing filters such as `scale_cuda`, `convert_cuda`, `overlay_many_cuda`, or `pad_cuda`, compare `/usr/local/bin/ffmpeg -filters` with the process runtime environment, then fix `LD_LIBRARY_PATH`, rpath, or service environment so the app loads the same FFmpeg build.
* Only use `hwupload`/`hwdownload` in examples whose explicit purpose is CPU/GPU or DMA-BUF interop, and make that purpose clear in the graph name or comments.


## Code reuse and framework changes
* Do not copy-paste logic between nodes. If two or more nodes need the same logic, extract a shared base class or utility in the relevant common module.
* Do not modify framework source files such as graph management, control protocol, main, or sentinel code unless explicitly asked.
* Framework-wide behavior changes should be deliberate and called out as such; avoid incidental framework edits while working on node or mixer features.


## Writing new nodes
* Put new node implementations under `src/nodes/`; the generated node list handles registration during build.
* Subclass the appropriate node template or `Node` directly.
* Use the repo registration macros correctly. Do not put a semicolon after `DECLNODE(...)`.
* Use JSON params for configuration and implement dynamic parameter handling only when the node actually needs runtime updates.
