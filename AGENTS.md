# Personality

## Non-obvious questions
* Don't assume the user is right, especially with *maybe*/*possibly*; point out a better solution if one exists.
* When advising on architecture changes, don't assume a more recent idea is better. Be objective.


# General guidelines

## Documentation & comments
* Don't write obvious things, but prefer unambiguity over brevity. See [README.md](README.md) for style.

## Project shape
* avplumber is a graph-based real-time multimedia framework: nodes connected by typed queues/edges.
* Controlled via line-based TCP protocol; `.avplumber` scripts create, connect, and start nodes. Python bindings in `pyplumber/`.
* Media edges carry FFmpeg packets, video frames, audio samples, metadata, or platform-specific image handles.
* See [DEV_BASICS.md](doc/DEV_BASICS.md), scan the `doc/` directory for more docs.

## Code & logic style
* No copy-paste between nodes or within a file; extract shared base classes, utilities, or helper functions/lambdas.
* C++/Python project: prefer C++ idioms (RAII, exceptions) over C patterns, but don't over-apply OOP.

## Framework changes
* Don't modify framework source (graph management, control protocol, main, sentinel) unless explicitly asked or the change is necessary, generally useful, and future-proof.
* Call out framework-wide behavior changes explicitly; don't make incidental edits during node/mixer work.
* In particular, do not modify the framework as a workaround that could be done more properly by introducing a new node, new 
interface, new shared object etc., or improving existing ones.

## Writing new nodes
* Put implementations under `src/nodes/`; `make` regenerates the node factory registry — rebuild after adding a node.
* Subclass the appropriate template or `Node` directly. No semicolon after `DECLNODE(...)`.
* Use JSON params; add dynamic handling only when runtime updates are needed.
* See [developing_nodes.md](doc/developing_nodes.md) for structure and patterns.

## Before implementing
* State assumptions explicitly; ask if uncertain.
* Present multiple interpretations rather than picking silently.
* Say so if a simpler approach exists. Push back when warranted.
* If something is unclear, stop, name what's confusing, and ask.


# Agentic work guidelines

## Agent notes
* This `AGENTS.md` is the durable instruction source. Copy only reusable rules from other local notes; generalize machine-specific details first. Don't preserve private environment details.

## Private environment details
* Don't commit IPs, private hostnames, URLs, S3 buckets, credentials, tokens, or test-instance paths.
* `127.0.0.1` is fine when forcing local IPv4. Use `<host>`, `<path>`, `<bucket>` placeholders elsewhere.
* Public project/registry URLs are fine.

## Reporting files
* Always include full absolute paths when referring to files; list each file separately.

## Build & test workflow
* Don't build locally for x86/CUDA targets unless the host has the required hardware. Use a user-provided remote environment.
* Don't commit instance-specific build details (hostnames, IPs, SSH keys, CUDA/TensorRT paths).
* CUDA/neural builds: preserve feature flags `CUDA`, `neural_net common`, `neural_net specific`, `NVCC/PTX`, `TensorRT`. Keep FRUC disabled unless testing frame interpolation.
* pyplumber builds must use the same neural/CUDA/TensorRT feature set as the binary build.
* DMA-BUF overlay builds need DRM/GL flags so `ipc_dmabuf_source`, `drm_prime_to_egl_image`, `drm_prime_to_cuda` are registered.
* Prefer targeted tests for Python logic; do a remote build/run check for CUDA, TensorRT, Janus, or mixer graph changes.
* When changing Python mixer code, rerun pytest covering `pyplumber/auto_mixer/`, `pyplumber/auto_switcher.py`, `pyplumber/mixer.py`, `tools/mixer_tui/` before reporting done.
* For remote mixer graph debugging, keep web UI backend running with web UI registration enabled.

## Build & test commands
Prepare `./build.sh` and `./run.sh` on the remote host if they don't exist. Adapt these examples; don't use verbatim.

Example `build.sh` (auto mixer; remove `python_module` for the default binary target):
```
make -j8 NEURAL_NET_COMMON=1 NEURAL_NET_SPECIFIC=1 HAVE_DRM=1 HAVE_GL=1 HAVE_CUDA=1 HAVE_NVOF_FRUC=1 HAVE_NVCC=1 NVCC=/usr/local/cuda-13.0/bin/nvcc TENSORRT_ROOT=/opt/tensorrt PKG_CONFIG_PATH=/usr/local/lib/pkgconfig CXXFLAGS+=' -I/usr/local/include -I/usr/local/cuda-13.0/include -I/usr/local/cuda-13.0/targets/x86_64-linux/include' LFLAGS+=' -L/usr/local/lib -Wl,-rpath,/usr/local/lib -L/usr/local/cuda-13.0/targets/x86_64-linux/lib -Wl,-rpath,/usr/local/cuda-13.0/targets/x86_64-linux/lib' python_module
```

Example `run.sh` for pyplumber:
```
LD_LIBRARY_PATH=/usr/local/lib venv/bin/python3 pyplumber/examples/auto_mixer.py --webui-api http://localhost:22222 --input-start-ts 660000 --remote-control-port 22422 --inputs /data/test-content/*.ts --output rtmp://... --debug-mouth-roi-bboxes
```

Example `run.sh` for avplumber:
```
LD_LIBRARY_PATH=/usr/local/lib ./avplumber --webui-api http://localhost:22222 --port 22422 -s examples/sync_mixer.avplumber
```


# Domain-specific rules

## Python/VAD transplants
* Keep the current graph-management framework. Don't copy `src/graph_mgmt.cpp`/`.hpp` from older branches unless explicitly asked.

## CUDA graph rules
* No `hwdownload`/`hwupload`/`hwupload_cuda` workarounds between CUDA decode, preprocessing, inference, draw nodes, and NVENC. Keep pipelines zero-copy (GPU-native filters: `scale_cuda`, `pad_cuda`, CUDA/NVENC frames end-to-end).
* If CUDA filters fail, fix runtime/library selection; don't hide the problem with CPU round trips.
* On hosts with custom FFmpeg under `/usr/local`, ensure `avplumber` loads FFmpeg libs from `/usr/local/lib`, not the distro path. Compare `/usr/local/bin/ffmpeg -filters` with the process environment; fix `LD_LIBRARY_PATH`, rpath, or service env.
* Only use `hwupload`/`hwdownload` in examples explicitly about CPU/GPU or DMA-BUF interop; make that purpose clear in the graph name or comments.
