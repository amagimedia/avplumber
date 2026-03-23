# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

**avplumber** is a graph-based real-time multimedia processing framework built on FFmpeg. It processes video/audio via directed graphs of nodes connected by queues. The binary accepts a control socket (default port 20200) and an optional script file of commands to configure and run media pipelines.

## Build Commands

```bash
# Debug build (default)
make -j$(nproc)

# Release build
make BUILD_TYPE=Release -j$(nproc)

# Build with hardware acceleration options
make HAVE_CUDA=1 HAVE_VAAPI=1 HAVE_DRM=1 -j$(nproc)

# Build static library (for embedding)
make static_library -j$(nproc)

# Clean
make clean
make clean_deps  # also clean submodule dependencies
```

### Remote build & test (AWS g4 instance)

```bash
# SSH to remote
ssh -i /home/jp/work-misc-stuff/awsdev.pem fedora@172.17.36.132

# Rsync files to remote
rsync -avz -e "ssh -i /home/jp/work-misc-stuff/awsdev.pem" <files> fedora@172.17.36.132:/home/fedora/avplumber/

# Build on remote (always use this exact command)
make -j8 \
  HAVE_CUDA=1 \
  HAVE_TENSORRT=1 \
  HAVE_NVCC=1 \
  NVCC=/usr/local/cuda-13.0/bin/nvcc \
  TENSORRT_ROOT=/opt/tensorrt \
  PKG_CONFIG_PATH=/apps/ffmpeg/lib/pkgconfig \
  CXXFLAGS+=' -I/usr/local/cuda-13.0/include -I/usr/local/cuda-13.0/targets/x86_64-linux/include' \
  LFLAGS+=' -L/apps/ffmpeg/lib -Wl,-rpath,/apps/ffmpeg/lib -L/usr/local/cuda-13.0/targets/x86_64-linux/lib -Wl,-rpath,/usr/local/cuda-13.0/targets/x86_64-linux/lib'
```

Remote avplumber path: `/home/fedora/avplumber`

### Build flags
| Flag | Effect |
|------|--------|
| `HAVE_CUDA=1` | Enable CUDA support (dynamic linking) |
| `HAVE_NVCC=1` | Compile CUDA PTX kernels |
| `HAVE_TENSORRT=1` | Enable TensorRT inference (YOLO, reframer, vert_infer) |
| `HAVE_VAAPI=1` | VAAPI hardware accel (also enables GL) |
| `HAVE_DRM=1` | DMA-BUF IPC and DRM paths |
| `HAVE_GL=1` | OpenGL/EGL support |
| `HAVE_JACK=1` | JACK audio output |
| `HAVE_SCTE35=1` | SCTE-35 ad insertion |
| `EMBED_IN=obs` | Build OBS plugin nodes |

**Code generation**: `make` runs `generate_node_list` which produces `graph_factory.generated.cpp` — the node factory registry. After adding a new node file to `src/nodes/`, re-run make to regenerate this file.

## Running

```bash
./avplumber -p 20200 -s examples/remux_analyze_audio.avplumber
```

Control protocol is line-based over TCP. Commands can be piped via netcat:
```bash
echo 'node.add {"name":"input1","type":"input","url":"rtmp://...","dst":"q1"}' | nc localhost 20200
```

## No Test Suite

There is no automated test suite. Testing is done manually via `.avplumber` script files and Docker Compose examples in `examples/compose/`.

## Architecture

### Core Abstractions (`src/graph_core.hpp`)

- **Node**: Base class for all processing units. Each blocking node runs in its own thread; non-blocking nodes share an event loop.
- **Edge**: A typed queue between nodes. Carries one of: `av::Packet`, `av::VideoFrame`, `av::AudioSamples`, or `EglImageFrame`.
- **Source/Sink**: Typed reader/writer ends of an Edge, owned by nodes.

### Node Templates (`src/graph_base.hpp`)

- `NodeSISO<In, Out>`: Single-input, single-output
- `NodeMultiInput<...>`: Multiple input queues
- `NodeMultiOutputs<...>`: Multiple output queues
- `DECLNODE_ATD_RAW`: Macro for auto-type-detecting nodes (works with any data type)

### Node Management (`src/graph_mgmt.hpp/.cpp`)

- **NodeWrapper**: Wraps a node with a thread and lifecycle management
- **NodeGroup**: Named group of nodes that can be started/stopped together
- **NodeManager**: Global registry of all nodes; handles add/delete/start/stop

### Shared State (`src/instance_shared.hpp/.cpp`, `src/instance.hpp`)

Global objects shared across nodes: hardware acceleration contexts, team objects, statistical collectors. Accessed via `InstanceData`.

### Teams (synchronization)

Team objects coordinate groups of nodes:
- `RealTimeTeam` — wallclock rate-limiting with inter-stream sync
- `PauseControlTeam` — pause/resume across multiple streams
- `SpeedControlTeam` — playback speed
- `InputSeekTeam` — coordinated seeking

### Event Loop (`src/EventLoop.hpp/.cpp`)

Non-blocking nodes use an event loop with `async`, `sleep`, and `schedule` operations — avoids spawning a thread per node for lightweight tasks.

### Control Protocol

Line-based MVCP-like protocol. Key command categories:
- `node.add`, `node.delete`, `node.start`, `node.stop`, `node.param.set/get`
- `pause`, `resume`, `seek`, `speed.set/get`
- `team.link`, `team.unlink`
- `event.on.node.finished`, `event.wait`
- `queue.plan_capacity`, `queues.stats`

## Writing New Nodes

See `doc/developing_nodes.md` for a full tutorial. Key points:

1. Create `.cpp` in `src/nodes/` — the `generate_node_list` script will auto-register it.
2. Subclass one of the node templates or `Node` directly.
3. Use `DECLNODE` / `DECLNODE_ATD_RAW` macros to register the node type name. **Never put a semicolon after `DECLNODE(...)`** — it will silently break node registration.
4. Implement `process()` (blocking nodes) or register callbacks with the event loop (non-blocking nodes).
5. Use `params` (nlohmann/json) for configuration; node params are set at creation time and can be updated via `node.param.set`.
6. Implement `IParameterizable` if the node accepts dynamic param updates.

## Dependencies

Submodules in `deps/`:
- **avcpp**: C++ wrapper around FFmpeg API (primary FFmpeg interface)
- **json** (nlohmann): JSON parsing for control protocol and node params
- **readerwriterqueue** / **concurrentqueue**: Lock-free queues used by edges
- **cpr**: HTTP client for stats reporting
- **libklvanc** / **libklscte35**: SCTE-35 ad insertion (autotools-built)
- **cuda_loader**: Dynamic CUDA driver loader (no CUDA SDK required at compile time)

System libraries required: FFmpeg (libav*), Boost (thread, system), libcurl, OpenSSL.
