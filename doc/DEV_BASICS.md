# avplumber development basics for LLMs

(if you're a human, see the full [README.md](../README.md))

avplumber is a graph-based real-time processing framework. Graphs reconfigure on the fly via a text TCP API. Nodes wrap FFmpeg libavcodec/libavformat/libavfilter. They handle timestamping, A/V sync, fallback slates. The framework handles queues, parallel processing.

Full reference: [README.md](../README.md) (don't waste time reading it unless you want to dig deeper) | Node development: [developing_nodes.md](developing_nodes.md)

---

## Quick start

```bash
# Docker
docker build -t avplumber .
docker run -p 20200:20200 avplumber -p 20200

# Ubuntu (plain)
apt install git gcc pkg-config make cmake \
  libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev \
  libavutil-dev libswresample-dev libcurl4-openssl-dev \
  libboost-thread-dev libboost-system-dev libssl-dev
make -j`nproc`
./avplumber        # then: nc localhost 20200
```

Clone with `--recursive`. Paste scripts from `examples/` into nc.

---

## Build flags

`make -j$(nproc) [FLAGS]`

| Flag | Effect |
|---|---|
| `BUILD_TYPE=Release` | Optimization flags (default: `Debug`, enables jittergen/delaygen) |
| `HAVE_CUDA=1` | CUDA nodes; dynlink loader – no compile-time CUDA needed |
| `HAVE_GL=1` | OpenGL/EGL; required by `drm_prime_to_cuda`, `cuda_to_egl_image` |
| `HAVE_VAAPI=1` | VAAPI (implies GL). Links `-lva -lGL -lEGL -lGLESv2` |
| `HAVE_DRM=1` | DMA-BUF IPC source and DRM paths. Requires `libdrm-dev` |
| `HAVE_TENSORRT=1` | TensorRT inference nodes. Optionally set `TENSORRT_ROOT=` |
| `HAVE_JACK=1` | `jack_sink`. Links `-ljack` |
| `HAVE_NVCC=1` | Compile CUDA PTX for CUDA processing nodes. Requires `nvcc` |
| `HAVE_SCTE35=1` | `scte35_parse` node |
| `EMBED_IN=obs` | OBS source plugin build |

Feature gates (node only built when all conditions met):
- `cuda_to_egl_image`: `HAVE_CUDA=1 HAVE_GL=1 HAVE_NVCC=1`
- `drm_prime_to_cuda`: `HAVE_CUDA=1 HAVE_GL=1 HAVE_DRM=1`
- `cuda_infer_yolo` / `cuda_infer_rtdetr`: `HAVE_CUDA=1 HAVE_TENSORRT=1 HAVE_NVCC=1`
- `HAVE_GL` auto-enabled by `HAVE_VAAPI=1`

Static library: `make static_library` → `libavplumber.a`. Public API: `src/avplumber.hpp`.

---

## Graph model

A directed acyclic graph of nodes connected by typed queues (edges). Edge type is inferred automatically.

**Data types on edges:**
- `av::Packet` – encoded packet
- `av::VideoFrame` – raw video frame
- `av::AudioSamples` – raw audio (typically 1024-sample blocks)
- `EglImageFrame` – GPU RGBA image via `EGLImageKHR`

When a node is generic (templated), type is inferred from connected edges. If ambiguous, use explicit syntax: `split<av::VideoFrame>`.

### Topology constraints

| Node | Requires upstream |
|---|---|
| `demux` | `input` / `input_rec` |
| `output` | `mux` |
| `enc_video` | video format source (`dec_video`, `assume_video_format`, `rescale_video`, `filter_video`) + FPS source |
| `enc_audio` / `sentinel_audio` | audio metadata source |
| `bsf`, `enc_*`, `filter_*`, `sentinel_*`, `extract_timestamps`, `resample_audio` | time base source |
| `mux` | `enc_video`, `enc_audio`, `bsf`, or `packet_relay` |

---

## Control protocol

TCP text protocol. Connect with `nc localhost 20200`.

On connect: `100 VTR READY`

Responses: `200 OK` (empty), `201 OK` (data follows, blank line ends), `400` (unknown cmd), `500` (error), `BYE` (on `bye`).

---

## Node JSON object

```jsonc
{
  "name": "unique_id",          // optional; default: "type@addr"
  "type": "node_type",          // required
  "group": "group_name",        // optional grouping for restart
  "auto_restart": "off",        // off | on | group | panic
  "src": "edge_name",           // string or list of strings
  "dst": "edge_name",           // string or list of strings
  "optional": false,            // swallow creation errors when true
  // ...node-specific params...
}
```

Value types: `"string"`, `"30000/1001"` (rational), `["a","b"]` (list), `{"k":"v"}` (dict), `true`/`false`, `31337` (int), `1337.42` (float).

### Non-blocking nodes

Add to node JSON:
- `"event_loop": "name"` – named event loop thread (default: `"default"`)
- `"tick_source": "name"` – wakes node on external clock ticks (e.g. `"obs"`); mutually exclusive with `event_loop`

---

## Instance-shared objects

Shared state between nodes; named objects stored per-instance. Name starting with `@` is process-global (useful when avplumber runs as a library with multiple instances).

**In C++ (nodes):**
```cpp
// Get (creates with default constructor if missing):
auto obj = InstanceSharedObjects<MyType>::get(nci.instance, name);

// Create with arguments:
InstanceSharedObjects<MyType>::emplace(nci.instance, name,
    ISOs::PolicyIfExists::Ignore, ...constructor_args...);
```

`PolicyIfExists`: `Overwrite`, `Ignore` (preferred for lazy init), `Throw`.

Implement as `struct MyType : InstanceShared<MyType> { ... }`. For circular node↔shared-object references use `std::weak_ptr` or declare an interface in `src/graph_interfaces.hpp`.

---

## Developing nodes

Full guide: [developing_nodes.md](developing_nodes.md). Summary:

- Node sources in `src/nodes/`; `make` auto-scans via `generate_node_list` – no index file to update.
- Every `.cpp` starts with `#include "node_common.hpp"`.
- End each node class with a `DECLNODE` macro (no trailing semicolon).

### DECLNODE macros

| Macro | Use |
|---|---|
| `DECLNODE(type, Class)` | Fixed queue types |
| `DECLNODE_ATD(type, Tpl)` | Auto-detect type from all edge types |
| `DECLNODE_ATD_RAW(type, Tpl)` | Auto-detect from raw frame/samples only |
| `DECLNODE_ATD_TYPES(type, Tpl, T1, T2, ...)` | Auto-detect from explicit type list |
| `DECLNODE_ALIAS(type, Class)` | Alias for `DECLNODE` node |
| `DECLNODE_ATD_ALIAS(type, Tpl)` | Alias for `DECLNODE_ATD` node |

### Helper base classes

| Base | I/O pattern |
|---|---|
| `NodeSingleInput<T>` | `source_->get()` / `peek(0)` + `pop()` |
| `NodeSingleOutput<T>` | `sink_->put(data)` |
| `NodeSISO<In,Out>` | combines both |
| `NodeMultiInput<T>` | `findSourceWithData()` → index into `source_edges_[i]` |
| `NodeMultiOutputs<T>` | `sink_edges_[i]->enqueue(data)` / `try_enqueue` |

Call base constructors or `using Base::Base;` in derived classes.

### Blocking vs non-blocking

**Blocking** (`process()` override): preferred for I/O, encoding/decoding, expensive compute.

**Non-blocking** (derive from `NonBlockingNode`, override `processNonBlocking(EventLoop&, bool ticks)`):
- Without `tick_source`: called once at start; schedule follow-up with:
  - `processWhenSignalled(edge->producedEvent())` – when input available
  - `processWhenSignalled(edge->consumedEvent())` – when output has space
  - `sleepAndProcess(ms)`, `scheduleProcess(ts)`, `yieldAndProcess()`
- With `tick_source`: called each tick (`ticks=true`); loop over all available input to avoid accumulation.

Stateful non-blocking processing: use a blocking node, or capture state via `std::weak_ptr` in `evl.asyncWaitAndExecute` closure (never capture raw `this`).

### `create` static method

```cpp
static std::shared_ptr<MyNode> create(NodeCreationInfo &nci) {
    return NodeSISO<In, Out>::template createCommon<MyNode>(
        nci.edges, nci.params /*, extra_ctor_args... */);
}
```

---

## Tips

### Change input on the fly
```
node.interrupt input
node.param.set input url "rtmp://new.stream/url"
```
Issue both commands within ~1 second. Alternatively (when input is running normally):
```
node.param.set input url "rtmp://new.stream/url"
node.auto_restart input
```

### Watch queue fill in real time
```bash
watch -n0.1 "echo 'queues.stats' | nc localhost 20200"
# or, if nc exits too early:
watch -n0.1 "echo -e 'queues.stats\nbye\n\n' | nc localhost 20200"
# compact output:
while true; do
  echo -e 'queues.stats\nbye\n\n' | nc localhost 20200 \
    | sed -E 's/#{16}/$/g; s/\.{16}/,/g'
  sleep 0.1
done
```

Find non-empty queues in log: regex `[1-9]0?/[0-9]{1,3},`

### Dump config from log
```bash
sed -e 's/^.\+\[control\] Executing: \(.\+\)$/\1/; t; d' < log
```
