# Rust Core Refactor Plan for avplumber

## 1. Feasibility Analysis: Rust Core + C++ Nodes

### ABI Strategy
Rust and C++ do not share a stable ABI, but both interoperate cleanly with C. The architecture is:

```
┌─────────────────────────────────────────────────┐
│  Rust Core (scheduler, graph management, edges)  │
│  Exposes C ABI via libavplumber_core.so          │
├─────────────────────────────────────────────────┤
│  C ABI Layer (opaque handles, function pointers) │
├─────────────────────────────────────────────────┤
│  C++ Node .so files (compiled separately)        │
│  Export: create_node(), get_node_vtable()        │
└─────────────────────────────────────────────────┘
```

### What the C ABI must cover

| C++ Concept | C ABI Equivalent |
|---|---|
| `std::shared_ptr<Node>` | Opaque `node_handle_t` with ref-counting via `node_acquire`/`node_release` |
| `Source<T>` / `Sink<T>` | Typed opaque edge handles: `edge_handle_t`, `edge_dequeue`/`edge_enqueue`/`edge_peek`/`edge_pop` |
| `virtual void process()` | C function pointer in node vtable |
| `EdgeManager::find<T>(name)` | `edge_find(manager, name_utf8, type_tag)` |
| `Parameters` (JSON) | C strings (owned/copied) or `json_handle_t` |
| `InstanceSharedObjects<T>` | String-keyed opaque pointer registry |
| `NodeCreationInfo` | Struct of C pointers |
| `Event` | 32-bit opaque handle (fd-based) |
| `EventLoop` | Rust implementation, C callbacks registered |
| Template dispatch `DECLNODE_ATD` | Explicit factory registration per type |

### Migration path

1. **Phase 1 - C API layer**: Define the full C ABI header (`avplumber_core.h`). Implement it in Rust. C++ passes JSON strings, receives opaque handles.

2. **Phase 2 - Node adapters**: Write a `NodeAdapter` C++ wrapper class that implements the C ABI callbacks and forwards to existing node `process()` / `create()` methods. Existing node code stays largely intact.

3. **Phase 3 - Incremental port**: Nodes can be migrated to Rust one at a time. Both C++ and Rust nodes coexist by implementing the same C ABI vtable.

4. **Phase 4 - avcpp replacement**: C++ nodes still use avcpp internally (FFmpeg C++ wrapper). The C ABI layer passes `av::Packet`, `av::VideoFrame` etc. as opaque pointers with well-known layout (they're thin wrappers over `AVPacket*`/`AVFrame*`). Alternatively, define C structs that mirror the FFmpeg types.

### Concrete Feasibility Verdict

**Feasible but substantial effort** (~3-6 months for a team of 2-3). The main risks:
- **avcpp interop**: The cleanest path is to keep avcpp in the C++ node compilation and just pass opaque pointers through C ABI. Rust doesn't need to understand avcpp types - it only manages the opaque handles.
- **Template-driven node factory**: The shell-script-generated factory map becomes a `HashMap<String, extern "C" fn>` in Rust. The `DECLNODE` macros would evolve into `extern "C"` registration functions.
- **Performance overhead of C ABI**: Negligible. These calls happen ~100 times/second (once per frame per node).

---

## 2. Architecture Redesign Points

### 2.1 Remove mandatory queues for co-located nodes

**Current**: Every node pair has an `Edge<T>` (lock-free queue, min capacity ~63 items default). Even blocking nodes use `source_->get()` which blocks on queue.

**Proposed**: Introduce two edge types:
- **BufferedEdge** (current `Edge<T>`): lock-free queue with configurable capacity. Required between thread boundaries or when bursts need absorption.
- **DirectEdge**: A single-item handoff slot protected by a mutex or atomic flag. No allocation, no queue overhead. Used when producer and consumer are in the same thread (event loop) or when backpressure is naturally handled.

The scheduler decides which edge type to create based on whether the two nodes share an execution context:
- Same event loop → DirectEdge (or even zero-cost callback)
- Different event loops / blocking thread → BufferedEdge

### 2.2 Rust native async/await for event loop

**Current**: `EventLoop` is a C++ poll-based loop with `moodycamel::ConcurrentQueue<Callable>` of `std::function<void(EventLoop&)>` callbacks. Non-blocking nodes call `processWhenSignalled`, `yieldAndProcess`, `scheduleProcess` which push callbacks to the queue.

**Proposed in Rust**: Use `tokio` runtime:
```rust
// Each event loop = a tokio task on a single-threaded runtime
struct EventLoop {
    // Nodes are spawned as tasks
}

// Node trait:
trait NonBlockingNode {
    async fn run(&mut self, ctx: NodeContext) -> Result<()>;
}

// Instead of processWhenSignalled:
// node waits on an async event:
async fn run(&mut self, ctx: NodeContext) {
    loop {
        let data = ctx.source.wait_for_data().await?;
        // process...
        ctx.sink.send(processed).await?;
    }
}
```

The `ticks` concept maps to `tokio::select!` between the tick source and the data source.

Benefits:
- Natural backpressure via async channel capacity
- Zero-cost pauses (future is just not polled)
- Clean composition with `tokio::select!`, `join!`, timeouts
- Scheduling priorities via task budgets/spawn options

### 2.3 Process scaffolds to reduce node boilerplate

**Current**: Every non-blocking node repeats:
```cpp
T* dataptr = source_->peek(0);
if (dataptr == nullptr) {
    if (!ticks) processWhenSignalled(...);
    return;
}
// process data
if (sink_->put(data, true)) {
    source_->pop();
    if (!ticks) yieldAndProcess();
} else {
    if (!ticks) processWhenSignalled(...);
}
```

**Proposed in Rust**: Provide `process_fn` pattern:
```rust
impl Node for MyNode {
    fn process_fn(&mut self, input: &T) -> Option<T> {
        // Transform or filter. Return None to drop.
        Some(self.transform(input))
    }
}
```

A provided `SISOAdapter<Node>` handles all the edge polling, backpressure, tick scheduling, and event loop integration. The node only implements the transformation logic.

For stateful nodes that need flow control, provide a `FlowNode` trait with explicit `poll_input`/`push_output` methods.

### 2.4 Fix playback control architecture

**Problems with current design**:
- `IFlushAndSeek` walks the graph upward, pausing/unpausing/resetting nodes via `dynamic_cast`
- Seeking requires 4 separate phases (`_start`, `flushAndSeek`, `_finish`, `_complete`) that must be called in order
- Flushing is implemented as an atomic flag on each edge that causes data to be silently dropped
- Speed changes walk the entire chain rescaling PTS in-flight
- The whole system feels layered on top of a graph that assumes continuous unidirectional flow

**Proposed redesign**:

Make "playback timeline" a first-class scheduler concept:

```rust
struct StreamContext {
    logical_time: AtomicTimestamp,    // current "playhead" position
    speed: AtomicF32,                 // 1.0 = normal, -1.0 = reverse
    epoch: AtomicU64,                 // incremented on each seek
    paused: AtomicBool,
}
```

- **Seeking**: Increments `epoch`, resets `logical_time`. All source nodes read `epoch` and see it changed → they drop enqueued data with old epoch. No explicit queue flushing needed.
- **Speed**: Each node applies `speed` to its output timestamps. No graph traversal needed.
- **Pause**: Nodes check `paused` before processing. The scheduler suspends non-blocking node tasks.
- **Flushing**: Implicit via epoch. When epoch changes, any buffered data with old epoch is discarded on next `peek()`.

This eliminates `IFlushAndSeek` entirely. Nodes only need to be aware of `StreamContext` (which is passed as part of `NodeContext`).

---

## 3. Node Porting Strategy

### C++ Node Adapter (for backward compatibility)

```cpp
// avplumber_core_bridge.hpp
// This becomes THE header that existing C++ nodes include instead of node_common.hpp

extern "C" {
    // Opaque handles
    typedef struct AvpCore AvpCore;
    typedef struct AvpNode AvpNode;
    typedef struct AvpEdge AvpEdge;
    typedef struct AvpEventLoop AvpEventLoop;
    typedef uint32_t AvpNodeTypeTag;  // identifies data type on edge

    // Node vtable (C function pointers)
    typedef struct {
        void (*process)(AvpNode* node);
        void (*process_nonblocking)(AvpNode* node, AvpEventLoop* evl, int ticks);
        void (*start)(AvpNode* node);
        void (*stop)(AvpNode* node);
        // ... other virtual methods
    } AvpNodeVtable;

    // Edge operations (per-type, per-node instantiation)
    typedef struct {
        int (*try_peek)(AvpEdge* edge, AvpNodeTypeTag tag, void** out_data);
        int (*try_enqueue)(AvpEdge* edge, AvpNodeTypeTag tag, const void* data);
        int (*pop)(AvpEdge* edge, AvpNodeTypeTag tag);
        int (*occupied)(AvpEdge* edge);
        // events
        int (*producer_event_fd)(AvpEdge* edge);
        int (*consumer_event_fd)(AvpEdge* edge);
    } AvpEdgeVtable;

    // Factory registration
    typedef AvpNode* (*AvpNodeFactoryFn)(AvpCore* core, const char* json_params);

    void avp_register_node_factory(AvpCore* core, const char* type_name, AvpNodeFactoryFn fn);
    AvpEdge* avp_edge_find_or_create(AvpCore* core, const char* name, AvpNodeTypeTag tag, size_t capacity);
    // ... etc
}
```

**Existing node example, adapted**:
```cpp
class Firewall : public AvpNodeAdapter {  // wraps C ABI
public:
    static AvpNode* create(AvpCore* core, const char* json) {
        auto params = parse_json(json);
        auto node = std::make_shared<Firewall>(/* ... */);
        return AvpNodeAdapter::register_node(core, node);
    }

    void process() override {
        T data = source_->get();
        if (!data.pts().isValid()) return;
        sink_->put(data);
    }
};

// Registration (replaces DECLNODE macro):
__attribute__((constructor))
static void register_firewall() {
    avp_register_node_factory(nullptr, "firewall", Firewall::create);
}
```

### Rust Node Example
```rust
#[avp_node(type = "my_filter", input = "av::Packet", output = "av::Packet")]
struct MyFilter {
    threshold: f64,
}

impl SisoNode for MyFilter {
    fn process(&mut self, input: Packet) -> Option<Packet> {
        if input.pts.seconds() > self.threshold {
            Some(input)
        } else {
            None  // drop
        }
    }
}
```

---

## 4. Implementation Phases

### Phase 0: C ABI Specification
- Design `avplumber_core.h` covering: graph management, edge operations, node lifecycle, event loop, JSON parameter passing, instance-shared objects
- Prototype with a simple 2-node graph (demux → null_sink)

### Phase 1: Rust Core MVP
- Graph data structure (DAG of nodes + edges)
- TCP control protocol server (reuse existing protocol)
- Node registry (factory map: string → function pointer)
- Edge creation with type-tagging
- Basic event loop (single threaded, poll-based) with tick source support
- `StreamContext` (logical time, epoch, speed, pause state)

### Phase 2: Node Adapter Toolkit
- `AvpNodeAdapter` C++ class wrapping C ABI
- `AvpSource`/`AvpSink` wrappers adapting `Source<T>`/`Sink<T>` to C ABI edge operations
- JSON parameter bridge
- Port one simple C++ node (`firewall`) as proof of concept

### Phase 3: Process Scaffolds (Rust side)
- `SisoNode` trait + adapter
- `MultiInputNode` trait + adapter
- `SourceNode` / `SinkNode` traits
- Process macro for auto-registration

### Phase 4: New Playback Control
- Implement epoch-based timeline in `StreamContext`
- Integrate speed, pause, seek as scheduler-level operations
- Remove `IFlushAndSeek` from node responsibilities
- Port `realtime`, `speed`, `pause` nodes as thin shims that read/write `StreamContext`

### Phase 5: DirectEdges & Zero-Queue Optimization
- Implement `DirectEdge` for same-thread nodes
- Scheduler detects co-located nodes and creates DirectEdge instead of BufferedEdge
- Measure latency improvement

### Phase 6: Async/Await Event Loop
- Replace poll-based event loop with tokio runtime
- Non-blocking nodes become async tasks
- Backpressure via bounded async channels
- Benchmark against current C++ EventLoop

### Phase 7: Incremental Node Porting
- Port nodes from C++ to Rust, starting with simple ones (split, one_to_many, firewall, null_sink)
- CUDA/GL nodes stay C++ long-term (no Rust CUDA ecosystem)
- Python bindings call Rust core via C ABI (pyo3 → C FFI)

### Phase 8: Deprecate Old Core
- Once all functionality is available via Rust core, remove C++ graph management code
- Existing `.avplumber` scripts and control protocol remain compatible

---

## 5. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| C ABI overhead on per-frame calls | Measure early. If significant (>1%), batch operations or use shared memory for hot paths |
| CUDA/GL/TensorRT nodes can't be ported to Rust | Accept these stay C++. The adapter layer handles them fine. |
| `avcpp` types crossing FFI boundary | Pass opaque `av::Packet*` / `av::VideoFrame*` pointers. Rust doesn't dereference them. |
| Control protocol parsing in Rust | Trivial. The protocol is simple line-based text with JSON payloads. Use `serde_json`. |
| Instance-shared objects thread safety | Same pattern: opaque handle + mutex in Rust, accessed via C ABI get/set |
| Team/linking logic (pause teams, speed teams) | Moves to Rust core; C++ nodes call `avp_team_get_status()` instead of shared_ptr to team object |
| Build system complexity | C++ nodes compile to `.so` with C bridge. Rust core is separate crate. Link at runtime via dlopen or compile-time linking. |

---

## 6. Open Questions

1. **Edge type representation**: Currently 5 known types + extensible. Should the type tag be runtime (string/enum) or compile-time (template specialization)? For C ABI, runtime tagging (`uint32_t type_id`) is simpler. This also addresses the article's point about templates being "premature optimization."

2. **Zero-copy between nodes**: If Rust and C++ nodes are in the same process, can we pass pointers to avcpp-wrapped FFmpeg objects directly? Yes - both sides use the same heap. The C ABI just passes `void*`.

3. **Python bindings**: `pyplumber/` currently creates avplumber instances and nodes via pybind11. With the Rust core, pyplumber would call the C ABI functions. PyO3 could wrap the Rust types directly for a cleaner Rust→Python path, but the C ABI approach works for both.

4. **Rust metaprogramming for boilerplate reduction**: The `DECLNODE` macros and `generate_node_list` script could be replaced by Rust proc macros:
   ```rust
   #[avp_node(type = "my_node")]
   struct MyNode { ... }
   ```
   This would auto-generate the factory registration, vtable, and C ABI export functions. Significant improvement over the current shell script + X-macro approach.
