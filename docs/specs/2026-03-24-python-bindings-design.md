# Python Bindings Design Spec

## Goal

Add embedded, in-process Python bindings for `avplumber` so a Python application can:

- construct and run an embedded `AVPlumber` instance
- build graphs in pure Python with full typing and IDE-friendly node APIs
- mix Python-backed nodes with existing C++ nodes in the same graph
- subscribe to frames, packets, and samples emitted by C++ nodes
- inspect low-level frame and packet properties from Python through read-only wrapper objects
- create Python-originated synthetic media objects, especially metadata-bearing sidecar frames, and pass them downstream into C++ nodes
- monitor graph, node, queue, and pipeline lifecycle state from Python

The recommended architecture is a hybrid design:

- Python exposes a native, typed SDK
- C++ remains the owner of media transport, graph execution, and heavy processing
- internal graph construction may compile to existing avplumber command semantics where practical
- new C++ binding surfaces are added where the current control model is insufficient, especially for typed schemas, subscriptions, synthetic frame emission, and state snapshots

## Scope

### In scope

- embedded, in-process Python only
- Python package API plus a documented underlying C++/pybind surface for maintainers
- typed graph authoring in Python
- mixed Python and C++ node graphs
- subscriptions and callbacks on graph edges
- read-only media bindings from C++ to Python
- Python-originated synthetic frames for metadata sidecar workflows
- runtime lifecycle and health monitoring from Python

### Out of scope

- remote control of an external avplumber process in v1
- full general-purpose two-way FFmpeg binding for arbitrary `AVFrame` and `AVPacket` mutation from Python
- Python-generated media objects that are automatically valid for every downstream media-processing node
- distributed execution, multi-process orchestration, or RPC transport in v1

## Why This Fits avplumber

`avplumber` already provides:

- a graph runtime with typed edges carrying `av::Packet`, `av::VideoFrame`, and `av::AudioSamples`
- metadata-heavy intelligence nodes such as `cuda_infer_yolo`, `basketball_analysis`, `draw_bbox`, and `draw_text`
- a natural sidecar metadata merge pattern via `join_metadata`
- internal edge callbacks through `addWiretapCallback`
- an embeddable `AVPlumber` library API

This makes avplumber a strong fit for a split architecture:

- C++ handles low-level media movement, codec work, GPU work, and deterministic graph behavior
- Python handles high-level intelligence, math, application logic, model orchestration, analytics, and custom domain-specific processing

## Recommended Approach: Hybrid Embedded Runtime

### Layers

1. Native Python SDK

- Python users interact with typed `Pipeline`, `Runtime`, node classes, subscriptions, and state snapshots.
- Python users do not author raw command strings in normal usage.

2. C++ binding surface

- C++ exposes runtime lifecycle, node schemas, subscriptions, read-only frame views, and synthetic frame builders.
- This is the canonical maintainers' API layer beneath the Python package.

3. Internal command compatibility

- Where practical, graph authoring compiles to existing avplumber node JSON and command semantics.
- Existing `.avplumber` scripts remain conceptually compatible.
- Python does not depend on remote transport or TCP control mode.

### Alternatives considered

#### Command-layer wrapper

Python would mostly wrap the existing command interface and add a thin embedded shell around it.

Pros:

- lowest implementation risk
- fastest path to basic graph creation

Cons:

- weak typing
- poor IDE experience
- state and subscriptions feel bolted on
- not a strong long-term app SDK

#### Native embedded runtime API

Python would talk only to a new, fully native C++ runtime API with no visible command model.

Pros:

- cleanest final API
- strongest long-term architecture

Cons:

- highest up-front implementation cost
- duplicates concepts already present in the repo

#### Hybrid

Python gets a clean native API, while implementation reuses existing node JSON semantics and graph construction model wherever useful.

Pros:

- best balance of ergonomics and leverage
- allows incremental implementation
- preserves compatibility with current architecture

Cons:

- requires clear separation between public SDK and internal command-derived representation

## High-Level Architecture

### Major components

- `Runtime`
- `Pipeline`
- typed node schema registry
- typed Python node classes
- subscription and event delivery layer
- read-only media view layer
- synthetic media builder layer
- pipeline state snapshot layer
- Python-backed node runtime support

### Ownership model

- C++ owns graph execution, queueing, node lifecycle, and media object lifetime.
- Python receives safe wrapper views or snapshots.
- Python may emit synthetic media objects through a controlled builder API.
- Python does not receive raw ownership of FFmpeg internals in v1.

## User-Facing Python API

### Runtime

```python
import avplumber as avp

rt = avp.Runtime(name="basketball-app")
pipe = avp.Pipeline(name="game-1")

pipe.add(avp.nodes.Input(name="input", url="nba.mp4", dst="in_v", group="in"))
pipe.add(avp.nodes.Demux(src="in_v", routing={"?v:0": "v_pkt"}, group="in"))
pipe.add(avp.nodes.DecVideo(name="Video_Dec", src="v_pkt", dst="v_dec_cuda", pixel_format="?cuda", hwaccel="@gpu"))
pipe.add(avp.nodes.CudaInferYolo(name="Yolo_Infer", src="v_pre_yolo", dst="v_post_yolo", hwaccel="@gpu", models=[...]))
pipe.add(avp.nodes.PyMetadataTransform(name="PyBasketball", src="v_post_yolo", dst="v_py_analysis", module="myapp.basketball", callable="analyze"))
pipe.add(avp.nodes.JoinMetadata(src=["v_dec_1080p", "v_py_analysis"], dst="v_1080p_with_md"))

rt.load(pipe)
rt.start()
state = rt.state()
nodes = rt.nodes()
queues = rt.queues()
rt.stop()
rt.shutdown()
```

### Runtime API requirements

- `Runtime.load(pipeline)`
- `Runtime.start()`
- `Runtime.stop()`
- `Runtime.shutdown()`
- `Runtime.state()`
- `Runtime.state_details()`
- `Runtime.nodes()`
- `Runtime.queues()`
- `Runtime.groups()`
- `Runtime.wait_for_state(target, timeout=None)`
- `Runtime.wait_for_stop(timeout=None)`

## Pipeline and Graph Authoring

### Requirements

- Graphs must be constructible entirely in Python.
- Typed node classes must be the primary authoring surface.
- A raw escape hatch must exist for unsupported or experimental nodes.
- The Python API must support import and export of an internal graph representation compatible with existing avplumber node JSON semantics.

### Required authoring methods

- `Pipeline.add(node)`
- `Pipeline.add_raw(dict_obj)`
- `Pipeline.plan_queue(name, capacity)`
- `Pipeline.hwaccel_init(name, type, **kwargs)`
- `Pipeline.to_dict()`
- `Pipeline.to_script()`

## Node Typing and Schema System

This is a required feature, not an optional enhancement.

### Problem

Current node APIs are defined inside per-node `create(...)` implementations. This is not sufficient for:

- IDE autocomplete
- static typing
- generated Python docs
- discoverability
- safe graph construction

### Requirement

Every supported node type must publish a canonical machine-readable schema from C++.

### Schema contents

Each schema must include:

- node type name
- input media kinds
- output media kinds
- required parameters
- optional parameters
- parameter types
- default values
- enum or literal value sets
- array and object element types
- documentation strings
- whether the parameter is create-time only or runtime-mutable
- whether the parameter is binding-only, graph-level, or native node-level

### Python surface generated from schema

The Python package must expose:

- typed node classes per node type
- typed constructor signatures
- nested typed config classes for structured params
- docstrings generated from schema descriptions
- runtime validation before graph load

### Example

```python
yolo = avp.nodes.CudaInferYolo(
    name="Yolo_Infer",
    src="v_pre_yolo",
    dst="v_post_yolo",
    hwaccel="@gpu",
    input_format="RGB",
    conf_thresh=0.20,
    iou_thresh=0.45,
    max_det=20,
    infer_every_n=1,
    metadata_key_out="yolo_detections_v1",
    models=[
        avp.nodes.CudaInferYolo.Model(
            engine="/models/ball.plan",
            class_names=["basketball"],
            output_box_format="end2end_xyxy",
        ),
    ],
)
```

### Schema registry API

The runtime must expose:

- `Runtime.node_types()`
- `Runtime.schema(node_type)`
- `Runtime.schemas()`

This supports dynamic tooling, notebooks, editors, and advanced Python applications.

## Mixed Python and C++ Nodes

Mixed graphs are a core requirement.

### Required behavior

A Python node must be able to sit between C++ nodes as long as edge types are compatible.

Examples:

- C++ decode -> C++ YOLO -> Python analysis -> C++ metadata merge -> C++ draw
- C++ decode -> Python sidecar metadata branch -> `join_metadata` -> C++ output
- Python synthetic metadata source -> C++ metadata reader

### Supported v1 Python node shapes

- `VideoFrame -> VideoFrame`
- `AudioSamples -> AudioSamples`
- optional `Packet -> Packet` if needed later

### Recommended v1 semantics

Python nodes should primarily:

- inspect incoming objects
- maintain Python-side state
- emit updated metadata
- pass through the original frame or emit a synthetic sidecar frame

Python nodes should not be required in v1 to:

- mutate arbitrary underlying FFmpeg objects in place
- own GPU frame memory
- act as full codec or filter replacements

## Read-Only Media Binding From C++ To Python

A one-way binding from C++ media into Python is a required v1 feature.

### Required wrapper types

- `PacketView`
- `VideoFrameView`
- `AudioSamplesView`

### `PacketView` fields

- `pts`
- `dts`
- `timebase`
- `stream_index`
- `size`
- `is_complete`
- optional payload bytes view
- metadata if present

### `VideoFrameView` fields

- `pts`
- `timebase`
- `width`
- `height`
- `pixel_format`
- `is_complete`
- `is_hw_frame`
- metadata
- optional read-only CPU image view
- optional side-data summary in later versions

### `AudioSamplesView` fields

- `pts`
- `timebase`
- `sample_rate`
- `sample_format`
- channel information
- sample count
- optional read-only sample buffer view

### CPU pixel and sample access

Supported in v1:

- read-only views
- explicit lifetime rules
- optional conversion helpers
- no implicit expensive copies unless requested

Not required in v1:

- general GPU pointer exposure
- raw `AVFrame*` ownership handoff

## Python-Originated Synthetic Media Objects

Python must be able to create its own downstream object and pass it into C++.

This is a required feature.

### Scope of the requirement

Python does not need full arbitrary FFmpeg object construction in v1.

Python does need controlled synthetic object construction for:

- metadata sidecar frames
- timestamp-aligned auxiliary streams
- Python-originated analysis branches
- selected downstream metadata-oriented C++ nodes

### Required v1 builder types

- `SyntheticVideoFrame`
- `SyntheticAudioSamples`
- optional `SyntheticPacket` later

### Required fields for synthetic video

- `pts`
- `timebase`
- `width`
- `height`
- `pixel_format`
- metadata payload

### Required fields for synthetic audio

- `pts`
- `timebase`
- sample format fields
- metadata payload

### Important semantics

Synthetic metadata objects are guaranteed to work for:

- metadata transport
- timestamp alignment
- `join_metadata`
- debug overlay consumption
- custom metadata-oriented C++ nodes

Synthetic metadata objects are not automatically guaranteed to work for:

- encoders
- media filters
- arbitrary real-image consumers
- nodes expecting valid pixel or sample planes

### Builder-style API

```python
f = avp.SyntheticVideoFrame(
    pts=frame.pts,
    timebase=frame.timebase,
    width=frame.width,
    height=frame.height,
    pixel_format=frame.pixel_format,
    metadata={"basketball_analysis_py_v1": analysis},
)
```

or:

```python
f = avp.VideoFrameBuilder() \
    .pts(frame.pts, timebase=frame.timebase) \
    .format(width=frame.width, height=frame.height, pixel_format=frame.pixel_format) \
    .metadata({"basketball_analysis_py_v1": analysis}) \
    .build()
```

## Python Node Models

The binding must support two distinct Python execution models.

### 1. Transform node

A Python-backed node exists in the graph and participates as a normal node.

Examples:

- `PyMetadataTransform`
- `PySidecarVideoSource`
- `PyPacketObserver` later

This is the right model for graph-resident basketball analysis.

### 2. Subscription callback

A Python application may subscribe to frames, packets, or samples emitted on an edge.

This is the right model for app-level observation, metrics, external side effects, and interactive tools.

Both models are required.

## Subscriptions and Callbacks

### Requirements

The runtime must support:

- edge subscriptions
- node output subscriptions as a convenience wrapper
- metadata-key-filtered subscriptions
- async callback delivery as the default mode

### Required API

- `Runtime.subscribe_edge(edge_name, callback, mode="async")`
- `Runtime.subscribe_node_output(node_name, callback, mode="async")`
- `Runtime.subscribe_metadata(metadata_key, callback, edge=None, node=None, mode="async")`

### Callback event object

Callbacks must receive a structured event object containing:

- source node name
- edge name
- media kind
- read-only media view
- parsed metadata
- enqueue timestamp if useful

### Delivery modes

#### Async mode

Default mode.

Behavior:

- C++ edge tap captures a safe event payload
- event is delivered on a Python worker thread or event queue
- the media pipeline is not blocked by normal callback processing

#### Sync mode

Advanced mode only.

Behavior:

- callback runs inline from the producing path
- intended only for very small, low-latency hooks
- clearly documented as pipeline-sensitive

### Queueing and backpressure

The spec must define:

- bounded subscription queues
- queue overflow policy
- configurable drop or latest-only strategies
- visibility into dropped callback events

## Pipeline State Model

A first-class pipeline state model is required.

Current node-level working flags are insufficient for Python applications.

### Required top-level states

- `EMPTY`
- `LOADED`
- `STARTING`
- `RUNNING`
- `DEGRADED`
- `STOPPING`
- `STOPPED`
- `FAILED`
- `SHUTDOWN`

### Required semantics

#### `EMPTY`

No graph loaded.

#### `LOADED`

Graph definition exists in the runtime but is not running.

#### `STARTING`

Start was requested and nodes and groups are converging.

#### `RUNNING`

All required graph components are active.

#### `DEGRADED`

The pipeline is alive, but one or more expected components are not healthy.

#### `STOPPING`

Stop was requested and shutdown is in progress.

#### `STOPPED`

The graph is loaded but not actively running.

#### `FAILED`

A required component failed and the runtime cannot consider the graph healthy.

#### `SHUTDOWN`

Runtime was destroyed and may not be reused.

### Required snapshot APIs

- `Runtime.state()`
- `Runtime.state_details()`
- `Runtime.nodes()`
- `Runtime.groups()`
- `Runtime.queues()`

### Required node snapshot fields

- `name`
- `type`
- `group`
- `working`
- `required`
- `last_error`
- current parameter snapshot

### Required group snapshot fields

- `name`
- `desired_state`
- `current_state`
- total node count
- working node count

### Required queue snapshot fields

- queue name
- media kind
- occupancy
- capacity
- rate counters
- timestamp information

## C++ Binding Surface

This spec is for a Python package plus maintainers' C++ binding API.

### Required C++ exported concepts

- `RuntimeHandle`
- `PipelineSpec`
- `NodeSchemaRegistry`
- `SubscriptionHandle`
- `PacketView`
- `VideoFrameView`
- `AudioSamplesView`
- synthetic media builders
- Python-backed node adapter types

### Recommended binding technology

Preferred: `pybind11`

Reasons:

- natural C++ class exposure
- good Python typing and doc support
- manageable object lifetime control
- common ecosystem choice

## Python-Backed Node Adapter Requirements

A Python-backed node adapter must:

- be registerable as a normal avplumber node type
- bind to typed edges
- manage callback execution safely with GIL awareness
- support Python state across frames
- convert C++ media into Python wrapper views
- convert Python synthetic outputs into C++ media objects
- propagate Python exceptions into node failure or configured policy behavior

### Required error policies

At minimum:

- `fail_node`
- `drop_output`
- `log_and_continue`

## Mixed-Graph Example

### Example architecture

- C++ `input`
- C++ `demux`
- C++ `dec_video`
- C++ `cuda_infer_yolo`
- Python `PyMetadataTransform`
- C++ `join_metadata`
- C++ `draw_bbox`
- C++ `draw_text`
- C++ `enc_video`
- C++ `output`

### Basketball use case

- YOLO emits `yolo_detections_v1`
- Python transform reads detection metadata and optional read-only CPU image data
- Python computes shot, pass, and possession aggregates
- Python emits synthetic sidecar frames with `basketball_analysis_py_v1`
- `join_metadata` merges Python-generated metadata onto the main video branch
- downstream overlays use merged metadata for debug-only visualization

## Documentation Requirements

The binding must produce user-facing docs from canonical schemas.

### Required docs outputs

- Python API reference
- node-type reference
- parameter reference
- synthetic frame API reference
- callback and subscription guide
- mixed Python and C++ graph examples
- lifecycle and state guide

### Docs generation principle

Node docs must be generated from the same schema registry that powers typing and validation.

README prose and examples remain supplementary, not canonical.

## Compatibility Requirements

- Existing C++ nodes remain valid and unchanged in core behavior.
- Existing `.avplumber` scripts remain conceptually compatible.
- Python graph authoring may export an equivalent script representation where possible.
- The binding must not require remote control or TCP server enablement.

## Performance and Safety Constraints

### Required safety principles

- C++ owns media lifetime by default.
- Python receives read-only views unless using explicit synthetic builders.
- callback delivery must not silently stall the pipeline in default mode.
- synthetic media objects must validate timestamps and required structural fields.
- Python exception handling must be explicit and policy-driven.

### Required performance principles

- metadata-only workflows should avoid unnecessary frame copies
- read-only CPU plane exposure should avoid hidden conversions unless explicitly requested
- async subscription queues must be bounded
- Python-backed nodes must document where copies occur

## Phased Implementation Plan

### Phase 1: Foundation

- introduce canonical C++ node schema system
- expose runtime lifecycle binding
- expose node, group, and queue snapshots
- expose typed Python node classes for a core subset
- support Python graph construction and loading
- support read-only frame and packet views
- support edge subscriptions in async mode

### Phase 2: Python graph intelligence

- add Python-backed transform node
- add synthetic metadata-bearing video frame emission
- support `join_metadata` sidecar workflows
- provide basketball-analysis example graph in Python
- generate docs from schemas

### Phase 3: Expanded media support

- add audio sample bindings
- add structured packet bindings if not already delivered
- add more node coverage in typed Python surface
- add richer side-data access
- add advanced subscription policies

### Phase 4: Future work

- optional data-backed synthetic frames
- optional GPU read-only view support
- richer introspection and tracing
- editor or notebook tooling based on schema registry

## Open Questions

- Whether typed node classes should be code-generated at build time or created dynamically from schemas at import time.
- Whether synthetic metadata-bearing frames should require explicit width, height, and pixel format or inherit them from a parent frame helper.
- Whether packet emission from Python is required in v1 or can follow later.
- Whether `group` concepts should be surfaced as explicit Python objects or remain metadata on nodes plus runtime group snapshots.
- Whether async callback delivery should integrate with `asyncio` in v1 or start with thread-based delivery only.

## Decision Summary

This spec chooses:

- embedded in-process Python only
- hybrid architecture
- canonical C++ node schemas
- typed Python graph authoring as a required feature
- mixed Python and C++ nodes as a required feature
- read-only C++ to Python media views as a required feature
- Python-originated synthetic metadata frame emission as a required feature
- async edge subscriptions as the default callback model
- first-class pipeline lifecycle state as a required feature

## Acceptance Criteria

The design is successful when a Python application can:

- create an embedded `Runtime`
- inspect available node schemas with full typing metadata
- build a valid graph in Python with typed node classes
- mix at least one Python-backed node with existing C++ nodes
- subscribe to YOLO output frames and inspect metadata in Python
- emit synthetic metadata-bearing sidecar frames from Python
- merge Python-generated metadata into a C++ video branch through existing graph patterns
- start, stop, and monitor the pipeline through a first-class Python lifecycle API
- receive generated documentation and IDE autocomplete for supported nodes
