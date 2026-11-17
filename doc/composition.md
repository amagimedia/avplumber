# Generic GPU composition

`pyplumber.composition` provides source-paced composition, native compositor
construction, crop geometry, backdrop effects and frame-timed move descriptions.
It is part of AVPlumber core and has no dependency on the mixer application or
private AI nodes. Importing it does not load the native extension until a native
node is constructed.

The mixer application uses this API through its CUDA backend. Reframers can
supply per-frame placement metadata without importing mixer configuration,
Preview/Program control, a show director or inference models.

## Timing

| API | Timing and purpose |
| --- | --- |
| `source_compositor(params)` / `compositor(params, timing="source")` | `CudaRectOverlay`: corresponding input frames, source PTS and EOF. |
| `compositor(params, timing="clock")` | `MixerCompositor`: independent inputs on the shared output clock. |
| `compositor(params, timing="clock", keyer=True)` | `MixerKeyer`: clocked downstream keys. |

Native parameters are passed through unchanged. The caller owns input routing,
CUDA device lifetime and encode/output. Clocked composition retains the mixer's
playout, latency and take-control behavior. A source-paced graph does not create
a mixer output clock. All variants use the existing native drawing primitives.

```python
from pyplumber.composition import compositor

program = compositor({
    "name": "program",
    "src": ["camera_a", "camera_b"],
    "dst": "program_video",
    "hwaccel": "@gpu",
    "width": 1920, "height": 1080,
    "sw_format": "nv12", "color": "sdr",
    "fps": "25/1",
    "layers": [{"input": 0, "dst_w": 1920, "dst_h": 1080}],
}, timing="clock")
```

## One-source composition graph

`SourceCompositionGraph` fans a single source into content pads and an optional
blurred backdrop. This convenience graph expects SDR NV12 CUDA frames. Other
working formats and colour contracts use the native construction API above.

```python
from pyplumber import node
from pyplumber.composition import SourceCompositionConfig, SourceCompositionGraph

cfg = SourceCompositionConfig(
    dest_width=1080, dest_height=1920,
    source_width=1920, source_height=1080,
    fps="25/1", background="blur", max_layers=3,
    metadata_key="composition_layers",
    move_metadata_key="composition_move",
    name="Portrait", edge_prefix="portrait", dst="portrait_video",
)
nodes = []
output = SourceCompositionGraph(cfg, node).build(nodes, "decoded_video")
# Add nodes to AVPlumber and connect output to the caller's encoder.
```

Choose distinct `name`, `edge_prefix` and `dst` values for multiple instances.
Every branch uses `hwaccel`; all pads remain active and drain at EOF. The graph
stamps the declared frame rate and BT.709 limited-range tags without downloading
frames. It does not select subjects or change the caller's framing decisions.

The blur defaults preserve the existing area-downsampled backdrop: centre crop,
shrink in stages of at most 4, apply a 30% white-wash LUT before bicubic enlargement.
`blur_fraction`, `blur_step` and `wash` expose those parameters. The renderer
performs pixel operations on CUDA; Python builds parameters and the static LUT.

## Geometry and motion

`geometry.layers_from_plan` converts normalized `src`/`tgt` rectangles to the
existing even-aligned NV12 metadata contract, with configurable pad count.
`geometry.source_crop`, `scaling.center_crop` and `scaling.cuda_scale` provide
standalone geometry/scaling helpers. Fractional crops, offscreen destinations,
rotation, mirror images (`flip_h`, `flip_v`), effects and source clipping can be
supplied directly as native layer fields; they are not restricted by the
normalized-placement convenience helper. A flip mirrors the layer's crop in its
box before the rotation; both flips together are a half turn.

```python
import json

from pyplumber.composition import SourceMove

take = SourceMove(old_layers, new_layers, start=17.0, duration=0.8,
                  curve="ease-in-out", unmatched="fade")
frame.metadata["composition_move"] = json.dumps(take.at(source_pts_seconds))
```

Endpoints are copied when the move is
created, so changing detections cannot wobble an active take. Progress is computed
from source PTS; native code evaluates the existing curve. Supported curves are
`linear`, `ease-in`, `ease-out` and `ease-in-out`.

`pyplumber.mixer.SourceMove` and `pyplumber.mixer.source_compositor` remain
compatibility imports. Reusable callers should import `pyplumber.composition`.
