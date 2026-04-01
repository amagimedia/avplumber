# CUDA Compositor Node Design

## Overview

A new `cuda_compositor` node that takes N CUDA video inputs (two cameras + one RGBA dmabuf overlay from Chromium) and composites them onto a single output canvas. Composition parameters (crop, scale, position per input) are driven by per-frame metadata from an upstream layout decision node, enabling continuous parametric layout changes driven by YOLO detection analysis.

## Pipeline Context

```
cameras → YOLO → analysis → layout_decision_node
                                    ↓ (metadata on frames)
cam1_q, cam2_q, chromium_q → cuda_compositor → output
```

The layout decision node reads detection/analysis metadata and writes composition parameters onto the control input's frame metadata. The compositor reads those parameters and executes the transforms.

**Precondition:** All inputs must be fps-normalized (e.g., via `force_fps` node) to the same frame rate before entering the compositor.

## Node Interface

**Node type:** `cuda_compositor`

**Base classes:**
- `NodeMultiInput<av::VideoFrame>` — N video inputs
- `NodeSingleOutput<av::VideoFrame>` — single composited output
- `IVideoFormatSource`, `IFrameRateSource`, `ITimeBaseSource`

**Creation params:**
- `src` — array of input edge names
- `dst` — output edge name
- `output_width`, `output_height` — canvas dimensions
- `metadata_key` — frame metadata key for layout (default: `"compositor_layout"`)
- `control_input` — which input index carries the metadata (default: 0)
- `hwaccel` — CUDA device name

**Example:**
```
node.add {
  "name": "comp",
  "type": "cuda_compositor",
  "src": ["cam1_q", "cam2_q", "overlay_q"],
  "dst": "comp_out",
  "output_width": 1920,
  "output_height": 1080,
  "metadata_key": "compositor_layout",
  "hwaccel": "cuda0"
}
```

## Metadata Format

The layout decision node writes JSON onto the control input's frame metadata under the configured key:

```json
{
  "inputs": [
    {
      "index": 0,
      "enabled": true,
      "z_order": 0,
      "src_rect": { "x": 100, "y": 50, "w": 1280, "h": 720 },
      "dst_rect": { "x": 0, "y": 0, "w": 1920, "h": 1080 }
    },
    {
      "index": 1,
      "enabled": true,
      "z_order": 1,
      "src_rect": null,
      "dst_rect": { "x": 1400, "y": 50, "w": 480, "h": 270 }
    },
    {
      "index": 2,
      "enabled": true,
      "z_order": 2,
      "src_rect": null,
      "dst_rect": { "x": 50, "y": 850, "w": 300, "h": 200 },
      "opacity": 0.9
    }
  ]
}
```

**Field semantics:**
- `index` — which input (by position in `src` array)
- `enabled` — if false, skip this input
- `z_order` — layering order (0 = bottom/background)
- `src_rect` — crop region from source frame. `null` = full frame. Fields: `x`, `y`, `w`, `h`
- `dst_rect` — placement on output canvas. `x`, `y` = top-left position. `w`, `h` = scale target size
- `opacity` — reserved for future dissolve/transition interface. Default: 1.0

**Alpha detection:** Determined automatically from input frame sw_format (RGBA, YUVA* → alpha path). No metadata flag needed.

**Missing metadata behavior:** Hold last valid layout. Before any metadata is received, default layout is input 0 fullscreen (scaled to canvas size), all other inputs disabled.

## Architecture

### Per-Input Transform

Each input has an independent libav filter graph, built with only the filters needed:

- **No crop, no scale needed** → skip filter graph entirely, direct passthrough
- **Crop only** → `[buffer] → crop_cuda → [buffersink]`
- **Scale only** → `[buffer] → scale_cuda → [buffersink]`
- **Crop + scale** → `[buffer] → crop_cuda → scale_cuda → [buffersink]`
- **Alpha input needing format conversion** → append `convert_cuda` to the chain (e.g., `crop? → scale? → convert_cuda → [buffersink]`)

Filter graph lifecycle:
- Built lazily on first frame
- Rebuilt when crop w/h or scale w/h changes from previous frame
- Crop x/y updates via `avfilter_process_command()` (no rebuild needed)
- Skipped when parameters are stable across frames

### Composition Step

Two paths, can be mixed in a single frame:

**Opaque path (fast):** `cuMemcpy2D` the transformed input directly onto the output canvas at `dst_rect` position. Used for inputs whose sw_format has no alpha channel (NV12). Copies per-plane: Y plane full size, UV plane half height (NV12 layout).

**Alpha path:** For inputs whose sw_format has alpha (RGBA, YUVA*, etc.):
1. Per-input filter graph outputs YUVA frame (convert_cuda included in chain)
2. Allocate YUVA scratch frame at canvas size, clear to transparent (`cuMemsetD8` per plane: Y=0, U=128, V=128, A=0)
3. `cuMemcpy2D` the YUVA frame onto the scratch at `dst_rect` position
4. Feed scratch frame(s) + canvas to `overlay_many_cuda`

**Composition order:**
1. Allocate output CUDA frame (NV12 canvas at output_width × output_height), fill with black
2. Process inputs sorted by z_order ascending
3. Opaque inputs: `cuMemcpy2D` onto canvas
4. After all opaque inputs: if any alpha inputs exist, run `overlay_many_cuda` pass

**Overlap handling:** For opaque inputs, higher z_order overwrites lower. The layout decision node is responsible for not creating nonsensical overlaps. If alpha blending is needed between inputs (e.g., during transitions), the layout node should use inputs that inherently have alpha.

### Frame Synchronization

All inputs are fps-normalized (precondition). The compositor pops one frame from each input per cycle. If an input has no frame ready, the last frame from that input is held and reused.

### Filter Graph Change Detection

Per-input state tracks last-applied parameters:

```cpp
struct InputSlot {
    AVFilterGraph *filter_graph = nullptr;
    AVFilterContext *buffersrc_ctx = nullptr;
    AVFilterContext *crop_ctx = nullptr;   // null if no crop needed
    AVFilterContext *scale_ctx = nullptr;  // null if no scale needed
    AVFilterContext *convert_ctx = nullptr; // null if no format conversion needed
    AVFilterContext *buffersink_ctx = nullptr;
    AVBufferRef *hw_frames_ctx = nullptr;

    // Change detection
    int last_src_x = 0, last_src_y = 0, last_src_w = 0, last_src_h = 0;
    int last_dst_w = 0, last_dst_h = 0;
    bool has_alpha = false;

    // Frame hold
    av::VideoFrame current_frame;
    av::VideoFrame transformed_frame; // output of filter graph
};
```

**Change detection logic:**
1. `src_rect` w/h or `dst_rect` w/h changed → rebuild filter graph
2. Only `src_rect` x/y changed → `avfilter_process_command()` on crop_ctx
3. Nothing changed → reuse existing graph, just push frame through

### Alpha Composition Filter Graph

Separate filter graph for `overlay_many_cuda`:

```
[canvas]    → overlay_many_cuda → [output]
[alpha_in0] ↗
[alpha_in1] ↗
```

- Built lazily on first frame with active alpha inputs
- Rebuilt only when the count of active alpha inputs changes
- Alpha inputs are pre-positioned via `cuMemcpy2D` onto YUVA scratch frames (not via `pad_cuda`), so position changes don't trigger graph rebuilds

### Output hw_frames_ctx

Allocate own `AVHWFramesContext` from the hwaccel device at `output_width × output_height` with NV12 sw_format. Allocated once on first frame, reused for all output frames.

For YUVA scratch frames (alpha input positioning), allocate a second `AVHWFramesContext` at canvas size with YUVA420P sw_format.

## Class Structure

```cpp
class CudaCompositor : public NodeMultiInput<av::VideoFrame>,
                       public NodeSingleOutput<av::VideoFrame>,
                       public IVideoFormatSource,
                       public IFrameRateSource,
                       public ITimeBaseSource
```

**Key methods:**
- `process()` — main loop: pop frames, parse layout, transform inputs, compose output
- `parseLayout(const av::VideoFrame&)` — extract and validate JSON from frame metadata
- `updateInputFilterGraph(InputSlot&, LayoutEntry&)` — rebuild or command-update per-input graph
- `processInputFrame(InputSlot&)` — push frame through filter graph, retrieve result
- `composeOpaque(AVFrame* canvas, InputSlot&, LayoutEntry&)` — cuMemcpy2D onto canvas (NV12, 2 planes)
- `composeAlpha(AVFrame* canvas, vector<InputSlot*>, vector<LayoutEntry*>)` — position onto YUVA scratch frames, run overlay_many_cuda
- `allocateOutputFrame()` — allocate CUDA NV12 frame at canvas size, fill black

## Pixel Format Handling

- **Opaque inputs:** Must be `AV_PIX_FMT_CUDA` with NV12 sw_format. If an input has a different opaque sw_format (e.g., YUV420P), the per-input filter graph includes `scale_cuda` to convert to NV12.
- **Alpha inputs:** `AV_PIX_FMT_CUDA` with RGBA or similar sw_format (from `drm_prime_to_cuda`). Per-input filter graph includes `convert_cuda` to produce YUVA420P.
- **Output canvas:** `AV_PIX_FMT_CUDA` with NV12 sw_format.
- **Alpha scratch frames:** YUVA420P sw_format at canvas dimensions.

## Transition Interface (not implemented initially)

The `opacity` field in metadata and a future `transition` object provide the interface for dissolve/fade:

```json
{
  "index": 1,
  "opacity": 0.5,
  "transition": { "type": "dissolve", "progress": 0.5 }
}
```

Initial implementation ignores `opacity` (treats as 1.0) and `transition`. The architecture supports adding this later by modulating the alpha channel before the overlay_many_cuda pass.

## Dependencies

- **FFmpeg CUDA filters:** `scale_cuda`, `crop_cuda`, `convert_cuda`, `overlay_many_cuda`
- **CUDA driver API:** `cuMemcpy2D`, `cuMemsetD8` for canvas operations
- **Existing avplumber:** `NodeMultiInput`, `NodeSingleOutput`, `HWAccelDevice`, `IVideoFormatSource`, `IFrameRateSource`, `ITimeBaseSource`
