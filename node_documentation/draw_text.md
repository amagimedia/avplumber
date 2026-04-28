# `draw_text`

`draw_text` renders a small text overlay onto CUDA `NV12` video frames using basketball-analysis metadata attached to each frame.

## What It Does

1. Reads a CUDA `av::VideoFrame` from `src`.
2. Copies the frame into a new CUDA output frame.
3. Reads JSON metadata from the outer `metadata_key`.
4. Looks up the nested analysis object at `analysis_object_key`.
5. Renders two lines in the top-left corner:
   - `SHOTS: <n>`
   - `LAST: <event_type>`

## Metadata Lookup

`draw_text` uses two configurable keys so it does not hardcode the analysis channel:

- `metadata_key`
  Outer `AVFrame` metadata key. Default: `basketball_analysis_v1`
- `analysis_object_key`
  Inner JSON object name inside that metadata blob. Default: `basketball_analysis`

Example lookup flow:

1. Read `frame.metadata["basketball_analysis_v1"]`
2. Parse the JSON payload
3. Read `json["basketball_analysis"]`
4. Read:
   - `analysis.totals.shots`
   - `analysis.event.type`

## Parameters

### Required

- `src`
  Input video edge.

- `dst`
  Output video edge.

### Optional

- `metadata_key`
  Outer frame metadata key to read. Default: `basketball_analysis_v1`.

- `analysis_object_key`
  Inner JSON object name inside the metadata payload. Default: `basketball_analysis`.

- `origin_x`
- `origin_y`
  Top-left text origin in pixels. Defaults: `48`, `48`.

- `font_scale`
  Integer bitmap-font scale factor. Default: `5`.

- `line_spacing`
  Extra pixel spacing between the two text lines. Default: `16`.

- `text_color`
  Named text color. Supported values: `white`, `black`, `red`, `green`, `yellow`, `light_blue`.
  Default: `white`.

- `background_color`
  Named background block color. Same supported values as `text_color`, plus `none` to disable the block.
  Default: `black`.

- `width`
- `height`
- `pixel_format`
- `real_pixel_format`
  Output-format hints used by the node graph.

- `debug_log_every_n`
  Logging interval. Default: `0`.

## Example

```txt
node.add {
  "type": "draw_text",
  "src": "v_annotated_cuda",
  "dst": "v_annotated_text_cuda",
  "metadata_key": "basketball_analysis_v1",
  "analysis_object_key": "basketball_analysis",
  "origin_x": 48,
  "origin_y": 48,
  "font_scale": 5,
  "line_spacing": 16,
  "text_color": "white",
  "background_color": "black",
  "width": 1920,
  "height": 1080,
  "pixel_format": "cuda",
  "real_pixel_format": "nv12"
}
```
