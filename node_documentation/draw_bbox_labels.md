# `draw_bbox_labels`

`draw_bbox_labels` renders metadata-driven text labels near YOLO bounding boxes on CUDA `NV12` frames.

## What It Does

1. Reads a CUDA `av::VideoFrame` from `src`.
2. Copies the frame into a new CUDA output frame.
3. Parses YOLO detections from `metadata_key`/`metadata_keys`.
4. Filters detections with the same rules as `draw_bbox` (`min_conf`, `allowed_classes`, `allowed_labels`).
5. Formats label text from `label_template`.
6. Draws multiline labels anchored at outer top-left of each bbox.

## Template Fields

Supported placeholders:

- `{track_id}`
- `{conf}`
- `{velocity}`
- `{label}`
- `{cls}`

Use `\n` for multiline text.

Default template:

`ID:{track_id}\nV:{velocity}`

Line behavior:

- If a field in a line is missing, that line is skipped.
- If all lines are skipped for a detection, no label is drawn.

## Parameters

### Required

- `src`
- `dst`

### Optional

- `metadata_key`
  Default: `yolo_players`.

- `metadata_keys`
  Alternative list of metadata keys. Duplicates are removed.

- `min_conf`
  Minimum YOLO confidence. Default: `0.0`.

- `allowed_classes`
  Optional class ID allowlist.

- `allowed_labels`
  Optional label allowlist.

- `label_template`
  Label template with placeholders and `\n` separators. Default: `ID:{track_id}\nV:{velocity}`.

- `velocity_precision`
  Decimal precision for `{velocity}` formatting. Default: `1`.

- `show_predicted_labels`
  Draw labels for detections with `predicted=true`. Default: `false`.

- `show_untracked`
  Draw labels when `track_id < 0`. Default: `false`.

- `glyph_preset`
  Built-in glyph set. One of: `5x7`, `10x14`. Default: `10x14`.

- `font_scale`
  Integer glyph scale. Default: `2`.

- `line_spacing`
  Pixel spacing between lines. Default: `8`.

- `offset_x`
- `offset_y`
  Label anchor offsets relative to bbox top-left. Defaults: `0`, `0`.

- `text_color`
  Named color. Default: `white`.

- `background_color`
  Named color or `none`. Default: `black`.

- `background_opacity`
  Background alpha in `[0,1]`. Default: `0.75`.

- `model_content_width`
- `model_content_height`
- `model_content_offset_x`
- `model_content_offset_y`
  Same model-content remap behavior as `draw_bbox`.

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
  "type": "draw_bbox_labels",
  "src": "v_bbox",
  "dst": "v_bbox_labels",
  "metadata_key": "yolo_players",
  "allowed_labels": ["Player", "Ref"],
  "label_template": "ID:{track_id}\nV:{velocity} px/f",
  "glyph_preset": "10x14",
  "font_scale": 2,
  "text_color": "white",
  "background_color": "black",
  "background_opacity": 0.75
}
```
