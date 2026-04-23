# Frame Metadata Reference

This document describes the JSON metadata attached to each video frame by the avplumber basketball analysis pipeline. All metadata lives on the AVFrame dictionary as JSON strings keyed by name.

A downstream consumer (LLM, analytics service, etc.) receives one aggregated JSON per frame via the `metadata_dump` node. This document explains how to read it.

## Coordinate System

All bounding boxes use **XYXY format**: `[x1, y1, x2, y2]` where (x1,y1) is top-left and (x2,y2) is bottom-right.

Coordinates are in **model space** (typically 960x544) unless noted otherwise. To convert to source resolution:

```
source_x = model_x * (source_width / model_width)
source_y = model_y * (source_height / model_height)
```

The `model_width` and `model_height` fields in each metadata block give the model input dimensions.

## Metadata Keys

### `yolo_players` — Detections + Tracking + Team Assignment

Contains all object detections from the 9-class YOLO model, enriched by downstream nodes.

```json
{
  "coord_space": "model",
  "model_width": 960,
  "model_height": 544,
  "detections": [
    {
      "cls": 3,
      "label": "Player",
      "conf": 0.94,
      "xyxy": [338.5, 199.0, 391.1, 292.1],
      "track_id": 7,
      "team": 0,
      "team_ab": "A",
      "team_conf": 0.12,
      "jersey_pixels": 420,
      "jersey_cloth_pixels": 310,
      "jersey_skin_pixels": 55,
      "jersey_confidence": 0.82,
      "jersey_uv_hist": [0.0, 0.01, ...],
      "jersey_l_hist": [0.05, 0.12, ...]
    }
  ]
}
```

**Detection fields (all classes):**

| Field | Type | Description |
|-------|------|-------------|
| `cls` | int | Class index: 0=_suppress, 1=Hoop, 2=Period, 3=Player, 4=Ref, 5=Shot Clock, 6=Team Name, 7=Team Points, 8=Time Remaining |
| `label` | string | Human-readable class name |
| `conf` | float | Detection confidence [0,1] |
| `xyxy` | [4]float | Bounding box in model coordinates |

**Player-specific fields (added by tracker + team classifier):**

| Field | Type | Description |
|-------|------|-------------|
| `track_id` | int | Persistent tracking ID across frames. -1 if untracked |
| `team` | int | Team index: 0 or 1. -1 if unassigned |
| `team_ab` | string | `"A"`, `"B"`, or `"?"` |
| `team_conf` | float | Margin between team distance scores (higher = more certain) |
| `jersey_pixels` | int | Total pixels in segmentation mask for this player |
| `jersey_cloth_pixels` | int | Non-skin pixels in jersey area |
| `jersey_skin_pixels` | int | Skin-colored pixels in jersey area |
| `jersey_confidence` | float | Confidence that the jersey histogram is reliable |
| `jersey_uv_hist` | [256]float | UV chrominance histogram of jersey (normalized) |
| `jersey_l_hist` | [16]float | Luminance histogram of jersey (normalized) |

### `yolo_ball` — Ball Detection + Tracking

```json
{
  "coord_space": "model",
  "model_width": 960,
  "model_height": 544,
  "detections": [
    {
      "cls": 0,
      "label": "basketball",
      "conf": 0.87,
      "xyxy": [542.0, 325.3, 555.4, 341.3],
      "track_id": 0,
      "source": "detected"
    }
  ],
  "trail": [
    {"x": 548.7, "y": 333.3, "conf": 0.87, "frame_age": 0},
    {"x": 545.2, "y": 330.1, "conf": 0.82, "frame_age": 1}
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `source` | string | `"detected"` (YOLO saw it), `"coasted"` (predicted from motion), `"lost"` |
| `trail` | array | Recent ball positions, newest first. `frame_age` = how many frames ago |

### `ball_handler` — Ball Possession

```json
{
  "coord_space": "model",
  "model_width": 960,
  "model_height": 544,
  "detections": [
    {
      "label": "BallHandler",
      "conf": 0.94,
      "xyxy": [338.5, 199.0, 391.1, 292.1],
      "track_id": 7,
      "ball_distance": 12.3
    }
  ]
}
```

The `detections` array has 0 or 1 entries. When present, the player at `track_id` is deemed to be handling the ball. `ball_distance` is the pixel distance between ball center and player bbox in model coordinates.

### `shot_info` — Shot Type Classification

```json
{
  "shot_type": "wide",
  "shot_transition": false
}
```

| Field | Type | Values |
|-------|------|--------|
| `shot_type` | string | `"wide"`, `"closeup"`, `"ambiguous"` |
| `shot_transition` | bool | True on the first frame of a new shot type |

### `yolo_seg` — Court Segmentation

```json
{
  "model_width": 960,
  "model_height": 544,
  "detections": [
    {
      "cls": 0,
      "conf": 0.95,
      "xyxy": [0, 0, 960, 544],
      "label": "basketball-court"
    },
    {
      "cls": 1,
      "conf": 0.88,
      "xyxy": [120, 200, 840, 500],
      "label": "three point line"
    }
  ]
}
```

Court segmentation provides bounding boxes for the court and three-point line regions. CPU pixel masks are available in frame side data (slot 0) but are not included in the JSON — only bounding boxes.

### `yolo_players_seg` — Player Segmentation + Jersey Analysis

Same structure as `yolo_players` but from the segmentation model. Each detection has jersey color histograms and pixel counts attached by `jersey_color_extract`. The `team` and `cls` fields are overwritten by the team classifier to reflect team assignment.

### `scoreboard` — OCR Scoreboard Data

```json
{
  "team_a": {
    "name": {"text": "DET", "conf": 0.94},
    "points": {"text": "15", "conf": 1.0}
  },
  "team_b": {
    "name": {"text": "MIA", "conf": 0.99},
    "points": {"text": "12", "conf": 0.98}
  },
  "period": {"text": "1ST", "conf": 0.99},
  "time_remaining": {"text": "11:23", "conf": 0.95},
  "shot_clock": {"text": "14", "conf": 0.99}
}
```

All fields are optional — only present when the YOLO model detects the corresponding scoreboard region and OCR produces a non-empty result. Team names are extracted from leading uppercase abbreviations found across all scoreboard detections. `team_a` is the left-side team, `team_b` is the right-side team.

Note: `team_a`/`team_b` here does NOT directly correspond to `team: 0`/`team: 1` in player detections. The player team assignment is based on jersey color clustering, while scoreboard teams are based on spatial position.

### `smoothed_crop_viewport_v1` — Auto-Framing Viewport

```json
{
  "detections": [
    {
      "label": "viewport",
      "xyxy": [200.5, 0.0, 760.5, 544.0]
    }
  ]
}
```

The viewport rectangle for automatic reframing (portrait crop from landscape). Coordinates are in model space.

## Aggregated Output Format

The `metadata_dump` node combines all the above into a single compact JSON per frame:

```json
{
  "pts": 12345678,
  "frame": 308,
  "w": 1920,
  "h": 1080,
  "model_w": 960,
  "model_h": 544,
  "shot": "wide",
  "clock": "11:23",
  "period": "1ST",
  "shot_clock": "14",
  "teams": {
    "a": {"name": "DET", "score": "15"},
    "b": {"name": "MIA", "score": "12"}
  },
  "players": [
    {
      "id": 7,
      "team": "A",
      "box": [338, 199, 391, 292],
      "conf": 0.94,
      "has_ball": true
    },
    {
      "id": 12,
      "team": "B",
      "box": [700, 190, 749, 304],
      "conf": 0.92,
      "has_ball": false
    }
  ],
  "refs": [
    {"box": [64, 317, 126, 432], "conf": 0.63}
  ],
  "ball": {
    "box": [542, 325, 555, 341],
    "conf": 0.87,
    "state": "detected",
    "handler_id": 7
  },
  "hoop": {
    "box": [96, 111, 130, 142],
    "conf": 0.45
  },
  "viewport": [200, 0, 760, 544]
}
```

### Design Principles

- **Minimal bytes per frame.** Field names are short. Jersey histograms, engine names, and model metadata are stripped. Bounding boxes are rounded to integers.
- **One object per player.** Tracking, team, and ball handler info are merged into a single `players` array entry. No cross-referencing between separate metadata blocks needed.
- **LLM-readable.** A language model receiving this JSON can immediately answer: "Which team has the ball?", "What's the score?", "Where are the players?", "Is this a wide shot or closeup?"
- **Stable IDs.** Player `id` values persist across frames via the tracker. The ball `handler_id` references a player `id`.
