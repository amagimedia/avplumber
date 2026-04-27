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

### `camera_shot_info` — Camera Shot Classification

```json
{
  "camera_shot_type": "wide",
  "camera_shot_transition": false
}
```

| Field | Type | Values |
|-------|------|--------|
| `camera_shot_type` | string | `"wide"`, `"closeup"`, `"ambiguous"` |
| `camera_shot_transition` | bool | True on the first frame of a new camera shot type |

### `shot_events` — Shot Attempt State

```json
{
  "release": false,
  "hoop_arrival": false,
  "in_flight": true,
  "ball_detected": true,
  "hoop_detected": true,
  "ball_in_air": true,
  "ball_hoop_dist": 74,
  "shot_result": true,
  "result": "scored",
  "result_source": "ball_vector",
  "result_conf": 0.82,
  "total_releases": 3,
  "total_hoop_arrivals": 2
}
```

| Field | Type | Description |
|-------|------|-------------|
| `release` | bool | True on the frame a shot release is confirmed |
| `hoop_arrival` | bool | True on the frame the tracked ball reaches the hoop radius |
| `in_flight` | bool | True while a confirmed shot is in flight |
| `ball_detected` | bool | True when the ball tracker has a ball for this frame |
| `hoop_detected` | bool | True when a hoop detection is present above the confidence threshold |
| `ball_in_air` | bool | True when the ball is detected and no handler is assigned |
| `ball_hoop_dist` | int | Ball-center to hoop-center distance in model pixels, when both are visible |
| `shot_result` | bool | True on the frame a synthesized result is available |
| `result` | string | `"scored"` or `"missed"` from near-hoop ball-vector analysis |
| `result_source` | string | Source of the synthesized result, currently `"ball_vector"` |
| `result_conf` | float | Confidence derived from the downward component of the near-hoop vector |
| `result_vx`, `result_vy` | float | Ball vector used for result classification |
| `attempt_type` | string | Currently `"unknown"`; 2pt/3pt classification is disabled until shooter/line geometry is reliable |
| `attempt_points` | int | Reserved for future 2pt/3pt classification |
| `points` | int | Reserved for future scored point totals once shot value is reliable |
| `release_frame` | int | Frame number where the current release sequence started |
| `total_releases` | int | Running count of confirmed shot releases |
| `total_hoop_arrivals` | int | Running count of confirmed hoop-arrival events |

### `possession_state` — Team Possession Summary

```json
{
  "possessing_team": "A",
  "handler_id": 7,
  "handler_team": "A",
  "ball_state": "controlled",
  "frames_in_possession": 37,
  "possession_id": 5
}
```

| Field | Type | Description |
|-------|------|-------------|
| `possessing_team` | string | `"A"` or `"B"` when possession can be assigned to a team |
| `handler_id` | int | Current tracked ball-handler ID when the ball is controlled |
| `handler_team` | string | Team of the current handler (`"A"` or `"B"`) |
| `ball_state` | string | `"controlled"`, `"loose"`, `"shot_in_air"`, or `"dead_or_unknown"` |
| `frames_in_possession` | int | Frames elapsed in the current stable possession window |
| `possession_id` | int | Monotonic possession counter, incremented when a new team possession is confirmed |

### `court_zone` — Offensive Court Context

```json
{
  "zone_source": "handler",
  "handler_zone": "right_wing",
  "ball_zone": "right_wing",
  "hoop_side": "right",
  "inside_court": true,
  "inside_three_point_area": true,
  "relative_to_hoop_x": -182,
  "relative_to_hoop_y": 46,
  "distance_to_hoop": 188
}
```

This is a model-space court context summary derived from the ball-handler / ball positions, hoop detection, and the `yolo_seg` CPU masks. It is heuristic, but stable enough to tell a downstream LLM whether the live action is in the paint, wing, corner, top, or backcourt.

| Field | Type | Description |
|-------|------|-------------|
| `zone_source` | string | Which point drove the main zone summary: `"handler"` or `"ball"` |
| `handler_zone` | string | Zone for the handler point when a handler exists |
| `ball_zone` | string | Zone for the ball point when the ball exists |
| `hoop_side` | string | `"left"` or `"right"` hoop in model-space broadcast view |
| `inside_court` | bool | True when the main point lies inside the court mask |
| `inside_three_point_area` | bool | True when the main point lies inside the three-point boundary toward the hoop |
| `relative_to_hoop_x` | int | Main point x offset from hoop center in model pixels |
| `relative_to_hoop_y` | int | Main point y offset from hoop center in model pixels |
| `distance_to_hoop` | int | Main point to hoop-center distance in model pixels |

Zone values currently are:
- `paint`
- `left_corner`
- `right_corner`
- `left_wing`
- `right_wing`
- `top`
- `backcourt`
- `unknown`

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

### `game_state` — Parsed Scoreboard State

```json
{
  "period_num": 1,
  "game_clock_sec": 683,
  "shot_clock_sec": 14,
  "team_a_abbrev": "DET",
  "team_b_abbrev": "MIA",
  "score_a": 15,
  "score_b": 12,
  "score_margin": 3,
  "leading_team": "DET"
}
```

This is a forgiving parser over the raw `scoreboard` OCR output. It normalizes clocks, scores, and team abbreviations into typed fields for downstream consumers.

| Field | Type | Description |
|-------|------|-------------|
| `period_num` | int | Parsed period number, e.g. `1..4`, `5` for OT |
| `game_clock_sec` | int | Parsed game clock in seconds |
| `shot_clock_sec` | int | Parsed shot clock in seconds |
| `team_a_abbrev` | string | Parsed left-side team abbreviation |
| `team_b_abbrev` | string | Parsed right-side team abbreviation |
| `score_a` | int | Parsed left-side score |
| `score_b` | int | Parsed right-side score |
| `score_margin` | int | Absolute score difference |
| `leading_team` | string | Leading team abbreviation, or `"TIE"` |

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
  "camera_shot": "wide",
  "camera_shot_transition": true,
  "ball_hoop_dist": 74,
  "clock": "11:23",
  "period": "1ST",
  "shot_clock": "14",
  "game_state": {
    "period_num": 1,
    "game_clock_sec": 683,
    "shot_clock_sec": 14,
    "team_a_abbrev": "DET",
    "team_b_abbrev": "MIA",
    "score_a": 15,
    "score_b": 12,
    "score_margin": 3,
    "leading_team": "DET"
  },
  "possession": {
    "possessing_team": "A",
    "handler_id": 7,
    "handler_team": "A",
    "ball_state": "controlled",
    "frames_in_possession": 37,
    "possession_id": 5
  },
  "shot_events": {
    "release": false,
    "hoop_arrival": false,
    "in_flight": true,
    "ball_detected": true,
    "hoop_detected": true,
    "ball_in_air": true,
    "ball_hoop_dist": 74,
    "total_releases": 3,
    "total_hoop_arrivals": 2
  },
  "teams": {
    "a": {"name": "DET", "score": "15"},
    "b": {"name": "MIA", "score": "12"}
  },
  "players": [
    {
      "id": 7,
      "team": "A",
      "box": [338, 199, 391, 292],
      "conf": 94,
      "team_conf": 12,
      "has_ball": true,
      "vel": [3, -1],
      "tracklet_len": 45,
      "torso": [346, 216, 383, 252],
      "outline": [[340, 200], [350, 199], [388, 210], [390, 290], [340, 291]]
    },
    {
      "id": 12,
      "team": "B",
      "box": [700, 190, 749, 304],
      "conf": 92,
      "team_conf": 8,
      "vel": [-2, 0],
      "tracklet_len": 120,
      "torso": [707, 210, 742, 256],
      "outline": [[702, 192], [745, 195], [748, 300], [701, 302]]
    }
  ],
  "refs": [
    {"box": [64, 317, 126, 432], "conf": 63}
  ],
  "ball": {
    "box": [542, 325, 555, 341],
    "conf": 87,
    "state": "detected",
    "vel": [8, -12],
    "handler_id": 7,
    "handler_team": "A",
    "handler_dist": 12
  },
  "hoop": {
    "box": [96, 111, 130, 142],
    "conf": 45
  },
  "viewport": [200, 0, 760, 544]
}
```

### Field Reference

**Top-level frame info:**

| Field | Type | Description |
|-------|------|-------------|
| `pts` | int | Presentation timestamp |
| `frame` | int | Frame counter since start |
| `w`, `h` | int | Source frame resolution |
| `model_w`, `model_h` | int | Model input resolution (coordinate space for boxes) |
| `camera_shot` | string | `"wide"`, `"closeup"`, or `"ambiguous"` |
| `camera_shot_transition` | bool | Present (true) on the first frame of a new camera shot type |
| `ball_hoop_dist` | int | Ball-center to hoop-center distance in model pixels, when both are visible |
| `clock` | string | Game clock text from OCR |
| `period` | string | Game period text from OCR |
| `shot_clock` | string | Shot clock text from OCR |
| `teams` | object | `a` and `b` sub-objects with `name` and `score` strings |
| `game_state` | object | Parsed scoreboard/game-state summary from `game_state` |
| `possession` | object | Per-frame possession state from `possession_tracker` |
| `shot_events` | object | Per-frame shot attempt state from `shot_attempt_detector` |
| `viewport` | [4]int | Auto-framing crop rectangle [x1, y1, x2, y2] in model coords |

**Player fields (`players` array):**

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Persistent tracking ID across frames |
| `team` | string | `"A"` or `"B"` (omitted if unassigned) |
| `box` | [4]int | Bounding box [x1, y1, x2, y2] in model coords |
| `conf` | int | Detection confidence 0-100 |
| `team_conf` | int | Team assignment margin 0-100 (higher = more certain) |
| `has_ball` | bool | Present (true) if this player is the ball handler |
| `vel` | [2]int | Kalman velocity [dx, dy] in pixels/frame from tracker |
| `tracklet_len` | int | Number of consecutive frames this track has been active |
| `track_state` | string | Omitted when `"tracked"`. `"lost"` or `"new"` when abnormal |
| `torso` | [4]int | Torso region [x1, y1, x2, y2] used for jersey analysis |

**Ball fields (`ball` object):**

| Field | Type | Description |
|-------|------|-------------|
| `box` | [4]int | Bounding box [x1, y1, x2, y2] in model coords |
| `conf` | int | Detection confidence 0-100 |
| `state` | string | `"detected"`, `"coasted"`, or `"lost"` |
| `vel` | [2]int | Kalman velocity [dx, dy] in pixels/frame |
| `coast_streak` | int | Consecutive frames without detection (only when > 0) |
| `handler_id` | int | Track ID of the player holding the ball |
| `handler_team` | string | Team of the handler (`"A"`, `"B"`, or `"?"`) |
| `handler_dist` | int | Pixel distance from ball center to handler bbox |

### Design Principles

- **Minimal bytes per frame.** Field names are short. Jersey histograms, engine names, and model metadata are stripped. Bounding boxes are rounded to integers.
- **One object per player.** Tracking, team, ball handler, velocity, torso, and segmentation outline are merged into a single `players` array entry. No cross-referencing between separate metadata blocks needed.
- **LLM-readable.** A language model receiving this JSON can immediately answer: "Which team has the ball?", "What's the score?", "Where are the players?", "Who is moving fastest?", "Is this a wide camera shot or closeup?"
- **Stable IDs.** Player `id` values persist across frames via the tracker. The ball `handler_id` references a player `id`.
- **Sparse by default.** Fields like `camera_shot_transition`, `coast_streak`, `track_state` are only emitted when they carry signal (transition happening, ball lost, track abnormal). Absence means normal state.

## Sidecar Files (Geometry & Trails)

Heavy per-frame geometry is written to separate JSON files to keep the main dump lean. Each sidecar is a JSON array of objects keyed by `pts` (presentation timestamp), matching the main file.

Configure via node params: `output_file_court`, `output_file_outlines`, `output_file_trail`. If a param is omitted, that sidecar is not written.

### Court outlines (`*_court.json`)

```json
[
  {
    "pts": 12345678,
    "regions": [
      {"label": "basketball-court", "outline": [[0,200],[960,200],[960,540],[0,540]]},
      {"label": "three point line", "outline": [[120,220],[840,220],[840,500],[120,500]]}
    ]
  }
]
```

One entry per dumped frame. Each `regions` entry has a segmentation class `label` and an `outline` polygon (vertices in model coords). Court outlines change slowly — many consecutive frames will have near-identical data.

### Player outlines (`*_outlines.json`)

```json
[
  {
    "pts": 12345678,
    "players": [
      {"id": 7, "outline": [[340,200],[350,199],[388,210],[390,290],[340,291]]},
      {"id": 12, "outline": [[702,192],[745,195],[748,300],[701,302]]}
    ]
  }
]
```

One entry per dumped frame. Each player is identified by `id` (matching the main dump's player `id`). The `outline` is the segmentation mask contour clipped to the player's bounding box, in model coordinates.

### Ball trail (`*_trail.json`)

```json
[
  {
    "pts": 12345678,
    "trail": [[548,333,308],[545,330,307],[540,328,306]]
  }
]
```

One entry per dumped frame. Each trail point is `[x, y, frame_number]` in model coordinates, most recent first. Trail grows up to 200 points.

### Merging sidecar data with the main dump

All files share the same `pts` values. To enrich a main-dump frame with geometry:

```python
import json

frames = json.load(open("metadata_dump.json"))
court = {e["pts"]: e["regions"] for e in json.load(open("metadata_dump_court.json"))}
outlines = {e["pts"]: {p["id"]: p["outline"] for p in e["players"]} for e in json.load(open("metadata_dump_outlines.json"))}
trails = {e["pts"]: e["trail"] for e in json.load(open("metadata_dump_trail.json"))}

for f in frames:
    pts = f["pts"]
    if pts in court:
        f["court"] = court[pts]
    if pts in trails:
        f["ball_trail"] = trails[pts]
    if pts in outlines:
        for p in f.get("players", []):
            if p.get("id") in outlines[pts]:
                p["outline"] = outlines[pts][p["id"]]
```

The main dump is designed to be self-sufficient for game understanding — an LLM can answer questions about score, possession, player positions, and camera shot type without the sidecars. The sidecars add spatial precision (exact player silhouettes, court boundaries, ball trajectory) for tasks that need it.
