# Player Tracker Design Spec

Multi-object player tracker for avplumber, integrating ByteTrack C++ into the neural_net pipeline.

## Problem

We detect all players per frame via `cuda_infer_yolo` but have no way to assign persistent unique IDs across frames. The existing `ball_tracker` is single-object only. We need multi-object tracking so downstream nodes can do per-player analytics and on-screen overlays with stable numbered bounding boxes.

## Approach

Copy ByteTrack's C++ implementation (~1150 lines) into `deps/bytetrack/`, modify it to remove OpenCV dependency and fix issues, then wrap it in a new `player_tracker` avplumber node in `sport_specific/`.

ByteTrack was chosen because:
- Well-proven two-stage association algorithm (high-confidence match, then low-confidence match)
- Small, self-contained C++ with minimal dependencies (only Eigen)
- Archived/stable upstream -- no moving target

The relevant C++ files are copied from the ByteTrack upstream repo (`ByteTrack/deploy/TensorRT/cpp/`) into `deps/bytetrack/` and committed as part of this project. Only the 10 core tracking files are needed -- no inference demo, no Python code.

## Data Flow

```
cuda_infer_yolo (e.g. "Yolo_Players", metadata_key: "yolo_players")
    | (frame + "yolo_players" JSON metadata)
    v
player_tracker (metadata_key: "yolo_players", target_labels: ["Player", "Ref"])
    |-- parse JSON, split: Player/Ref detections vs other (Hoop, etc.)
    |-- convert xyxy -> tlwh Object structs (carry source detection index)
    |-- ByteTrack::update(objects) -> tracked STracks with IDs
    |   (called every frame -- empty vector if no detections present)
    |-- map STracks back to source detections via stored detection index
    |-- annotate: track_id, tracklet_len, track_state, velocity, predicted_xyxy
    |-- if predict_on_empty and no detections: emit predicted bboxes marked "predicted": true
    |-- merge: annotated + passthrough detections
    |-- write updated JSON to frame metadata
    | (frame + enriched "yolo_players" JSON)
    v
join_metadata -> downstream (draw_bbox, basketball_analysis, etc.)
```

**Important:** `tracker_->update()` is called every frame regardless of whether detections are present. This keeps ByteTrack's internal `frame_id` in sync with real FPS, so `max_time_lost` always counts real frames.

## Output Format

Each tracked detection gets these fields added to its JSON object:

```json
{
  "cls": 3,
  "label": "Player",
  "conf": 0.92,
  "xyxy": [120, 50, 200, 300],
  "model_index": 0,
  "track_id": 7,
  "tracklet_len": 45,
  "track_state": "tracked",
  "velocity": [1.2, -0.5],
  "predicted_xyxy": [121.2, 49.5, 201.2, 299.5]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `track_id` | int | Unique persistent ID per player. -1 if detection below tracking threshold |
| `tracklet_len` | int | Number of frames this track has existed |
| `track_state` | string | "tracked" for matched detections; "lost" only for emitted lost/predicted boxes |
| `velocity` | [float, float] | Kalman-estimated [dx, dy] per frame |
| `predicted_xyxy` | [float x4] | Kalman-predicted bounding box (smoothed position) |
| `predicted` | bool | true if this bbox was emitted from prediction (no detection on this frame) |

Non-target detections (ball, hoop, etc.) pass through unchanged with no tracking fields added.

## Node Parameters

```
metadata_key      : string  = "yolo_players"     -- metadata key to read/write
target_labels     : array   = ["Player"]          -- labels to track
target_class      : int     = -1                  -- match by class ID (-1 = unused)
label_case_sensitive : bool = false               -- false = compare labels case-insensitively
min_conf          : float   = 0.01               -- pre-filter before ByteTrack
frame_rate        : int     = 30                  -- input FPS (must match pipeline)
track_buffer      : int     = 30                  -- frames to keep lost tracks
track_thresh      : float   = 0.5                -- min confidence for ByteTrack to use detection
high_thresh       : float   = 0.6                -- min confidence to start new track
match_thresh      : float   = 0.8                -- IoU threshold for first association
low_match_thresh  : float   = 0.5                -- IoU threshold for second (low-conf) association
predict_on_empty  : bool    = true               -- when target detections are empty, emit predicted boxes from active/lost tracks
emit_lost_tracks  : bool    = false              -- include tracker lost-state boxes in output when available
```

## File Layout

```
deps/bytetrack/
  include/
    BYTETracker.h        -- Object struct (no OpenCV), tracker API
    STrack.h             -- track state (no OpenCV, using namespace std inside namespace bytetrack only)
    kalmanFilter.h       -- 8D Kalman filter
    lapjv.h              -- Hungarian algorithm (LAPJV_ prefixed macros)
    dataType.h           -- Eigen typedefs
  src/
    BYTETracker.cpp      -- tracker logic, per-instance ID counter, low_match_thresh parameter
    STrack.cpp           -- track lifecycle
    kalmanFilter.cpp     -- Kalman predict/update (throw not exit)
    lapjv.cpp            -- Hungarian algorithm
    utils.cpp            -- IoU, distance matrices (throw not exit)

deps/include/bytetrack -> ../bytetrack/include   (symlink)

src/nodes/neural_net/sport_specific/player_tracker.cpp  (avplumber node)
```

## ByteTrack Modifications

1. **Remove OpenCV**: replace `cv::Rect_<float>` in Object with plain floats `{x, y, w, h}`, remove `#include <opencv2/opencv.hpp>`, remove `using namespace cv`, remove `get_color()`
2. **Namespace**: wrap all code in `namespace bytetrack { }`. Keep `using namespace std;` inside the namespace scope (avoids ~50 mechanical edits, no pollution outside `bytetrack::`)
3. **Per-instance ID counter**: move `STrack::next_id()` static counter to `BYTETracker` member. Pass by reference to `activate()`/`re_activate()`. Resets when tracker is recreated on stream seek.
4. **Error handling**: replace `exit(0)`/`system("pause")` with `throw std::runtime_error`
5. **Macro safety**: prefix `lapjv.h` macros (`NEW`, `FREE`, `TRUE`, `FALSE`) with `LAPJV_`
6. **Stable association handoff**: extend `Object`/`STrack` with source detection index so node output mapping uses tracker-assigned identity directly, not post-hoc IoU rematching.
7. **Configurable second-pass threshold**: replace hardcoded `0.5` low-confidence match threshold with `low_match_thresh`.

## Dependencies

- **Eigen** (system): `eigen3-devel` package. Required by ByteTrack Kalman filter. Header-only. Must be installed on build machines. Install on remote: `sudo dnf install eigen3-devel`
- No OpenCV dependency after modifications.
- Gated behind `NEURAL_NET_SPECIFIC=1` build flag -- no impact on builds without it.

## Makefile Integration

```makefile
# Under the existing NEURAL_NET_SPECIFIC block (line 44-46):
ifeq ($(NEURAL_NET_SPECIFIC),1)
NODES_SRC += $(shell find $(SRCDIR)/nodes/neural_net/sport_specific -maxdepth 1 -name '*.cpp')
BYTETRACK_SRC = $(wildcard deps/bytetrack/src/*.cpp)
override CXXFLAGS += -I/usr/include/eigen3
endif

# Line 186 -- append to CPPSRC_LIB:
CPPSRC_LIB = $(addprefix src/,$(CPPSRC)) $(nodes_list_file) $(NODES_SRC) $(BYTETRACK_SRC)
```

Note: `generate_node_list` only scans `NODES_SRC` (maxdepth 1 in sport_specific/), so ByteTrack .cpp files in deps/ are not picked up for node registration.

## Node Implementation

```
class PlayerTracker : public NodeSISO<av::VideoFrame, av::VideoFrame>, public IInputReset
DECLNODE(player_tracker, PlayerTracker)
```

### process() pseudocode

```
1. frame = source_->get()
2. if EOF: resetState(), propagate, return

3. parse metadata_key if present:
     a. if present and valid: split detections into target_dets + passthrough_dets
     b. if absent/invalid: treat as target_dets empty, passthrough_dets empty
     c. apply min_conf to target_dets

4. if target_dets is non-empty:
     a. convert target_dets xyxy -> tlwh Object structs, storing source index
     b. output_tracks = tracker_->update(objects)
     c. map each output STrack directly to source detection index
     d. annotate matched target detections with:
          - track_id, tracklet_len, track_state="tracked"
          - velocity from Kalman state (mean[4], mean[5])
          - predicted_xyxy from STrack.tlbr
     e. unmatched target detections get track_id = -1
     f. optionally append lost tracks when emit_lost_tracks=true
     g. write metadata = passthrough_dets + annotated target_dets (+ optional lost tracks)

5. if target_dets is empty:
     a. tracker_->update({})  -- always call to keep frame_id in sync
     b. if predict_on_empty=true:
          - emit predicted boxes from tracker state with predicted=true, track_state="lost"
          - write metadata = passthrough_dets + predicted boxes
     c. if predict_on_empty=false:
          - if metadata existed: preserve/write passthrough_dets only
          - if metadata did not exist: pass frame unchanged

6. emit frame
```

### IInputReset

On `resetInput()`: destroy and recreate `BYTETracker`. Clears all tracks and resets the per-instance ID counter.

## Example Pipeline

`examples/yolo/yolo_infer_all_players_tracker.avplumber` — copy of `yolo_infer_all_players.avplumber` with `player_tracker` inserted after `Yolo_Players`, before the first `join_metadata`:

```
Yolo_Players (metadata_key: "yolo_players")
    -> player_tracker (metadata_key: "yolo_players", target_labels: ["Player", "Ref"])
    -> join_metadata (merge with ball detections)
    -> ...rest of pipeline unchanged...
```

## Files Changed Summary

| Path | Action |
|------|--------|
| `deps/bytetrack/include/` (5 headers) | New — copied + modified from ByteTrack upstream |
| `deps/bytetrack/src/` (5 sources) | New — copied + modified from ByteTrack upstream |
| `deps/include/bytetrack` | New symlink -> `../bytetrack/include` |
| `src/nodes/neural_net/sport_specific/player_tracker.cpp` | New — avplumber node |
| `examples/yolo/yolo_infer_all_players_tracker.avplumber` | New — example pipeline |
| `Makefile` | Modified — 2 new lines + 1 modified line |
| `doc/specs/2026-04-09-player-tracker-design.md` | New — this spec |

## Follow-up Tasks (not in this spec)

- **draw_bbox track ID rendering**: display numeric track_id in top-left corner of each bbox

## Verification

- Build: `make -j$(nproc) NEURAL_NET_COMMON=1 NEURAL_NET_SPECIFIC=1 HAVE_CUDA=1`
- Check `player_tracker` appears in `graph_factory.generated.cpp`
- Test with example: `./avplumber -s examples/yolo/yolo_infer_all_players_tracker.avplumber`
- Verify: track_ids are stable across short occlusions and remain unique per active player
- Verify: IDs reset after input reset/seek (`resetInput()`)
- Verify: with `infer_every_n > 1`, `predict_on_empty=true` keeps boxes continuous between inference frames
- Verify: with `predict_on_empty=false`, frames with no target detections do not emit synthetic tracked boxes
- Verify: non-target detections are byte-for-byte preserved except for ordering changes documented by implementation
