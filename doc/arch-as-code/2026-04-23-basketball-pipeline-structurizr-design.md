# Basketball Pipeline Structurizr Design

Date: 2026-04-23

## Goal

Represent `examples/yolo/yolo_infer_all_players_tracker_live_teams_siglip_nba.avplumber`
as architecture-as-code using Structurizr JSON.

The output should document the graph at two levels across three views:

- High level: one container view for the end-to-end runtime.
- Low level: two component views.
  - Internal-only view.
  - External-aware view including runtime dependencies and endpoints.

The generated files should live under `docs/arch-as-code/`.
No automatic git commit should be made.

## Source Graph

The source graph is a single avplumber runtime pipeline that:

- ingests recorded video,
- demuxes and decodes on CUDA,
- fans frames out into multiple inference branches,
- derives metadata for players, ball, court, teams, and scoreboard,
- renders overlays and reframing guidance,
- exports metadata,
- encodes and publishes annotated video.

## Modeling Approach

The Structurizr model will use C4-style abstractions rather than a literal one-node-per-line mapping.

- `softwareSystem`: the full basketball analysis pipeline.
- `container`: the avplumber runtime process executing the graph.
- `components`: stable processing responsibilities inside that runtime.
- `external systems`: file/video source, CUDA acceleration, TensorRT runtime, model engines, metadata file, and RTMP endpoint.

This keeps the model readable and useful as architecture documentation instead of turning it into a wiring dump.

## Views

### High-Level Container View

The high-level view shows:

- the `AVPlumber Basketball Analysis Pipeline` software system,
- the `AVPlumber Runtime` container,
- the external input source,
- the metadata output file,
- the RTMP publish endpoint.

This view answers what the pipeline is, what runs it, and what it interacts with.

### Low-Level Internal Component View

The internal-only component view shows the major responsibilities inside the runtime:

- `Ingest`
- `Decode And Timing`
- `Frame Fanout`
- `Detection Inference`
- `Segmentation Inference`
- `Scene And Event Understanding`
- `Tracking And Ball Possession`
- `Team Classification`
- `Scoreboard OCR`
- `Visualization And Reframing`
- `Metadata Export`
- `Encoding And Delivery`

This view excludes external dependencies so that the internal processing flow is easy to read.

### Low-Level External-Aware Component View

The external-aware component view reuses the same internal components and adds:

- `Input Video Source`
- `CUDA Video Acceleration`
- `TensorRT Inference Runtime`
- `Model Engine Set`
- `Metadata Dump File`
- `RTMP Publish Endpoint`

This view answers both processing flow and dependency shape.

## Relationship Model

The main internal component relationships are:

1. `Ingest -> Decode And Timing`
2. `Decode And Timing -> Frame Fanout`
3. `Frame Fanout -> Detection Inference`
4. `Frame Fanout -> Segmentation Inference`
5. `Detection Inference -> Scene And Event Understanding`
6. `Segmentation Inference -> Scene And Event Understanding`
7. `Scene And Event Understanding -> Tracking And Ball Possession`
8. `Tracking And Ball Possession -> Team Classification`
9. `Segmentation Inference -> Team Classification`
10. `Decode And Timing -> Scoreboard OCR`
11. `Team Classification -> Scoreboard OCR`
12. `Scoreboard OCR -> Visualization And Reframing`
13. `Team Classification -> Visualization And Reframing`
14. `Visualization And Reframing -> Metadata Export`
15. `Visualization And Reframing -> Encoding And Delivery`

The external-aware relationships add:

- `Input Video Source -> Ingest`
- `Decode And Timing -> CUDA Video Acceleration`
- `Encoding And Delivery -> CUDA Video Acceleration`
- `Detection Inference -> TensorRT Inference Runtime`
- `Segmentation Inference -> TensorRT Inference Runtime`
- `Team Classification -> TensorRT Inference Runtime`
- `Scoreboard OCR -> TensorRT Inference Runtime`
- `TensorRT Inference Runtime -> Model Engine Set`
- `Metadata Export -> Metadata Dump File`
- `Encoding And Delivery -> RTMP Publish Endpoint`

## Boundary Decisions

The model deliberately does not represent:

- individual `split` and `join_metadata` nodes,
- queue capacities,
- exact metadata key names,
- per-node thresholds and tuning values,
- each draw node as its own component,
- every engine file as a separate component.

These details belong to the graph and runtime configuration, not the architecture view.

## File Plan

Planned generated files:

- `docs/arch-as-code/2026-04-23-basketball-pipeline-structurizr-design.md`
- `docs/arch-as-code/basketball-pipeline-structurizr.json`

## Success Criteria

The Structurizr JSON is correct if:

- it loads as a single workspace,
- it contains one software system and one runtime container,
- it contains one high-level container view,
- it contains one internal-only component view,
- it contains one external-aware component view,
- the component breakdown reflects the approved architecture boundaries,
- the external-aware view includes runtime dependencies without duplicating the internal model.
