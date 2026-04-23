# SigLIP Team Classifier Design

## Goal

Replace the current histogram-based team classifier with a crop-embedding pipeline that is robust to skin tone, lighting shifts, and white/black jersey combinations.

The first implementation targets the existing live basketball pipeline and must remain usable in realtime on the remote TensorRT box.

## Problem Summary

The current histogram path fails for two structural reasons:

- it depends on hand-tuned color extraction from a guessed torso region
- it tries to separate teams in low-level color space, where skin, lighting, and court reflections interfere with the signal

The live pipeline also showed that player segmentation masks are often visually more reliable than the torso sampling heuristic. That shifts the design toward embedding masked player crops rather than extracting handcrafted jersey histograms.

## Scope

In scope:

- a TensorRT-based SigLIP vision embedding path for player crops
- a new team classifier that clusters and smooths those embeddings
- integration with existing `yolo_players`, `yolo_players_seg`, tracker, and debug overlay flow

Out of scope for v1:

- training or fine-tuning a custom model
- explicit team-name recognition
- generic reusable embedding / manifold-learning infrastructure
- online UMAP in the hot path

## Design Summary

The replacement consists of two new nodes:

1. `player_mask_embed`
2. `team_embed_classifier`

`player_mask_embed` uses tracked player detections plus matched player segmentation masks to produce masked full-player crops, then runs a SigLIP vision encoder in TensorRT and writes the embedding back onto the corresponding `yolo_players` detection.

`team_embed_classifier` consumes those per-player embeddings, performs warmup-time clustering into two teams, smooths assignments over time, and writes team IDs back to both `yolo_players` and `yolo_players_seg`.

## Why This Shape

This design keeps the expensive vision preprocessing and inference separate from the stateful classification logic.

That separation gives three benefits:

- bad crops can be debugged independently from bad clustering
- the classifier can evolve without touching TensorRT preprocessing
- the graph stays simple and close to the current `team_classifier` integration

The rejected alternatives are:

- one monolithic `team_classifier_siglip` node: faster to ship, much harder to debug
- a generic crop -> embedding -> reducer -> cluster graph: cleaner in theory, too much new machinery for the current problem

## Model Choice

Use pretrained SigLIP vision weights. No custom training dataset is required for v1.

Chosen checkpoint for v1:

- `google/siglip-base-patch16-224`

Model reference:

- `https://huggingface.co/google/siglip-base-patch16-224`

The first version uses only the vision encoder. It does not depend on text prompts or zero-shot text matching.

The initial runtime path is:

- fixed spatial input size: `224x224`
- dynamic batch size: support at least `1..16`
- TensorRT engine, not ONNX Runtime

Model artifacts should live on the remote host under `/home/fedora/tensorrt/` using the same convention as the existing model inventory. The implementation should keep both:

- the exported `.onnx`
- the built TensorRT `.plan`

under a dedicated SigLIP model subdirectory or clearly named files in `/home/fedora/tensorrt/`.

The chosen SigLIP model and resulting artifact paths must also be added to `doc/models.md`.

Reasoning:

- current live detections in `1080p` map to roughly `105 x 230` full-player boxes on average
- the derived masked full-player crop contains enough information for `224x224`
- larger inputs are likely wasted compute relative to current source detail

## Crop Strategy

The embedding crop is based on the full player segmentation mask, not a hand-authored torso heuristic.

For each matched tracked player:

1. find the matched player segmentation instance
2. compute the tight bounding box of the visible mask
3. extract that crop from the frame
4. zero-fill pixels outside the mask within the crop
5. pad the crop to square with black fill
6. resize to `224x224`
7. normalize as required by the SigLIP vision encoder

This keeps precise player pixels while removing most court and background contamination.

Black fill is chosen for v1 because it is deterministic, simple, and easy to reproduce on GPU.

## Node 1: `player_mask_embed`

### Inputs

- video frame
- player detection metadata key, default `yolo_players`
- player segmentation metadata key, default `yolo_players_seg`
- player segmentation GPU side-data slot

### Responsibilities

- parse tracked player detections
- parse player segmentation detections and masks
- match tracked players to segmentation instances
- build masked crops for matched players
- batch crops for TensorRT inference
- run SigLIP vision encoder
- attach embedding vectors and embedding quality metadata to the corresponding player detections

### Output

The node passes through the input frame and rewrites the `yolo_players` metadata in place. Each matched player detection gets new fields such as:

- `embed_model`
- `embed_dim`
- `embed_valid`
- `embed_conf` or similar quality score
- `embed_vec` or compact serialized embedding representation

The exact field names should be short and specific to avoid inflating metadata unnecessarily.

### Matching

Matching should follow the same general approach as the current team path:

- IoU-first matching between tracked player bbox and seg bbox
- fallback center-distance threshold when IoU is weak

Only one seg instance may be assigned to a given track in a frame.

### Failure Behavior

If no valid seg match exists for a track on a frame:

- do not emit a fresh embedding for that track
- preserve prior classifier state downstream
- expose debug counters for unmatched tracks

If the crop is too small or the mask is degenerate:

- mark `embed_valid=0`
- skip TensorRT inference for that instance

### Parameters

The first version should expose only the controls that matter operationally:

- `player_metadata_key`
- `seg_metadata_key`
- `seg_side_data_slot`
- `player_labels`
- `seg_labels`
- `iou_match_threshold`
- `fallback_center_distance_px`
- `min_mask_pixels`
- `min_crop_width`
- `min_crop_height`
- `input_size`
- `engine`
- `max_batch`
- `debug_log_every_n`

## Node 2: `team_embed_classifier`

### Inputs

- video frame
- player metadata key with embeddings already attached
- player segmentation metadata key for write-back
- shot metadata key

### Responsibilities

- collect valid embeddings during warmup
- fit two team prototypes from the current game
- assign team IDs from embedding distance
- smooth assignments per track over time
- hand off team identity across short detection gaps
- write team labels back to both tracked players and segmentation detections

### Clustering Strategy

v1 will not use UMAP.

The classifier path is:

1. gather valid embeddings from warmup frames
2. L2-normalize embeddings
3. cluster into two groups using k-means or equivalent centroid fitting
4. store two frozen team prototypes
5. classify later embeddings by nearest prototype

This is simpler and more stable for a streaming system than fitting UMAP online.

If needed later, PCA can be added before clustering without changing crop generation.

### Temporal Smoothing

The classifier should behave like a track-state machine, not a frame-by-frame relabeler.

Per track, maintain:

- recent embedding EMA or rolling average
- current team ID
- assignment confidence / margin
- hit count
- age since last strong evidence

Assignment should require a margin over the competing prototype and should not flip on weak single-frame evidence.

### Warmup

Warmup should only use wide-shot frames when shot metadata says the frame is suitable. This matches the current overlay and tracker assumptions and reduces close-up bias.

Warmup conditions:

- enough distinct tracks observed
- enough valid embeddings
- minimum prototype separation

If warmup fails, the node remains in an unbootstrapped state instead of forcing a split.

### Write-Back

Write the final team label back to:

- `yolo_players`
- `yolo_players_seg`

This preserves compatibility with:

- `draw_segmask`
- player label overlays
- any downstream logic already reading team fields from player detections

### Parameters

Initial parameter set:

- `player_metadata_key`
- `seg_metadata_key`
- `shot_metadata_key`
- `player_labels`
- `seg_labels`
- `bootstrap_frames`
- `bootstrap_min_tracks`
- `bootstrap_min_embeddings`
- `bootstrap_min_prototype_distance`
- `ema_alpha_track`
- `ema_alpha_centroid`
- `assignment_margin`
- `soft_assignment_margin`
- `track_idle_frames`
- `handoff_max_age_frames`
- `handoff_max_center_distance_rel`
- `write_back_to_seg`
- `rewrite_seg_cls`
- `rewrite_label`
- `debug_log_every_n`

## Metadata Contract

The graph should continue treating `yolo_players` as the canonical tracked-player metadata stream.

`player_mask_embed` augments `yolo_players` with embeddings. `team_embed_classifier` consumes the augmented detections and adds team fields to the same metadata object.

This avoids introducing another metadata stream and keeps existing nodes unchanged wherever possible.

## Graph Changes

Add a new example graph for the SigLIP path instead of overwriting the current histogram example.

Create:

- `examples/yolo/yolo_infer_all_players_tracker_live_teams_siglip.avplumber`

That new example should reuse the existing live pipeline structure but replace the histogram path with:

1. `Yolo_Player_Seg`
2. `player_mask_embed`
3. `team_embed_classifier`

and remove from the SigLIP example:

- `jersey_color_extract`
- histogram-specific team-classifier assumptions

The rest of the live graph should remain intact, especially:

- `player_tracker`
- `shot_classifier`
- `draw_segmask`
- `draw_bbox_labels`

## Realtime Expectations

This path is expected to be realtime-capable on the remote TensorRT machine if it follows these constraints:

- batch all visible player crops per frame
- keep the model at `224x224`
- support dynamic batch instead of dynamic spatial resolution
- allow inference every frame initially, with room to decimate later if needed

The main performance risk is crop batching and GPU memory movement, not clustering.

## Error Handling And Debuggability

Both nodes must emit compact periodic debug logs that make failures obvious.

`player_mask_embed` should log:

- tracked players
- seg instances
- matched instances
- valid crops
- skipped crops
- TensorRT batch size
- average crop dimensions before resize

`team_embed_classifier` should log:

- bootstrapped state
- valid embeddings seen
- active tracks
- assignment counts per team
- ambiguous / weak assignments
- prototype separation

For debugging visual quality, a follow-up optional node can later dump or draw the exact masked crops being embedded. That is not required for v1 implementation.

## Testing

Implementation is complete only when all of the following are checked:

1. local compile passes for touched translation units where possible
2. remote x86/CUDA build passes
3. the SigLIP `.onnx` and `.plan` are present under `/home/fedora/tensorrt/`
4. `doc/models.md` is updated with the SigLIP model entry and paths
5. the live RTMP SigLIP example runs without TensorRT shape or context errors
6. debug logs show stable embedding production on most visible players
7. team overlay shows a meaningful split after warmup on representative footage

## Risks

- segmentation may still miss some players, which reduces embedding coverage
- a full-player crop may include shorts and shoes, which can dilute jersey signal
- embedding metadata may become bulky if serialized poorly
- TensorRT export details for SigLIP may require model-specific preprocessing care

## Follow-Up Work

If v1 is still not robust enough, the next steps should be:

1. add a debug crop-dump path to inspect actual embedding inputs
2. test a mild upper-body weighting within the masked crop instead of hard torso slicing
3. add PCA before clustering
4. only then consider UMAP

UMAP is explicitly deferred because it adds complexity to fit/transform handling without being necessary for the first proof of value.
