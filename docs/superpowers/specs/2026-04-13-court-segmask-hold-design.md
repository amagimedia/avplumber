# Court Segmentation Mask Hold Design

## Summary

The court segmentation overlay flickers because `draw_segmask` is currently stateless. When `cuda_infer_yolo` produces no segmentation detections on a frame, it emits no GPU mask side data, and `draw_segmask` immediately skips drawing. The next valid segmentation frame restores the overlay, causing visible flicker.

This design adds a small, local hold mechanism to `draw_segmask` so it can reuse the last valid court segmentation mask for a limited number of frames.

## Goals

- Eliminate short court-mask dropouts in the rendered overlay.
- Keep the change local to `draw_segmask`.
- Preserve existing inference, metadata, and graph behavior.
- Make the hold window configurable, with a default of `6` frames.

## Non-Goals

- No change to `cuda_infer_yolo` output semantics.
- No change to `join_metadata`.
- No persistence of stale segmentation beyond a bounded window.
- No attempt to smooth or blend masks over time.

## Proposed Approach

Add a new `draw_segmask` parameter:

- `hold_last_frames`: integer, default `6`

`draw_segmask` will maintain a cache of the last valid drawable segmentation result:

- GPU mask buffer reference (`AVBufferRef`)
- `GpuMaskSideDataHeader`
- parsed detection list used for drawing

### Valid Update

A frame is considered a valid segmentation update only if:

- `AV_FRAME_DATA_YOLO_SEG_MASKS_GPU` exists
- the side-data buffer is large enough for `GpuMaskSideDataHeader`
- the header contains a non-null GPU pointer and positive mask dimensions
- parsing metadata produces at least one drawable detection

When that happens, `draw_segmask` refreshes the cache and resets a miss counter to `0`.

### Missing Or Empty Update

If the current frame has:

- no GPU segmentation side data, or
- invalid GPU segmentation side data, or
- zero drawable segmentation detections

then `draw_segmask` will not clear the cache immediately. Instead, it will keep drawing the cached segmentation result while the number of consecutive misses is less than `hold_last_frames`.

This includes explicit segmentation frames where YOLO returns `0` detections. Those frames are treated as "no new usable update", not as a cache reset.

### Cache Expiry

Once the consecutive miss count reaches `hold_last_frames`, `draw_segmask` stops drawing the cached mask and waits for a new valid segmentation update to repopulate the cache.

## Data Flow

1. `draw_segmask` receives a CUDA frame.
2. It tries to read current-frame segmentation side data and segmentation metadata.
3. If the current frame contains a valid drawable segmentation result, the cache is refreshed and drawn.
4. Otherwise, if cached data exists and is still within the configured hold window, cached data is drawn instead.
5. Otherwise, the frame passes through without segmentation overlay.

## Error Handling

- Invalid or partial side data is treated as a miss, not as a fatal error.
- Kernel launch failures remain unchanged and still abort drawing for that frame.
- Cache state is owned entirely by `draw_segmask` and cleaned up by normal buffer unref on node destruction or cache replacement.

## Testing

### Functional

- Run the neural demo with the court segmentation overlay enabled.
- Verify that short segmentation gaps no longer make the court mask disappear immediately.
- Verify that long runs of missing segmentation eventually remove the overlay after `hold_last_frames`.

### Configuration

- Verify default behavior with no explicit `hold_last_frames` parameter.
- Verify `hold_last_frames: 0` disables the hold behavior.
- Verify a custom value such as `hold_last_frames: 12` extends retention.

### Regression

- Verify no changes to `shot_classifier`, `ball_tracker`, or other consumers of segmentation metadata.
- Verify the live and VOD pipelines still build and run without template changes.
