# Video Mixer

AVPlumber's mixer is a two-slot program/preview video switcher. The reusable
graph builder is `avpmixer/graph.py`; the native control implementation is in
`src/mixer/mixer_orchestrator.cpp`; the maintained example is `demos/mixer/`.

The mixer carries video frames only. It has no audio routing, VAD, speaker
selection, face tracking, or camera policy.

## Graph

Each source supplies one CUDA video-frame edge. A source can either fan out to
the two mixer slots through `one_to_many`, or use `preheat_video_router` outputs
for a catalogue that exceeds the compositor input limit.

```text
CUDA source frames
  -> source fanout (or catalogue router)
  -> cuda_rect_overlay A / cuda_rect_overlay B
  -> permanent transition_cuda
  -> source_switcher
  -> NVENC
```

Each slot has its own layer rectangles because an outgoing scene and incoming
scene must remain live at the same time during a transition. Scene changes are
scheduled against a shared timeline so router selection, compositor inputs,
and the program selector change at consistent frame timestamps.

`MixerGraphBuilder` returns the final video-frame edge. The application owns
input decode, output encoding/muxing, startup order, and shutdown.

## Preheating

Applications that need immediate transitions must preheat the complete path:

1. Start and pace all input groups; wait for input frames.
2. Start the catalogue router if used, then publish initial scene routes.
3. Start both compositors with the scaling kernel loaded and fixed output pools.
4. Temporarily feed both compositor outputs to `transition_cuda`, wait for a
   transition output frame, then restore steady routing.
5. Open the output gate for fresh frames, start the encoder/output group and
   declare the graph ready.

Sources registered with `default_graph=""` feed the compositors directly.
Their scene layers use `dst_x`, `dst_y`, `dst_w`, `dst_h`, optional `crop`, and
`fit` (`contain` or `stretch`). Dimensions and pitch are resolved from each
frame. Optional `source_canvas: {"w": 1920, "h": 1080}` preserves letterboxing
into that virtual source canvas without an intermediate GPU frame. Explicit
FFmpeg preprocessing graphs remain available to existing callers.

There is no useful cold fallback for a low-latency production graph. The
generic demo fails startup when any required preheat stage times out.

## Transitions

The control protocol accepts JSON objects:

```text
mixer.preview {"mixer":"mixer","scene":"grid_4_page_0"}
mixer.cut {"mixer":"mixer","scene":"grid_4_page_0"}
mixer.fade {"mixer":"mixer","scene":"grid_4_page_0","duration_sec":0.5}
mixer.wipe {"mixer":"mixer","scene":"grid_4_page_0","wipe_file":"<path>/wipe.mov"}
```

Cut, Fade and transparent media-file Wipe are supported. Fade uses the permanent
CUDA transition filter. The media wipe graph is predeclared; the orchestrator
loads the selected clip. A new take can interrupt an ongoing transition using
the current output picture.

`mixer.status <name>` returns the current PGM/PVW scene and transition state.
`mixer.scenes <name>` lists registered scenes.

The alpha media path decodes the wipe in software and uploads it to the mixer
CUDA device. This is separate from the GPU-native program video path.

## Zero-copy contract

The generic demo's production video-frame path stays in CUDA memory from
hardware decode through normalization, routing, geometry, composition,
transition, and NVENC. It must not contain `hwdownload`, `hwupload`, or
`hwupload_cuda` filters. Packet edges before decode and after encode, RTP/RTCP,
muxing, and control messages are not video-frame memory paths.

The demo uses these CUDA operations:

- optional `scale_cuda` and `pad_cuda` for homogeneous catalogue inputs;
- `cuda_rect_overlay` for per-layer scaling and scene composition;
- `transition_cuda` for fades;
- NVDEC and NVENC at the graph boundaries.

## Generic demo

`demos/mixer/mixer.py` accepts a repeated `--input` option. File paths and URLs
are runtime configuration and are never embedded in the repository. File
inputs are paced in realtime and can be looped with `--loop-inputs`.

The fixed portrait canvas is 1080x1920. The scene set contains one fullscreen
scene per input plus paged 2-, 4-, 8-, and 16-box layouts. Manual control is
available through `demos/mixer/tui.py`; output can be a video-only recording or
a video-only Janus RTP mountpoint.
