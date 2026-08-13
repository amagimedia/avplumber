# Video Mixer

AVPlumber's mixer is a two-slot program/preview video switcher. The reusable
graph builder is `pyplumber/mixer.py`; the native control implementation is in
`src/mixer_orchestrator.cpp`; the maintained example is `demos/mixer/`.

The mixer carries video frames only. It has no audio routing, VAD, speaker
selection, face tracking, or camera policy.

## Graph

Each source supplies one CUDA video-frame edge. A source can either fan out to
the two mixer slots through `one_to_many`, or use `preheat_video_router` outputs
when its geometry paths are kept hot.

```text
CUDA source frames
  -> preheat_video_router
  -> scale_cuda + pad_cuda geometry for slot A and slot B
  -> cuda_rect_overlay A / cuda_rect_overlay B
  -> permanent transition_cuda
  -> source_switcher
  -> NVENC
```

Each slot has its own geometry routes because an outgoing scene and incoming
scene must remain live at the same time during a transition. Scene changes are
scheduled against a shared timeline so router selection, compositor inputs,
and the program selector change at consistent frame timestamps.

`MixerGraphBuilder` returns the final video-frame edge. The application owns
input decode, output encoding/muxing, startup order, and shutdown.

## Preheating

Applications that need immediate transitions must preheat the complete path:

1. Start and pace all input groups.
2. Start the preheat router with temporary valid routes.
3. Start both copies of every geometry path and wait for frames on their output
   edges.
4. Publish the initial scene routes.
5. Start both compositors and the mixer group.
6. Temporarily feed both compositor outputs to `transition_cuda`, wait for a
   transition output frame, then restore steady routing.
7. Start the encoder/output group and declare the graph ready.

There is no useful cold fallback for a low-latency production graph. The
generic demo fails startup when any required preheat stage times out.

## Transitions

The control protocol accepts JSON objects:

```text
mixer.preview {"mixer":"mixer","scene":"grid_4_page_0"}
mixer.cut {"mixer":"mixer","scene":"grid_4_page_0"}
mixer.fade {"mixer":"mixer","scene":"grid_4_page_0","duration_sec":0.5}
mixer.cuda_wipe {"mixer":"mixer","scene":"grid_4_page_0","style":"wipe_left","duration_sec":0.5}
```

Supported procedural wipe styles are `wipe_left`, `wipe_right`, `wipe_down`,
and `wipe_up`. Fades and procedural wipes share one permanent CUDA transition
filter. Runtime filter commands change its expression and mode; no transition
node is created or initialized during a take.

`mixer.status <name>` returns the current PGM/PVW scene and transition state.
`mixer.scenes <name>` lists registered scenes.

The builder also retains an optional media-file wipe compatibility path. That
path converts an alpha-bearing wipe asset in software before uploading it and
is therefore not part of the zero-copy generic demo.

## Zero-copy contract

The generic demo's production video-frame path stays in CUDA memory from
hardware decode through normalization, routing, geometry, composition,
transition, and NVENC. It must not contain `hwdownload`, `hwupload`, or
`hwupload_cuda` filters. Packet edges before decode and after encode, RTP/RTCP,
muxing, and control messages are not video-frame memory paths.

The demo uses these CUDA operations:

- `scale_cuda` and `pad_cuda` for normalization and layout geometry;
- `cuda_rect_overlay` for scene composition;
- `transition_cuda` for fades and procedural wipes;
- NVDEC and NVENC at the graph boundaries.

## Generic demo

`demos/mixer/mixer.py` accepts a repeated `--input` option. File paths and URLs
are runtime configuration and are never embedded in the repository. File
inputs are paced in realtime and can be looped with `--loop-inputs`.

The fixed portrait canvas is 1080x1920. The scene set contains one fullscreen
scene per input plus paged 2-, 4-, 8-, and 16-box layouts. Manual control is
available through `demos/mixer/tui.py`; output can be a video-only recording or
a video-only Janus RTP mountpoint.
