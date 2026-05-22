# Auto Mixer Architecture

The auto mixer is the Python application layer that builds and drives an avplumber
media graph for talk-show style switching. It is separate from the C++
`MixerOrchestrator`: the Python code constructs the input, analysis, mixer, output,
and policy graph; the C++ mixer core performs PTS-aligned scene switching after it
receives `mixer.*` control commands.

Related docs:

- `doc/mixer_orchestrator.md` describes the C++ two-slot mixer, timeline semantics,
  cuts, fades, and wipes.
- `doc/specs/2026-03-24-python-bindings-design.md` records the original Python
  binding design.
- `pyplumber/examples/README.md` gives smaller pyplumber binding examples.

## Main Entry Points

- `pyplumber/examples/auto_mixer.py` is a compatibility wrapper.
- `pyplumber/auto_mixer/cli.py` is the real command-line entry point. It parses
  runtime options, creates `AVPlumber`, builds all graph sections, starts groups,
  registers with the Web UI, and starts the auto-switcher policy.
- `pyplumber/auto_mixer/runtime.py` owns process lifetime: group start ordering,
  status logging, signal handling, and shutdown of policy/control helpers.

The normal startup sequence is:

1. Create `AutoMixerAVPlumber`, enable logging/control/Web UI registration, initialize CUDA.
2. Build one input subgraph per URL.
3. Build the `MixerGraphBuilder`, register scenes, and materialize mixer nodes.
4. Add optional HTML overlay, recording output, and Janus RTP output.
5. Add Python registry bridge nodes for audio and visual speech.
6. Start input groups, optional preheat groups, mixer groups, and output group.
7. Mark the control server ready and start `AutoSwitcher`.

## Python Bindings Model

`pyplumber.core.AVPlumber` subclasses the pybind-exposed `_avplumber.AVPlumber`.
Most graph construction is Python code that instantiates lightweight wrapper classes
from `pyplumber/node.py` and passes their parameter dictionaries to the native
`NodeManager`.

There are two categories of Python-side node classes:

- Native node wrappers, such as `InputRec`, `Demux`, `DecVideo`, `FilterVideo`,
  `CudaInferYolo`, `OneToMany`, `SourceSwitcher`, `EncVideo`, and `Output`. These
  only describe C++ node parameters; media processing remains native.
- Real Python nodes, subclasses of `PythonNode`. These are attached to native
  `python_node_*` wrappers, receive typed edge handles, and implement `process()`
  in Python.

`PythonNode` selects the native wrapper type from its `src` and `dst` shape:

- `python_node_siso`, `simo`, `miso`, `mimo` for video/audio/metadata transforms.
- `python_node_si` and `mi` for sink-style observers.
- `python_node_so` and `mo` for source-style producers.
- `python_node_audio_to_metadata` for audio samples in and metadata frames out.

When `AVPlumber.addNode()` receives a `PythonNode`, it passes the Python object into
the native wrapper. After the node is added, `_avplumber_initialized()` resolves
typed edge handles with `AVPlumber.getEdge()`. A Python node then calls methods such
as `_src.get()`, `_src.tryGet()`, and `_dst.enqueue()` from its `process()` method.

The auto mixer uses this split deliberately: GPU decode, filtering, inference,
composition, encoding, muxing, and queueing stay in native nodes; Python nodes
mostly inspect metadata/audio, annotate frames, and update control state.

## Top-Level Data Flow

```text
inputs
  -> per-input decode/realtime/fps normalization
  -> face detection + face crop + visual speech metadata
  -> audio VAD metadata
  -> mixer sources/scenes
  -> C++ MixerOrchestrator controlled two-slot mixer
  -> optional HTML overlay
  -> recording output and/or Janus RTP output

Speaker registry
  <- Silero audio VAD bridge
  <- visual speech registry bridge
  -> AutoSwitcher policy
  -> MixerGraphBuilder.cut/fade/wipe
  -> native mixer control commands
```

The media graph is still an avplumber graph of typed edges. The Python policy thread
does not carry media frames; it reads the `Speaker` registry and sends control
commands.

## Per-Input Subgraph

`pyplumber/auto_mixer/inputs.py` builds the per-camera graph. Each input is placed
in `input_<index>` so it can be started/restarted as a unit.

Video path:

1. `input_rec -> demux -> dec_video`.
2. `realtime(set_pts=true)` aligns decoded frames to wallclock PTS.
3. `force_fps` produces the common frame grid expected by joins and the mixer.
4. `split` fans out to full-resolution, YOLO, and visual-speech branches.
5. `filter_video` scales for the TensorRT model, then `cuda_infer_yolo` attaches
   face/part detections as metadata.
6. `player_tracker`, `join_metadata`, and `smooth_crop_viewport` produce stable
   viewport metadata for the tracked face.
7. `crop_metadata_cuda` creates the 9:16 face crop leg.
8. `smooth_timestamps` normalizes the original and face-crop legs before they feed
   mixer sources.

Audio path:

1. `dec_audio -> realtime(set_pts=true) -> resample_audio`.
2. A split keeps one normalized program-audio edge and one VAD edge.
3. The VAD edge is resampled to 16 kHz mono and fed to `SileroVADNode`.

The returned subgraph dictionary contains the edges needed by the mixer and policy
layers: `orig_edge`, `face_edge`, `program_audio_edge`, `vad_events_edge`,
`visual_speech_edge`, `vs_key`, and `input_group`.

## Python Speech Nodes

The auto mixer uses Python nodes where the value is in policy and metadata handling,
not raw pixel movement.

`SileroVADNode` in `pyplumber/vad.py` is an `AudioToMetadataPythonNode`. It reads
audio samples, runs the Silero model, and emits `MetadataFrame` events such as
`speech_start`, `speech_update`, `speech_stop`, and completed `speech` segments.
It also carries media PTS/timebase information so the policy can schedule mixer
commands against media time rather than only wallclock observation time.

`SileroVadRegistryBridge` in `pyplumber/audio_vad.py` is a metadata sink. It consumes
live Silero events and updates the shared `Speaker` registry with audio speaking
state, level, duration, and event PTS.

`FaceAnchoredMouthTrackerNode` in `pyplumber/mouth_tracker.py` is a video Python
node. It reads YOLO face/part metadata and writes mouth ROI metadata. If the model
does not provide a usable mouth box, it can track or estimate the mouth position
inside the selected face.

`VisualSpeechGateNode` in `pyplumber/visual_speech.py` is a video Python node. It
passes frames through while attaching visual-speaking metadata derived from mouth
motion/opening. Its hysteresis lives in the node, so downstream consumers get a
stable speaking/not-speaking state.

`VisualSpeechRegistryNode` in `pyplumber/audio_vad.py` is a video sink. It reads the
visual speech metadata and updates `Speaker.visual_speaking`.

The `Speaker` registry is the crossing point between frame/audio-rate Python nodes
and the slower policy loop. It is thread-safe and stores one `SpeakerEntry` per
input, including audio state, visual state, combined state, PTS, and observed time.

## Mixer Builder Layer

`pyplumber/mixer.py` contains `MixerGraphBuilder`, the Python builder for the native
C++ mixer topology. The builder owns mixer-internal nodes and emits the control
commands that initialize the native `MixerState`.

For each registered source, it creates:

- a source `one_to_many` node, unless the source is routed from a preheat router;
- two per-slot `filter_video` nodes, one feeding slot A and one feeding slot B.

For the mixer core, it creates:

- two `cuda_rect_overlay` compositors;
- per-slot `force_fps` normalizers;
- per-slot post-compositor `one_to_many` nodes;
- the final `source_switcher`;
- optional wipe nodes when wipe support is enabled.

At build time, `MixerGraphBuilder` sends:

- `mixer.init` with timeline, slot, and wipe configuration;
- `mixer.source` or `mixer.routed_source` for each logical source;
- `mixer.scene` for each scene definition.

At runtime, `MixerGraphBuilder.cut()`, `fade()`, `wipe()`, and `preview()` serialize
Python method calls into native control commands. The media transition is then
owned by `MixerOrchestrator` and `SharedTimeline`, not by Python.

## Preheated Scene Geometry

`pyplumber/auto_mixer/preheated.py` reduces switch-time GPU work by keeping common
geometry filters warm. It defines template families such as:

- `face_full`
- `face_square`
- `face_conf_thumb`
- `orig_stack`
- `orig_pip_thumb`

`PreheatVideoRouter` nodes select which live camera feeds each hot geometry slot.
The mixer sees those hot slots as routed logical sources. Scene changes update
router routes and compositor layers instead of constantly restarting crop/scale
graphs for every camera/layout combination.

This is why the auto mixer can have many named scenes while still keeping the
runtime graph bounded: only the active scene, preview slot, and configured preheat
templates do work.

## Scene and Shot Policy

`pyplumber/auto_mixer/scenes.py` defines deterministic scene families such as
full-face, videoconference, vertical stacks, PiP, and sampled manual scenes.
When preheating is enabled, `preheated.py` defines equivalent scenes using routed
hot geometry sources.

`pyplumber/auto_mixer/shot_selector.py` and `profiles/` choose which scene should
represent the current speaker context. The fixed profile maps one active speaker
to a stable scene family; richer profiles can account for recent scenes, manual
suggestions, conversation windows, and layout variety.

`pyplumber/auto_switcher.py` runs the actual policy loop. On each tick it:

1. Observes the current program scene when remote control is enabled.
2. Enforces transition cooldown and minimum program dwell.
3. Selects a priority VAD-only speaker if configured.
4. Otherwise chooses a speaker where audio VAD and visual speech are both active.
5. Computes an optional scheduled `start_pts_ms` from audio event PTS.
6. Calls `MixerGraphBuilder.cut()`, `fade()`, or `wipe()`.

The policy thread controls the graph; it does not perform media processing.

## Output Layer

`pyplumber/auto_mixer/outputs.py` builds output-specific native node chains:

- HTML overlay input from DMA-BUF, conversion to CUDA frames, overlay gating, and
  final `source_switcher`.
- Video normalization and encoding.
- Audio normalization and encoding.
- File/RTMP mux output.
- Janus RTP video and optional audio output.

The output layer is intentionally outside `MixerOrchestrator`. For example, the
HTML overlay is toggled by direct node-object controls, while scene switching stays
inside the mixer command surface.

## Control Surfaces

There are three control paths:

- Python direct calls on `MixerGraphBuilder` during startup and auto-switching.
- The avplumber line-based control server, enabled by `--remote-control-port`.
- Web UI registration, enabled by `--webui-api` and `--instance-name`.

`AutoSwitchControlCommands` exposes runtime auto-switch controls through the main
control API. It can start/stop the switcher, change transition mode, set fade
duration, set the wipe file, and report status/native exceptions.

The Textual TUI in `tools/mixer_tui/` is an operator client for the same control
surface. It is not part of the media graph.

## Lifecycle and Groups

The graph is divided into groups:

- `input_<index>`: input decode, inference, speech nodes, and source legs.
- `preheat_routers`: native routers feeding hot geometry slots.
- `preheated_scene_geometry`: hot geometry filter nodes.
- `mixer_a`, `mixer_b`, `mixer`: native mixer slot/output groups.
- `output`: encoders, muxers, Janus RTP, and optional overlay chain.

`start_graph_groups()` starts them in dependency order with short startup gaps:
inputs first, then preheat routers/geometries, then mixer groups, then output. This
keeps downstream nodes from starting before upstream edges have a chance to produce.

Shutdown is handled by the runtime signal handler. It stops the auto-switcher and
control helpers; native node/group shutdown is owned by the underlying avplumber
process.

## Design Boundaries

- Python builds and controls the graph; native nodes do the heavy media work.
- Python nodes are acceptable for metadata, audio analysis, visual-speech state, and
  policy bridges. Avoid putting GPU frame transforms or high-bandwidth pixel copies
  in Python nodes.
- Mixer transitions should be expressed through `mixer.cut`, `mixer.fade`,
  `mixer.wipe`, and `SharedTimeline`; Python should not manually rewrite internal
  mixer nodes during a transition.
- Preheated geometry is an optimization layer above the mixer. The C++ mixer should
  not need to know whether a source is a raw camera edge or a hot routed geometry
  slot.
- Runtime-specific paths, hostnames, URLs, model locations, and wipe directories
  belong in command-line args or deployment configuration, not in committed graph
  defaults.
