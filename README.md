# avplumber - make your own libav processing graph

avplumber is a graph-based real-time processing framework. Graph can be reconfigured on the fly using a text API. Most nodes are based on FFmpeg's libavcodec, libavformat & libavfilter. You can create entire transcoding & filtering chain in it, replacing FFmpeg in many use cases.

avplumber was created because we were experienced with FFmpeg and wanted to have its features, plus more flexibility. For example, it is possible to:

* encode once and send encoded packets to multiple outputs.
* filter video (using FFmpeg's filter graph syntax) in multiple threads. It is possible since FFmpeg 6.0, but we needed this feature long before its release.
* maintain output timestamps continuity **and** audio-video synchronization even when input timestamps jump.
* insert fallback slate ("we'll be back shortly") when input stream breaks.
* monitor input stream health, analyzing speed, actual FPS & sample rate, audio levels.
* reconfigure processing graph on the fly.

Furthermore, it was designed to allow easy prototyping of new video & audio processing blocks (nodes in graph) without writing so much boilerplate code that is needed in case of libavfilter or GStreamer.

However, it does not replace FFmpeg in all use cases. For example, subtitles aren't supported due to limitations of the underlying library - avcpp.

Curious about history and applications of this project? **Read [Story of avplumber — open source multimedia streaming engine from Amagi](https://medium.com/amagi-engineering/story-of-avplumber-open-source-multimedia-streaming-engine-from-amagi-fc649cce2637)** at [Amagi Engineering](https://medium.com/amagi-engineering) blog.

## Quick start

Note: be sure to [check other branches](https://github.com/amagimedia/avplumber/branches/active) ([tree view](https://github.com/amagimedia/avplumber/network)) if you want to test latest features.

Make sure to clone this repo with `--recursive` option.

    git clone --recursive https://github.com/amagimedia/avplumber
    docker build -t avplumber .
    docker run -p 20200:20200 avplumber -p 20200

or if you don't want to use Docker but have Ubuntu:

    apt install git gcc pkg-config make cmake libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavutil-dev libswresample-dev libcurl4-openssl-dev libboost-thread-dev libboost-system-dev libssl-dev
    make -j`nproc`
    ./avplumber

and in a different terminal:

    nc localhost 20200

and you can type some commands (see [Control protocol](#control-protocol)) or paste a script (e.g. from `examples/` directory)

### Development on Windows

Development on Windows can be done using Docker and VSCode Dev Containers.

1. Enable symbolic links by following [these steps](https://stackoverflow.com/a/59761201).
2. Clone this repo `git clone --recursive https://github.com/amagimedia/avplumber`
3. Open it in VSCode
4. Open Command Palette and run *Dev Containers: Reopen in Container* command

Development container comes with all required dependencies and clangd installed.

### Demo

To quickly run demo with FFmpeg test source, use the provided Docker Compose file:

    script=remux_analyze_audio.avplumber docker compose -f examples/compose/rtmp_test_source.yml up

After Docker pulls and builds everything, you should see stream statistics JSON lines, once per second.

Output stream will be available at `rtmp://localhost/live/output`

Change script to `complicated_transcoder.avplumber` to test transcoding.

This demo uses [MediaMTX](https://github.com/bluenviron/mediamtx) as streaming server.

### Running Docker on recent Mac OSX versions

    brew install docker docker-compose colima
    colima start

## Build process details

The build is driven by Makefile variables. Set them on the `make` command line, e.g.:

    make -j`nproc` HAVE_CUDA=1 HAVE_DRM=1 HAVE_NVCC=1

-   BUILD_TYPE: `Debug` (default) or `Release`
    -   Debug enables debug-only nodes (`jittergen`, `delaygen`).
    -   Release sets compiler flags to more optimization.
-   HAVE_CUDA=1: enable CUDA support and CUDA-based nodes. Uses dynlink loader, so does not require anything during compilation and lack of CUDA libraries in runtime is non-fatal (nodes not using CUDA will work normally)
-   HAVE_GL=1: enable OpenGL & EGL dependency, required by `drm_prime_to_cuda`, `cuda_to_egl_image`
-   HAVE_VAAPI=1: enable VAAPI paths (and implicitly OpenGL/EGL). Links `-lva -lGL -lEGL -lGLESv2`. Requires `libva-dev` and GL/EGL development packages.
-   HAVE_DRM=1: enable DMA-BUF IPC source and DRM-dependent paths. Requires `libdrm-dev`.
-   HAVE_TENSORRT=1: enable TensorRT inference nodes (`cuda_infer_yolo`, `cuda_infer_rtdetr`). Links `-lnvinfer -lnvinfer_plugin`. Optionally set `TENSORRT_ROOT=/path/to/TensorRT`.
-   HAVE_MEDIAPIPE=1: enable MediaPipe GPU face mesh sidecar node (`mediapipe_face_mesh_gpu`). Requires `HAVE_GL=1`. By default this runs `scripts/build_mediapipe_face_mesh.sh` to clone a pinned MediaPipe source tree and build only the AVP face-mesh bridge.
    -   The MediaPipe bridge build expects Bazel/Bazelisk, `git`, `pkg-config`, and OpenCV development files (`opencv-devel` on Fedora, providing `opencv4.pc`).
    -   **Docker build**: `Dockerfile.mediapipe` builds a self-contained image with the MediaPipe bridge, models, and pyplumber. Build it with `docker build -f Dockerfile.mediapipe -t avp-mediapipe:latest .` from the repo root on an x86 GPU host.
    -   **Runtime**: `libavp_mediapipe_face_mesh.so` must be on `LD_LIBRARY_PATH`. The CuPy overlay node also requires `libnvrtc.so` at runtime (NVIDIA Runtime Compilation); on hosts that have CUDA toolkit installed it is typically under `/usr/local/cuda/lib64/`. When running via Docker with `--gpus all`, mount or install `libnvrtc.so` inside the container and add its path to `LD_LIBRARY_PATH`.
-   HAVE_JACK=1: enable `jack_sink`. Links `-ljack`. Requires `libjack-dev`.
-   HAVE_NVCC=1: build CUDA PTX used by CUDA processing nodes (`cuda_to_egl_image`, `cuda_infer_yolo`, `cuda_infer_rtdetr`). Requires `nvcc`.
-   HAVE_SCTE35=1: build SCTE35 libraries and `scte35_parse` node (used for inserting [ads](https://ublockorigin.com/) and switching to regional programs in TV distribution systems)
-   EMBED_IN=obs: [builds nodes and adds fields specific to OBS source plugin](library_examples/obs-avplumber-source/README.md)

Feature gates:
-   `cuda_to_egl_image` builds only when `HAVE_CUDA=1 HAVE_GL=1 HAVE_NVCC=1`.
-   `mediapipe_face_mesh_gpu` builds only when `HAVE_MEDIAPIPE=1 HAVE_GL=1`.
-   `drm_prime_to_cuda` builds only when `HAVE_CUDA=1 HAVE_GL=1 HAVE_DRM=1`.
-   `cuda_infer_yolo` builds only when `HAVE_CUDA=1 HAVE_TENSORRT=1 HAVE_NVCC=1`.
-   `HAVE_GL` is auto-enabled when `HAVE_VAAPI=1`
-   `scte35_parse` builds only when `HAVE_SCTE35=1`

MediaPipe bridge variables:
-   `MEDIAPIPE_AUTO_BUILD=1` (default) builds the local face-mesh bridge under `deps/mediapipe-face-mesh` when `HAVE_MEDIAPIPE=1`.
-   `MEDIAPIPE_AUTO_BUILD=0` skips the clone/build step and expects `MEDIAPIPE_INSTALL_DIR` or `MEDIAPIPE_LIBS` to point at an existing bridge install.
-   `MEDIAPIPE_REV` pins the cloned MediaPipe revision used by `scripts/build_mediapipe_face_mesh.sh`.
-   `MEDIAPIPE_BRIDGE_SOURCE_DIR` (default `deps/mediapipe-bridge`) points at the AVP-owned bridge overlay copied into the MediaPipe checkout.
-   `MEDIAPIPE_SRC_DIR` (default `deps/mediapipe-src`) and `MEDIAPIPE_INSTALL_DIR` (default `deps/mediapipe-face-mesh`) control where the optional MediaPipe checkout and exported bridge are stored.


### Using as a library

avplumber can be built as a static library: `make static_library` will make `libavplumber.a` which your app or library can link to. [`library_examples/obs-avplumber-source/CMakeLists.txt`](library_examples/obs-avplumber-source/CMakeLists.txt) is an example of CMake integration.

Public API is contained in [`src/avplumber.hpp`](src/avplumber.hpp).

Example: `library_examples/obs-avplumber-source` - source plugin for [OBS](https://github.com/obsproject/obs-studio) supporting video decoder to texture direct VRAM copy.

## Developing custom nodes

See [doc/developing_nodes.md](doc/developing_nodes.md)

## Graph
An avplumber instance consists of a [directed acyclic graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph) of interconnected nodes.

### Edges = queues
Nodes in the graph are connected by edges. Edge is implemented as a queue. `queue.plan_capacity` can be used to change its size. Type of data inside queue is determinated automatically when the queue is created.

Data types:
* [`av::Packet`](https://h4tr3d.github.io/avcpp/classav_1_1Packet.html) - encoded media packet
* [`av::VideoFrame`](https://h4tr3d.github.io/avcpp/classav_1_1VideoFrame.html) - raw video frame
* [`av::AudioSamples`](https://h4tr3d.github.io/avcpp/classav_1_1AudioSamples.html) - raw audio frame (usually 1024 samples of all channels)
* `EglImageFrame` - GPU RGBA image passed by `EGLImageKHR` handle with PTS/timebase

Some nodes support multiple input/output types - they work like templates/generics in programming languages (and are implemented this way). If the data type can be deduced from source or sink edges, there is no need to provide it explicitly. But if it can't be, use template syntax in `type` field of the node JSON object:

```node_type<data_type>```

for example:

```split<av::VideoFrame>```


### Topology

Some nodes require that other node implementing specific features (an *interface*) is placed before (up) or after (down) it:

* `input`/`input_rec` before `demux`
* `mux` before `output`
* video format metadata source before `enc_video`. It can be `dec_video`, `assume_video_format`, `rescale_video` or `filter_video`
* FPS metadata source before `enc_video`, `extract_timestamps` and `filter_video`. It can be `dec_video`, `force_fps`, `filter_video` or `sentinel_video`
* audio metadata source before `enc_audio` and `sentinel_audio`. It can be `dec_audio`, `assume_audio_format` or `filter_audio`
* time base source before `bsf`, `enc_video`, `enc_audio`, `extract_timestamps`, `filter_video`, `filter_audio`, `sentinel_video`, `sentinel_audio`. It can be `assume_video_format`, `assume_audio_format`, `dec_video`, `dec_audio`, `filter_video`, `filter_audio`, `force_fps`, `packet_relay` or `resample_audio`
* encoder (`enc_video`/`enc_audio`), `bsf` or `packet_relay` before `mux`

## Control methods
avplumber is controlled using text commands on TCP socket, so it can be controlled manually using `netcat` or `telnet`. `--port` argument specifies the port to listen on.

`--script` argument specifies commands to execute on startup.

[Control protocol and all commands documentation](doc/control_protocol.md)

## Node object

Each node is described by a JSON object consisting of the following fields:

* `name` (string without spaces) - optional, specifies identifier that can be later used for controlling the node
  * if specified, must be unique within the instance
  * if unspecified, the string `type@memory_address` will be generated and used
* `type` (string) - mandatory
* `group` (string) - used for grouping together nearby nodes. Example: transcoder that will have separate input and output groups so that when input URL is changed, only demuxer and decoders will be restarted, not encoders and muxer.
* `auto_restart` (string) - optional:
  * `off` (default) - let the node stop without restarting
  * `on` - restart single node when it finishes/crashes
  * `group` - restart the whole group to which the node belongs
  * `panic` - when the node finishes/crashes, shutdown the whole avplumber instance
* `src` (string for single-input nodes, list of strings for multi-input nodes) - source edge
* `dst` (string for single-output nodes, list of strings for multi-output nodes) - sink edge
* `optional` (bool) - optional: when creating the node fails:
  * `true` - ignore exceptions (return 20x) and pretend nothing bad happened
  * `false` (default) - fail the whole operation (e.g. starting a group)

Most nodes have also their specific parameters which are specified on the same level as the fields above.

[List of all node types](doc/NODES.md)

### Non-blocking nodes

Some node types are non-blocking, which means that there is no separate thread to run the node, but it processes data in an event-based manner, which is configurable using the following fields:

* `event_loop` (string, name of instance-shared object) - name of the event loop, if not specified, `default` event loop will be used. Each event loop works in a separate thread.
* `tick_source` (string, name of instance-shared object) - name of the tick source. If not specified, node will work in tickless manner, waking up only when necessary (e.g. a node above in graph has put some data into queue). On the other hand, if this field is specified, the tick source will wake up the node at regular intervals synchronized to some external clock. This reduces latency and jitter. Currently useful only in [`OBS avplumber plugin`](library_examples/obs-avplumber-source/README.md) - specify `obs` as a `tick_source` to synchronize a non-blocking node to the video mixer's FPS.

The tick source has its own event loop (or may even bypass it and call the node in its own thread to reduce latency) so you can't specify both `event_loop` and `tick_source`.

### Example JSON syntax for fields

* string: `"string"`
* string of URL: `"protocol://domain/path"`
* string of rational: `"30000/1001"` (so-called 29.97 fps)
* list of strings: `["string1", "string2", "string3"]`
* dictionary (also known as map): `{"key1":"value1", "key2":"value2"}`
* bool: `true` or `false`
* int: `31337`
* float: `1337.42`
* name of an [instance-shared object](#instance-shared-objects): `"object"`
* name of a global instance-shared object: `"@global_object"`

## Node types

### `input`

1 output: `av::Packet`

-   `url` (string of URL)
-   `options` (dictionary) - options for libavformat
-   `timeout` (float, seconds) - packet read timeout
-   `initial_timeout` (float, seconds) - URL open timeout

### `input_rec`

1 output: `av::Packet`

This node is a special case on the `input` node with some changes required to plae & seek recoding files.
It has the same parameters as the `input` node and some additional

-   `url` (string of URL)
-   `options` (dictionary) - options for libavformat
-   `timeout` (float, seconds) - packet read timeout
-   `initial_timeout` (float, seconds) - URL open timeout
-   `live_delay` (float, seconds) - time to delay between latest available packet and currently displayed in live mode (default `1`)
-   `start_ts` (string) - start timestamp
-   `stop_ts` (string) - stop timestamp
-   `loop` (bool) - if set to true, video will be played in a loop
-   `stop_delay` (int, miliseconds) - time to hold on last read frame before reporting end od processing
-   `seek_table` (string of URL) - file with fast-seek offsets (may be generated by the `output` node)
-   `ts_offsets` (string of URL) - file with timestamp offsets (may be generated by the `sentinel` node)
-   `timestamp_source` (string) - timestamp source used to synchronize multiple videos. Possible values are `wallclock` and `input`. Both values are calculated from the timestamps offsets file (`ts_offsets`)
-   `preseek` (float, seconds, default 0) - how many seconds to preseek back to increase chance of finding a keyframe, when seek to timestamp (`seek` command) is requested
-   `team` (string, name of instance-shared object) - if specified, seeks on the `input` nodes may be synchronized inside specified team 

### `realtime`

Rate limit output packets/frames to wallclock. This way, DTS (in
packets) or PTS (in frames) differences will equal wallclock differences
at this node's sink.

Also, allows inter-stream synchronization (as long as timestamps in them are synchronized) and automatic flushing if too much data is buffered in queues before this node.

This node is non-blocking.

1 input, 1 output: anything

-   `leak_after` (float, seconds) - if specified, bypass rate limiting
    after having input packets available immediately (in other words, at
    least one packet was always enqueued) for specified time. Intended
    for segmented inputs for which realtime node is useful to prevent
    bursts, while we don't want clock drift problems and missed segments
    in long running streams.
-   `speed` (float) - default 1, implemented by scaling wallclock's
    timebase (millisecond precision) so values between ~0.9995 and
    ~1.0006 are treated as 1.
-   `tick_period` (string of rational, seconds) - if specified and [`tick_source`](#non-blocking-nodes) is also specified, anti-jitter filter will be enabled, assuming that tick source emits a tick every `tick_period`. Generally should be set to 1/FPS, e.g. `1/60`. The filter maintains its own clock independent of wallclock, but will resync to the wallclock if it drifts too much. If unspecified, wallclock will be used.
-   `set_pts` (bool, default false) - set PTS to wallclock timestamps corresponding to time when packets are outputted (or, more precisely, when they would be outputted if there was no jitter)

Input tolerance parameters:

-   `negative_time_tolerance` (float, seconds) - default `0.25`. Do not
    resync if newly arrived packet should have been emitted at most that
    much time in the past. 0 to disable and always resync in such
    situation - effectively increasing latency until sufficient
    buffering for smooth output is achieved.
-   `negative_time_discard` (float, seconds) - if specified, if newly
    arrived packet should have been emitted at least that much time in
    the past, discard this packet. Discarding has lower priority than
    resyncing (`negative_time_tolerance`), so the value must be less
    than `negative_time_tolerance` to make sense, equal or higher values
    disable discarding.
-   `discontinuity_threshold` (float, seconds) - default `1`. If we need to
    wait for more than specified time, treat as discontinuity and
    resync. Default value may be unsuitable (too small) for multiple
    source synchronization in case more data is buffered.
-   `jitter_margin` (float, seconds) - default `0`. When (re)syncing, add
    this value to the time to be waited. This prevents frequent
    resyncing and visible jitter at the cost of higher latency. It makes
    sense only with unbuffered output (e.g. display)
-   `initial_jitter_margin` (float, seconds) - default = `jitter_margin`.
    `jitter_margin` to use for the first frame received after node start,
    after discontinuity or after `leak_after`-triggered bypass, **but
    not** after "*negative time to wait (...), resyncing*"

Inter-stream synchronization parameters:

-   `team` (string, name of instance-shared object) - if specified,
    realtime nodes with the same team will cooperate to have their
    output synchronized. Use only if timestamps are synchronized.
-   `master` (bool) - default `true`. Only masters are allowed to resync in
    case of discontinuity. A team can have multiple masters.

Automatic flushing parameters (experimental):

-   `input_ts_queue` (string, name of queue) - which queue should be treated as the beginning of buffering chain. Usually should be set to the output of the demuxer. If unspecified, automatic flushing is disabled.
-   `intermediate_queues` (list of strings) - list of intermediate queues that will be examined whether they contain packets
-   `max_buffered` (float, seconds, default 5.5) - start flushing when the buffering chain has more than this amount buffered (calculated as difference between timestamp of last packet inserted into the `input_ts_queue` and timestamp of the frame coming to this node)
-   `min_buffered` (float, seconds, default 0.5) - stop flushing when the buffering chain has less than this amount buffered, or `input_ts_queue` and `intermediate_queues` are all empty

For each passing packet, time to wait is computed (how long should we
sleep before outputting that packet, to maintain realtime output rate)
and different actions are performed based on its value

-   if **timeToWait &lt;= 0**:
    -   if **timeToWait &lt; -negative\_time\_tolerance**:
        -   resync and emit packet
    -   else if **timeToWait &lt; -negative\_time\_discard**:
        -   discard this packet
    -   else: /\* **-negative\_time\_discard &lt;= timeToWait &lt;= 0**
        \*/
        -   emit packet immediately
-   else: /\* **timeToWait &gt; 0** \*/
    -   if **timeToWait &lt; discontinuity\_threshold**:
        -   *normal behavior* - wait timeToWait and emit packet
    -   else: /\* **timeToWait &gt;= discontinuity\_threshold** \*/
        -   resync and emit packet

Note: This pseudocode omits `leak_after` logic.

Recommended options for encoding from segmented input (HLS, DASH):

-   speed: 1.01
-   leak\_after: segment length \* 2

Recommended options for displaying live video:

-   negative\_time\_tolerance: 0.001 - 0.02 depending on clock precision of your system
-   jitter\_margin: 0.1 or some more

### `demux`

1 input, many outputs: `av::Packet`

-   `streams_filter` (string): if present, filter input streams according
    to ffmpeg syntax (for example "`p:1`" to select Program 1 - useful
    for HLSes) before parsing routing keys
-   `routing` (dictionary of string => string): stream mapping, keys are
    streams in input file, values are queues, like
    `{ "v:0": "videoin", "a:0": "audioin" }`, used instead of `dst`
    -   keys may be prefixed with `?` to indicate optional input (ignore
        the route if stream not found)
-   `wait_for_keyframe` (bool): default false, discard packets until a
    keyframe appears in any video stream

### `dec_video`, `dec_audio`

1 input: `av::Packet`, 1 output: `av::VideoFrame` or `av::AudioSamples` respectively

-   `codec` (string) - optional, codec name, auto-detected by libavcodec
    if not specified
-   `codec_map` (dictionary of string => string) - optional, use
    specified decoder for matching stream's codec, for example to use
    cuvid for h264 and hevc streams:
    `"codec_map": {"h264": "h264_cuvid", "hevc": "hevc_cuvid"}`
-   `pixel_format` (string) - optional, if unspecified, libavcodec and/or
    codec will select best possible pixel format for given input stream.
    As seen in `pix_fmt` field of FFmpeg's `AVPacket`, so it doesn't have to be any
    real pixel format (e.g. `yuv420p`) but can also be hardware
    acceleration specification (e.g. `cuda`)
    -   if starts with `?`, prefer specified pixel format, e.g. `?cuda`,
        but allow use of any
    -   if doesn't start with `?`, force specified pixel format, fail if
        it's incompatible with codec or stream
-   `hwaccel` (string, name of instance-shared object) - optional, name of
    hwaccel previously created with `hwaccel.init`
-   `hwaccel_only_for_codecs` (list of strings) - use hwaccel only for
    specified input stream codecs, useful because apparently setting
    `hw_device_ctx` in normally-software libavcodecs triggers frame
    corruption bugs
-   `options` (dictionary) - optional, options passed to libavcodec

### `extract_timestamps`

Set PTS to timecode from video frame's side data.

1 input, 1 output: `av::VideoFrame`

-   `team` (string, name of instance-shared object) - if specified,
    multiple `extract_timestamps` and `extract_timestamps_slave` nodes
    within the same team will share the same offset, so streams not
    containing timecode side data (e.g. audio) will be synchronized to
    the same timecode, too (as long as their PTSes are synchronized to
    each other).
-   `passthrough_before_available` (bool) - default `false`, meaning that
    processing will be blocked (and input queue will grow) as long as
    timecode isn't available. If true, frames will be passed through
    with PTS unchanged in such case.
-   `drop_before_available` (bool) - discard incoming packets before
    timecode is available. Disabled by default. Has lower priority than
    `passthrough_before_available`.
-   `timecodes` (list of strings), default `["S12M"]` - side data to get
    timecodes from. If specified timecode doesn't exist, next one in the
    list is tried. Possible items:
    -   `S12M.1` or `S12M` - SMPTE 12M = SEI
    -   `S12M.2`
    -   `S12M.3`
    -   `GOP`
-   `liveu` (bool, default false) - workaround for LiveU-encoded SMPTE
    12M. Treat drop bit as a part of frames field.
-   `frame_rate_source` (string) - timecodes have frame numbers, so to
    calculate PTS from them, number of frames per second must be known.
    Possible values:
    -   `fps` - default. Use FPS from nearest node implementing
        IFrameRateSource (decoder, filter or `force_fps`)
    -   `timebase` - use 1/timebase from nearest node implementing
        ITimeBaseSource (decoder, filter or `force_fps`)
    
    both values may yield the same or different results depending on
    input stream. Some video streams are generated by skipping every
    second frame from higher FPS stream, with S12M side-data preserved
    in remaining frames. In such cases `fps` is wrong and `timebase` is
    correct.

### `extract_timestamps_slave`

Set PTS to timecode extracted by `extract_timestamps` node.

1 input, 1 output: `av::VideoFrame` or `av::AudioSamples`

Supports parameters working the same as in `extract_timestamps` node:

-   `team` - mandatory
-   `passthrough_before_available`
-   `drop_before_available`

### `filter_video`, `filter_audio`

1 input, 1 output: `av::VideoFrame` or `av::AudioSamples`, respectively

-   `graph` (string) - [FFmpeg filter
    graph](https://ffmpeg.org/ffmpeg-filters.html#Filtergraph-description)
-   `hwaccel` (string, name of instance-shared object) - optional
    (mandatory for some filters), name of hwaccel previously created
    with `hwaccel.init`

### `speed_video`, `speed_audio`

Change timestamps so that playback speed changes in real-time.

1 input, 1 output: `av::VideoFrame` or `av::AudioSamples`, respectively

-   `team` (string, name of instance-shared object, default `"default"`) - team name that will be used for changing the speed using `speed.set` command
-   `discard_when_speed_changed` (bool, default false) - discard frames when speed isn't equal to 1. Intended for audio streams.
-   `timebase` (string of rational) - optional, if specified, will scale incoming timestamps to this timebase. Otherwise original timebase will be preserved. To get as smooth output as possible, set it to your native output timebase (1/fps).
-   `speed` (float) - initial speed, default: 1
-   `sync_team` (string, name of instance-shared object) - `realtime` team name. Required by smooth change playback direction (eg. from forward to backward) to flush all the queues betweeen input and the `realtime` nodes.

### `pause`

Stop passing through frames/packets and resume on request (`pause`, `resume` commands).

1 input, 1 output: anything

-   `team` (string, name of instance-shared object, default `"default"`) - team name that will be used for control
-   `paused` (bool) - sets the team initially in paused state

### `force_fps`

Duplicate and drop frames to achieve requested FPS

1 input, 1 output: `av::VideoFrame`

-   `fps` (string of rational) - target FPS as a string, e.g. `25` or `30000/1001`

### `smooth_timestamps`

Overwrite timestamps with a smoothed monotonic timeline (previous + duration), while keeping A/V sync by resyncing to input timestamps when averaged drift grows too large.

This node is non-blocking.

1 input, 1 output: `av::VideoFrame` or `av::AudioSamples`

For video:
- `duration` (string of rational, seconds) - frame duration, typically `1/FPS` (e.g. `1/25`, `1001/30000`), **or**
- `fps` (string of rational) - frames per second (alternative to `duration`)

For both video and audio:
- `resync_threshold` (float seconds, default `0.02`) - resync output timeline to input PTS when averaged drift exceeds this
- `drift_window` (int, default `300`) - drift averaging window size (samples); larger = smoother, smaller = more reactive
- `min_samples_before_resync` (int, default `150`) - ignore drift until this many frames/samples have been observed
- `discontinuity_threshold` (float seconds, default `2.0`) - hard reset (resync) when input PTS jump exceeds this

### `assume_video_format` / `assume_audio_format`

Set initial metadata to allow nodes that rely on them to start
when real metadata aren't available yet.

1 input, 1 output: `av::VideoFrame` or `av::AudioSamples`

Parameters for video:
-   `width` (int) - default 1920
-   `height` (int) - default 1080
-   `pixel_format` (string) - default `yuv420p`
-   `real_pixel_format` (string) - specify only if `pixel_format` is hardware-accelerated (e.g. `cuda`)

Parameters for audio:
-   `sample_rate` (int) - default 48000
-   `sample_format` (string) - default `s32p`
-   `channel_layout` (string) - default `stereo`

### `sentinel_video` / `sentinel_audio`

A sentinel
* guards streams against wild timestamps - PTSes jumping forward or backward, or repeating timestamps
* inserts backup frames when input signal is not available for specified time:
  * in audio stream: silence
  * in video stream, for maximum time of `freeze` parameter: repeated last frame
  * in video stream: custom slate, can be used for adding "we'll be back shortly" card

Sentinel's output has "ideal" timestamps with tolerance specified in sentinel's parameters. In other words, it ensures output stream continuity.

1 input, 1 output: `av::VideoFrame`/`av::AudioSamples`

-   `timeout` (float, seconds) - default 1, seconds to wait for input frame before
    inserting frozen or backup frame
-   `correction_group` (string, name of instance-shared object) -
    optional, defaults to `"default"`, used for sharing the clock between
    streams
-   `forward_start_shift` (bool):
    -   `true`: if input streams are present when starting the sentinels,
        forward relative shifts of their first packets (i.e. A-V offset)
        to output.
    -   `false` (default): start all output streams at exact PTS = `start_ts`
-   `max_streams_diff` (float, seconds) - default `0.001`, tolerance in seconds
-   `start_ts` (float, seconds) - default `10`, first output timestamp
-   `lock_timeshift` (bool) - after receiving first PTS, maintain constant input-output PTS difference. Disabled by default. Enable only if you're sure that input timestamps are synchronized to real-time clock.
-   `reporting_url` (optional, string of URL) - if specified, correction
    time shift changes will be reported to this URL as HTTP POST with
    JSON body:
      
    `{"changed_at":128.1,"input_pts_offset":126.75999999999999,"output_pts_offset":10.0}`

    -   `changed_at` - output timestamp of the change relative to first
        output PTS (`output_pts_offset`)
    -   `input_pts_offset` - what sentinel needs to add to input
        timestamp to achieve output PTS, minus `output_pts_offset`
    -   `output_pts_offset` = first output PTS, constant through
        processing, changeable using `start_ts` parameter

For video only:

-   `freeze` (float) - default 5, seconds to duplicate last good frame
    before outputting backup frame
-   `backup_frame` (string of URL) - backup frame (slate) image
-   `backup_picture_buffer` (string, name of instance-shared object) - read backup frame (slate) from this buffer. Use `picture_buffer_sink` to write frame to the buffer. Sentinel will reload the slate from the picture buffer every 64 frames and on every input signal break triggering slate insertion.
-   `initial_picture_buffer` (string, name of instance-shared object) - initialize last frame buffer with this buffer, so that at the beginning of stream it will be used for at most `freeze` duration. Useful to insert black frame instead of slate at the beginning when `forward_start_shift` is set to false. If unspecified, regular `backup_frame` or `backup_picture_buffer` will be used.

For `sentinel_video`, either `backup_frame` or `backup_picture_buffer` must be provided.

### `rescale_video`

Dynamic video scaler, maintains constant output dimensions and pixel
format even if input stream changes parameters. **Does not resample
FPS** (see `force_fps` node)

1 input, 1 output: `av::VideoFrame`

-   `dst_width` (int)
-   `dst_height` (int)
-   `dst_pixel_format` (string)
-   `flags` (list of strings) - list of possible flags:
    <https://www.ffmpeg.org/doxygen/3.2/swscale_8h_source.html#l00057>

### `resample_audio`

Dynamic audio resampler, maintains continuity of output stream even if
input stream changes parameters.

1 input, 1 output: `av::AudioSamples`

-   `dst_sample_rate` (int)
-   `dst_channel_layout` (string)
-   `dst_sample_format` (string)
-   `compensation` (float)
    -   `0` (default) means brutal compensation using built-in avplumber's
        sample dropping algorithm
    -   any negative value means brutal compensation using
        libswresample, **may not work correctly**
    -   positive value between 0 and 1 means soft compensation using
        libswresample, value means fraction of samples to compensate,
        **may not work correctly**

### `split`

1 input, multi outputs: anything

-   `drop` (bool) - drop packets if output queue is full, disabled by default

### `join_metadata`

Join metadata from an auxiliary stream into a primary stream by exact timestamp match.

2 inputs, 1 output: `av::VideoFrame` or `av::AudioSamples`

-   `src` must contain exactly 2 queues in this order: `[primary, auxiliary]`
-   timestamps are compared exactly (`primary.pts == auxiliary.pts`)
-   empty queue is treated as "not ready yet" (the node waits), not as missing frame
-   when both heads are present:
    - if timestamps match: copy auxiliary metadata to primary (`av_dict_copy`) and emit primary
    - if `primary.pts < auxiliary.pts`: emit primary unchanged (auxiliary missing for that primary timestamp)
    - if `auxiliary.pts < primary.pts`: drop auxiliary frame (primary missing for that auxiliary timestamp)
-   output timestamp/timebase remains the same as primary input

Useful for running heavy processing (e.g. neural network inference) on downscaled frames and merge produced metadata back onto original-resolution frames.

### `force_keyframe`

Set keyframe flag in frame to make encoder output keyframe. Unlike `-g`
encoder option in FFmpeg, works with non-integer FPS.

1 input, 1 output: `av::VideoFrame`

-   `interval_sec` (int / float / string of rational) - keyframe
    interval, in seconds

### `enc_video`, `enc_audio`

Encodes video or audio frames.

1 input: `av::VideoFrame` or `av::AudioSamples`, 1 output: `av::Packet`

-   `codec` (string) - mandatory
-   `options` (dictionary) - options passed to libavcodec
-   `hwaccel` (string, name of instance-shared object) - optional
    (mandatory for some encoders), name of hwaccel previously created
    with `hwaccel.init`
-   `timestamps_passthrough` (bool) - default `false`, intended for codecs
    that don't buffer data (otherwise bad things like repeated
    timestamps may happen), replace PTS & DTS in outgoing packet with
    incoming PTS

### `packet_relay`

Insert it between demuxer and muxer to remux packets without
transcoding.

1 input, 1 output: `av::Packet`

no parameters

### `bsf`

[BitStream filter](https://ffmpeg.org/ffmpeg-bitstream-filters.html)

1 input, 1 output: `av::Packet`

-   `bsf` (string) - name of bsf to use

### `mux`

multiple inputs, 1 output: `av::Packet`

-   `fix_timestamps` (bool) - shift PTSes and DTSes so
    that DTSes are always increasing and (PTS &gt;= DTS). Disabled by default.
-   `ts_sort_wait` (float, seconds) - default `2.5`, maximum time to wait
    for all streams to select the packet with least DTS. Set to `0` to
    emit packets as soon as they arrive.
-   `stream_ids` (list of positive integers) - if specified, will set custom stream ids in output container. Useful for `mpegts` format output.
-   `metadata` (list of dictionaries) - if specified, will set metadata in output container's streams.

### `output`

1 input: `av::Packet`

-   `format` (string) - mandatory
-   `url` (string of URL) - mandatory
-   `options` (dictionary) - format options that will be passed to libavformat
-   `seek_table` (string of file name) - if specified, binary file with native endianness with seek table will be written.
-   `seek_table_text` (string of file name) - if specified, text file with seek table will be written.

Format of binary seek table: 16-byte records containing:
* timestamp of the frame in milliseconds: int64_t
* bytes offset in the output file: uint64_t

Format of text seek table: values as above separated by space, each record in one line

### `jack_sink`

1 input: `av::AudioSamples` (sample format must be `fltp`, sample rate must be equal to JACK's)

Output audio to [JACK Audio Connection Kit](https://jackaudio.org/) server. Can be used for both inter-app routing and outputting audio to the sound card. Note that JACK has its own clock (soundcard clock or in case of dummy device - OS monotonic clock) so in long running streams underruns or overruns may occur, unless the stream's clock and JACK clock are synchronized (for example, input is AES67 RTP and the hardware audio interface is synchronized to AES67 master clock using wordclock or S/PDIF).

Both sample format and sample rate must match configuration of the JACK server (sample format is always `fltp` in JACK). Use [`resample_audio`](#resample_audio) node to convert.

This node has an internal buffer to ensure that JACK thread can run in real time. In case of bursty streams (e.g. coming from the Internet) this buffering may be insufficinent. The [`realtime`](#realtime) node with a small `negative_time_tolerance` (below JACK period size) will help in such cases.

-   `channels_count` (int, default 2) - number of JACK ports to create. If incoming audio stream has less channels, the remaining ones will be filled with silence. If more, the excessive channels will be discarded.
-   `port_prefix` (string, default empty string) - if specified, will append channel index to this string and it'll become the JACK port name. If unspecified, JACK port name will be only the channel index.
-   `connect_port_prefix` (string) - if specified, will append port number to this string and try to connect to an input port with that name in the JACK graph
-   `client_name` (string) - mandatory, name of the JACK client. If multiple `jack_sink` nodes are created with the same client name, ports will belong to the same JACK client (make sure to set `port_prefix` in such cases). If they have different client names, multiple JACK clients will be created. What is better depends on the use case - multiple JACK clients allow parallel processing but put more overhead on the CPU.

### `raw_output`

Write stream of raw packets or frames to file or pipe. Unlike `output`
node, can be used anywhere in graph. Outputs whatever will be thrown on
it. Use nodes `rescale_video` or `resample_audio` to convert to required
format.

1 input: anything

-   `path` (string) - file/pipe name, **not** URL, must be reachable via
    Unix file structure, libavformat protocols like `tcp://` aren't
    supported
-   `output_group` (string) - default "`default`". Name of output group
    for control (see `output.start` and `output.stop` commands) and
    synchronization. Since raw output doesn't have PTSes, avplumber will try
    to synchronize audio with video by cutting first audio frame to make
    it start together with first video frame.

### `reinterpret_planes_video` / `reinterpret_planes_audio`

Reinterpret plane pointers (like [`reinterpret_cast`](https://en.cppreference.com/w/cpp/language/reinterpret_cast.html)) without copying data. Works like a controlled cast of frame layout:
- For video, re-map data planes and change destination pixel format metadata.
- For audio (not tested yet), re-map planar channels and change destination sample format metadata.

No memory copies are performed. Plane pointers and buffer references are reused. Dimensions and linesizes are validated using `av_pix_fmt_` functions.

Intended as a more versatile alternative to [`shuffleplanes`](https://ffmpeg.org/ffmpeg-filters.html#shuffleplanes) which does not support hardware frames.

1 input, 1 output: `av::VideoFrame` or `av::AudioSamples` respectively

-   `plane_map` (object) - optional, `{ "dst_plane": src_plane }` mapping. If omitted, planes are mapped 1:1 up to the minimum plane count (e.g. `yuva420p` -> `yuv420p` drops alpha by default).

For video only:

-   `dst_pixel_format` (string, required) - destination software pixel format name (e.g. `gray`, `yuv420p`). When using hardware frames, this updates only `hw_frames_ctx->sw_format`; `AVFrame::format` (the hardware pixel format, e.g. `cuda`) stays unchanged.
-   `hw_frames` (bool, video) - default `false`. If `true`, the node expects a hardware input frame and keeps the hardware `AVFrame::format` unchanged. It clones `hw_frames_ctx` and sets its `sw_format` to `dst_pixel_format`.

For audio only:

-   `dst_sample_format` (string, required) - destination sample format name (e.g. `fltp`). Interleaved audio is supported only when source and destination sample formats match exactly (no copying is performed).

Constraints:
-   No conversions are performed. The node does not copy, up/download, or resample; it only reinterprets pointers and metadata.
-   For video, mapping between hardware and non-hardware must be consistent: `hw_frames=true` is required for hardware inputs; `hw_frames=false` rejects hardware inputs. Otherwise pointers would become invalid.
-   For audio, planar-to-planar remaps are supported; interleaved requires identical formats and is effectively pass-through.

Examples:

Map V component of `yuv420p` to grayscale:

```
{"type":"reinterpret_planes_video","name":"v_to_gray","group":"proc",
 "src":"v_in","dst":"v_out",
 "dst_pixel_format":"gray",
 "plane_map":{"0":2}}
```

Drop alpha from `yuva420p` to `yuv420p` (default plane map) in hardware frames:

```
{"type":"reinterpret_planes_video","name":"drop_alpha","group":"proc",
 "src":"v_in","dst":"v_out",
 "dst_pixel_format":"yuv420p",
 "hw_frames":true}
```

Audio: select channel 1 (second plane) as mono FLT planar:

```
{"type":"reinterpret_planes_audio","name":"pick_ch1","group":"proc",
 "src":"a_in","dst":"a_out",
 "dst_sample_format":"fltp",
 "plane_map":{"0":1}}
```

### `picture_buffer_sink`

Take a frame and write it to picture buffer that can be later used by `sentinel_video`.

1 input: `av::VideoFrame`

-   `buffer` (string, name of instance-shared object) - mandatory, picture buffer name
-   `once` (bool) - default true, finish after getting a single frame

### `null_sink`

Discards incoming packets just like `/dev/null`.

1 input: anything

no parameters


### `write_audio_envelope`

Writes audio envelope data for waveform display (no images; data only). One binary file per granularity level (like mipmaps for different zoom), plus an `index.json` describing layout. Files are appended incrementally; recording length need not be known in advance. To align envelope samples to video frames, use the video frame duration as one of the granularities (e.g. `"1/25"` for 25 fps).

1 input: `av::AudioSamples`

-   `path` (string) - directory (or base path) for output files; must be a filesystem path
-   `granularities` (list of strings) - segment duration per envelope sample, as rationals in seconds (e.g. `["1/25", "1/2", "1"]`). One file per entry

The node writes `index.json` (version, sample_rate, channels, **levels** map keyed by duration_sec with file name per level, **metrics** map keyed by id with offset_bytes and stride_bytes).

#### Binary layout

Each sample is stored as interleaved records with channel data, currently containing: positive peak, negative peak, RMS. To read one metric across channels, use `offset_bytes` and `stride_bytes` from `index.json`. For forward compatibility, do not assume that each channel's record will always contain 3 bytes.

Example for 2 channels (one envelope sample = currently 6 bytes):

```
  byte index:   0     1     2   |  3     4     5   |  6   ...
                ^     ^     ^   |  ^     ^     ^   |  ^
  metric:      pos   neg   rms  | pos   neg   rms  | pos
  channel:     ch0   ch0   ch0  | ch1   ch1   ch1  | ch0
  time range:  \________________________________/    \____...
                               0..1                     1...   (unit: segment duration)

  positive_peak: offset_bytes=0, stride_bytes=3  -->  read at 0, 3
  negative_peak: offset_bytes=1, stride_bytes=3  -->  read at 1, 4
  rms:           offset_bytes=2, stride_bytes=3  -->  read at 2, 5
```

Each envelope sample is 1 byte per value, 0.5 dB resolution, -127..0 dB range: 0 = -127dB or less, 254 = -0.5..=0dB, 255 = clip

#### Example `index.json`
```
{
  "channels": 2,
  "levels": {
    "1": { "file": "level_2.bin" },
    "1/60": { "file": "level_0.bin" },
    "1/8": { "file": "level_1.bin" }
  },
  "metrics": {
    "negative_peak": { "offset_bytes": 1, "stride_bytes": 3 },
    "positive_peak": { "offset_bytes": 0, "stride_bytes": 3 },
    "rms": { "offset_bytes": 2, "stride_bytes": 3 }
  },
  "sample_rate": 48000,
  "version": 1
}
```

#### Rendering envelope image recommendations

The renderer needs to:
1. read `index.json`
2. open binary file corresponding to the needed time resolution (recommended: greater or equal to waveform image resolution), specified in `levels` map
3. for each needed metric, initially jump to the `offset_bytes` and then jump by `stride_bytes` to read subsequent interleaved samples (so the actual stride for a single channel is `stride_bytes*channels`)
4. if waveform is to be rectified, compute `max(positive_peak, negative_peak)`
5. if mono waveform is to be displayed, compute maximum (for peaks) or RMS (for RMS) of all channels
6. if the read time resolution was greater than resolution of the waveform to be displayed, compute maximum or RMS of waveform values that correspond to the same “pixel” on the resulting image


### `ipc_cuda_source`

Get video frames from CUDA IPC memory. Frame pointer and parameters are read from named pipe. See `src/nodes/cuda/ipc_cuda_source.cpp` for structure.

1 output: `av::VideoFrame` (frame's pixel format will always be `cuda`)

-   `pipe` (string) - mandatory, path to named pipe
-   `hwaccel` (string, name of instance-shared object) - mandatory, name of
    hwaccel previously created with `hwaccel.init`

### `ipc_audio_source`

Get audio frames from named pipe. See `src/nodes/ipc_audio_source.cpp` for header structure. Header must be followed by interleaved audio samples.

1 output: `av::AudioSamples`

-   `pipe` (string) - mandatory, path to named pipe

### `parse_scte35`

Parse SCTE35 `SPLICE_INSERT` command.

1 input: `av::Packet`, must be connected to SCTE35 data stream (e.g. `d:0` key in routing map of the `demux` node)

-   `url` (string of URL) - if specified, JSON object with splice insert data will be HTTP POSTed to this url. If unspecified, the object will be printed to the log for debugging purposes.

### `extract_cc_data`

1 input: `av::VideoFrame`, 1 output: `av::Packet`

Extract ATSC A53 Part 4 Closed Captions data from video frame. Subtitle codec is usually EIA-708 or 608 in such side data. When outputted to UDP with libavformat's [special `data` 'muxer'](https://ffmpeg.org/ffmpeg-formats.html#Raw-muxers) (see [`examples/extract_cc_data.avplumber`](examples/extract_cc_data.avplumber)), subtitles can be parsed using [CCExtractor](https://ccextractor.org/) or GStreamer (YMMV).

no parameters

### `firewall`

Drop data with invalid timestamps; pass through otherwise. Useful to prevent malformed inputs from propagating downstream.

1 input, 1 output: anything

no parameters

### `ipc_dmabuf_source`

Receive GPU frames via a UNIX domain socket with FD passing (DMA-BUF). Produces DRM PRIME frames with metadata taken from the sender.

1 output: `av::VideoFrame` (hardware "pixel format" `DRM_PRIME`)

Parameters:
-   `socket` (string, required) - path to the UNIX domain socket
-   `hwaccel` (string, optional) - name of `hwaccel` object; when set, a matching `hw_frames_ctx` is attached for downstream filters/encoders

### `ipc_socket_audio_source`

Receive audio frames over a UNIX domain socket. Expects a simple header followed by interleaved float32 PCM.

1 output: `av::AudioSamples`

Parameters:
-   `socket` (string, required) - path to the UNIX domain socket
-   `sample_rate` (int, default `48000`)
-   `channels` (int, default `2`)
<!-- -   `bytes_per_sample` (int, default `4` for float32) CHANGING NOT IMPLEMENTED -->
-   `reconnect_delay_ms` (int, default `50`)

### `cuda_to_egl_image`

Convert CUDA `av::VideoFrame` to `EglImageFrame` (RGBA) using CUDA kernels and an EGL-backed texture pool. Intended for zero-copy rendering paths (e.g. OBS).

1 input: `av::VideoFrame` (pixel format `cuda`), 1 output: `EglImageFrame`

Parameters:
-   `pool_id` (name of instance-shared object, default `"default"`) - shared EGL image pool id
-   `pool_size` (int, default `8`) - pool capacity
-   `pool_max_size` (int, default `pool_size`) - maximum capacity after on-demand growth
-   `pool_grow_step` (int, default `8`) - entries added on each growth attempt
-   `sync` (bool, default `false`) - synchronize CUDA after conversion before publishing the EGL image

### `mediapipe_face_mesh_gpu`

Run MediaPipe Framework Face Mesh GPU on an `EglImageFrame` sidecar branch and emit face landmark metadata.

1 input: `EglImageFrame`, 1 output: `MetadataFrame`

This node is built only with `HAVE_MEDIAPIPE=1 HAVE_GL=1`. AVP does not include MediaPipe headers directly in the node; the optional build script exports a small bridge library plus the face detection/landmark model files. A typical CUDA decode path is:

```
dec_video(pixel_format="?cuda")
  -> filter_video(scale_cuda=...)
  -> cuda_to_egl_image
  -> mediapipe_face_mesh_gpu
  -> MetadataFrame consumer
```

Parameters:
-   `metadata_key` (string, default `"face_landmarks_v1"`) - key used in the emitted `MetadataFrame` object
-   `resource_root` (string, default `MEDIAPIPE_INSTALL_DIR/share`) - root used by the bridge to resolve MediaPipe model resources
-   `max_faces` (int, default `1`) - maximum faces tracked by the MediaPipe graph
-   `with_attention` (bool, default `true`) - use the attention face-landmark model for better lips/eyes landmarks
-   `use_prev_landmarks` (bool, default `true`) - let MediaPipe reuse previous landmarks for tracking
-   `infer_every_n` (int, default `1`) - process every Nth input frame
-   `emit_dropped_metadata` (bool, default `true`) - emit metadata for skipped frames
-   `speaking_start_open_ratio` (float, default `0.045`) - mouth-open ratio required to start visual-speaking state
-   `speaking_stop_open_ratio` (float, default `0.030`) - mouth-open ratio below which visual-speaking can stop
-   `speaking_start_confirm_frames` (int, default `2`) - consecutive frames required to emit `started_speaking`
-   `speaking_stop_confirm_frames` (int, default `5`) - consecutive frames required to emit `stopped_speaking`
-   `debug_log_every_n` (int, default `0`) - if non-zero, log inference status every N frames (face count, mouth open ratio, head pose, speaking state)

The metadata emitted per frame under `metadata_key` has the following structure:
```json
{
  "version": 1,
  "source": "mediapipe_face_mesh_gpu",
  "status": "ok",
  "pts": 8036,
  "timebase": {"num": 1, "den": 1000},
  "width": 960, "height": 540,
  "faces": [
    {
      "id": 0,
      "landmark_count": 478,
      "landmarks": [[x, y, z], ...],
      "head_pose": {"available": true, "yaw_deg": 12.5, "pitch_deg": -15.3, "roll_deg": -13.4},
      "mouth": {"available": true, "open_ratio": 0.003, "motion_score": 0.001, "speaking": false}
    }
  ],
  "events": []
}
```
`status` is `"ok"` when faces were detected, `"no_face"` when the graph ran but found nothing, `"skipped"` for frames dropped by `infer_every_n`, or `"error:<msg>"` on inference failure.

### `drm_prime_to_cuda`

Import DRM PRIME frames into CUDA frames via EGL/GL interop. Non-DRM PRIME frames are passed through unchanged.

1 input: `av::VideoFrame` (expects `DRM_PRIME` hardware "pixel format", pass-through otherwise), 1 output: `av::VideoFrame` (hardware "pixel format" `cuda`)

Parameters:
-   `hwaccel` (string, required) - CUDA device created with `hwaccel.init`

### `cuda_infer_yolo`

Run YOLO object detection on preprocessed CUDA frames using a prebuilt TensorRT engine (`.plan` / `.engine`).

1 input: `av::VideoFrame` (expects CUDA frame, currently NV12 sw_format), 1 output: `av::VideoFrame` (same frame, with detection metadata attached)

This node is inference-only in v1:
- upstream graph must handle resize/pad/crop/format preprocessing
- model input dimensions are read from the TensorRT engine
- detections are attached in metadata key (default `yolo_detections_v1`)

Parameters:
-   `models` (array of objects, required) - one or more model definitions. Each object has:
    -   `engine` (string, required) - path to TensorRT serialized engine (`.plan`/`.engine`)
    -   `class_names` (array of strings, optional) - class-label mapping by index
    -   `class_index_remap` (array of ints, optional) - remap decoded class IDs (e.g. `[1, 0]` swaps class 0 and 1)
    -   `output_box_format` (string, optional, default `end2end_xyxy`) - `end2end_xyxy` or `raw_cxcywh`
-   `hwaccel` (string, required) - CUDA device created with `hwaccel.init`
-   `metadata_key_out` (string, optional, default `yolo_detections_v1`) - output frame metadata key for detections JSON
-   `input_format` (string, optional, default `RGB`) - tensor channel order expected by model (`RGB` or `BGR`)
-   TensorRT input binding datatype may be `float32` or `float16`; node preprocess supports both and selects matching CUDA kernel automatically.
-   `conf_thresh` (float, optional, default `0.25`) - confidence threshold
-   `iou_thresh` (float, optional, default `0.45`) - NMS IoU threshold
-   `max_det` (int, optional, default `300`) - max detections per frame after NMS
-   `infer_every_n` (int, optional, default `1`) - run inference every Nth frame, pass through others unchanged
-   `debug_log_metadata` (bool, optional, default `false`) - print detection metadata to logs periodically
-   `debug_log_every_n` (int, optional, default `30`) - log period used with `debug_log_metadata`

Detection coordinates in metadata are emitted in model space (`coord_space = "model"`).

Example graph (RTMP -> CUVID decode -> CUDA preprocess -> YOLO -> null sink):
- `library_examples/obs-avplumber-source/examples/rtmp_input_hw_dec_cuda_yolo.txt`

### `cuda_infer_rtdetr`

Run RT-DETR object detection on preprocessed CUDA frames using a prebuilt TensorRT engine (`.plan` / `.engine`).

1 input: `av::VideoFrame` (expects CUDA frame, currently NV12 sw_format), 1 output: `av::VideoFrame` (same frame, with detection metadata attached)

v1 constraints:
- exactly one model entry in `models`
- fixed-shape batch-1 engine
- mandatory `output_contract: "rtdetr_e2e_v1"`
- expects end-to-end outputs compatible with `boxes[1,N,4]`, `scores[1,N]/[N]`, `labels[1,N]/[N]`
- `boxes_normalized=true` is not supported in v1

Parameters:
- `models` (array, required, size must be 1), model object fields:
  - `engine` (string, required) - TensorRT engine path
  - `output_contract` (string, required) - must be `rtdetr_e2e_v1`
  - `class_names` (array of strings, optional) - class-label mapping by index
  - `class_index_remap` (array of ints, optional) - remap decoded class IDs
  - `boxes_normalized` (bool, optional, default `false`) - unsupported in v1
- `metadata_key_detection` (string, optional, default `yolo_detections`) - output metadata key
- `input_format` (string, optional, default `RGB`) - tensor channel order (`RGB` or `BGR`)
- `conf_thresh` (float, optional, default `0.25`) - confidence threshold
- `max_det` (int, optional, default `300`) - max detections per frame
- `infer_every_n` (int, optional, default `1`) - run inference every Nth frame, pass through others unchanged
- `debug_log_metadata` (bool, optional, default `false`) - print metadata periodically
- `debug_log_every_n` (int, optional, default `30`) - log period when debug logging is enabled

### `drm_prime_to_egl_image`

Import DRM PRIME (DMA-BUF) frames into an `EGLImageKHR` via `EGL_EXT_image_dma_buf_import` and output them as `EglImageFrame` (no CUDA processing).

1 input: `av::VideoFrame` (expects `DRM_PRIME` hardware "pixel format"; non-DRM PRIME frames are dropped), 1 output: `EglImageFrame`

Supported DRM formats (layer0/plane0 only):
- `DRM_FORMAT_ABGR8888`
- `DRM_FORMAT_ARGB8888`

Cache behavior:
- Maintains an internal cache keyed by the incoming DMA-BUF FD number.
- Cache entries are evicted when an FD is not seen for `ttl` seconds, and the whole cache is purged when resolution changes.

Parameters:
- `ttl` (float seconds, optional, default `5.0`) - cache entry time-to-live

### `jittergen`

Enabled only if avplumber is compiled with `BUILD_TYPE=Debug`. Delay packets
or frames for a random time. Timestamps aren't modified. Delay will be
gradually decreased down to 1ms if a congestion is detected.

1 input, 1 output: anything

### `delaygen`

Enabled only if avplumber is compiled with `BUILD_TYPE=Debug`. Delay packets or frames for specified time. Timestamps aren't modified.

1 input, 1 output: anything

-   `delay` (float) - mandatory, delay in seconds

## Instance-shared objects

Some nodes (`sentinel`, `realtime`) can have shared state. It's stored in
instance-shared objects. Other nodes (`encoder`, `filter`) need the
instance-shared object created (`hwaccel.init`) before it's used in them.

If a name of an instance-shared object starts with `@`, it is global in
process address space. If not, its scope is limited to avplumber instance.

In case of avplumber launched as a standalone process, instance==process and
using global objects doesn't have any benefit.

In case of avplumber used as a library, each AVPlumber object is an avplumber instance.
Global objects can be used to share state between nodes of
different instances as long as they're within the same operating
system's process.

## Seeking infrastructure & playback control (experimental)

Despite the architecture initially being designed solely for handling live streams, latest updates to avplumber bring playback control support.

Seeking is complicated because queues need to be flushed to ensure that user doesn't have to wait for them to drain after requesting a seek. Also, we want to display frame after seek even when the player is paused. That's why seek commands (`seek`) need the name of the downmost node in the graph that limits output speed (in a video player it would be `realtime`). The graph is walked up, passing needed requests to decoder nodes and issuing the actual seek request to the `input_rec` node.

See `examples/video_player.avplumber` for a typical graph with playback control including seeking. Example control commands compatible with it:

* `seek rtsync now 30000` - seek to DTS=30s
* `pause p`, `resume p`
* `speed.set s 0.25` - set speed to 4 times slower than realtime
* `speed.set s 2` - set speed to 2 times faster than realtime
* `speed.set s -1` - set speed to reverse (1x)

### Fast seek

If you want seeking to be as fast as possible, you'll need a specially encoded file. You can make it with avplumber, too.

* Use intra-frame-only codec for `enc_video`
* Specify `seek_table` option of the `output` node

In your application controlling the player, parse the generated seek table and find byte offset corresponding to the timestamp you want to seek to. Then issue the command:

`seek rtsync now <timestamp>`

Make sure that `preseek` is set to 0 (or unspecified) in the player's `input_rec` node.


## Tips & tricks

### How to quickly change input on the fly

```
node.interrupt input
node.param.set input url "rtmp://new.stream/url"
```

Important: Execute the second command immediately after the first.

The first command stops input close to immediately (even if it's being
restarted right now). Input (if configured properly by `auto_restart`
policy) will restart itself (or the whole group) after a second. So we
issue the second command within that second, before internal lock on
nodes manager is acquired.

Note that if input is running normally (i.e. not starting right now),
the following commands will do effectively the same:

```
node.param.set input url "rtmp://new.stream/url"
node.auto_restart input
```

### Dump avplumber config from log

```
sed -e 's/^.\+\[control\] Executing: \(.\+\)$/\1/; t; d' < log
```

### Show nodes graph based on add node commands from log

```
./tools/graph_from_log_to_dot log > graph.dot
dot -Tsvg graph.dot -o graph.svg
xdg-open graph.svg
```

(may not work correctly with `detach` or `retry` commands, will not work with dangling edges, pull requests welcome!)

### View log without HLS muxer's spam

```
grep -Ev '^(EXT-X-MEDIA-SEQUENCE:[0-9]+|\[AVIOContext @ 0x[a-f0-9]+\] Statistics: [0-9]+ seeks, [0-9]+ writeouts|\[hls @ 0x[a-f0-9]+\] Opening '\''.+\.tmp'\'' for writing)$' logfile | less
```

### Watch queues fill in real time

```
watch -n0.1 "echo 'queues.stats' | nc localhost 20200"
```

In some versions of netcat it doesn't work. Try this:

```
watch -n0.1 "echo 'queues.stats\nbye\n\n' | nc localhost 20200"
```

If you have big queues, they may occupy multiple lines in terminal. To make them shorter:

```
while true; do echo -e 'queues.stats\nbye\n\n' | nc localhost 20200 | sed -E 's/#{16}/\$/g; s/\.{16}/,/g' ; sleep 0.1; done
```

### Find non-empty queues

open log file in less, press `/` or `?` and use this regular expression:

```
[1-9]0?/[0-9]{1,3},
```

## License and acknowledgements

Created by Teodor Wozniak <teodor.wozniak@amagi.com> https://lumifaza.org

Copyright (c) 2018-2024 Amagi Media Labs Pvt. Ltd https://amagi.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the [GNU Affero General Public License](LICENSE)
along with this program.  If not, see <https://www.gnu.org/licenses/>.


### FFmpeg

This program uses FFmpeg libraries.

FFmpeg codebase is mainly LGPL-licensed with optional components licensed under GPL. Please refer to [its LICENSE file](https://git.ffmpeg.org/gitweb/ffmpeg.git/blob/HEAD:/LICENSE.md) for detailed information.

### AvCpp

This program uses AvCpp - C++ wrapper for FFmpeg dual-licensed under the [GNU Lesser General Public License, version 2.1](deps/avcpp/LICENSE-lgpl2.txt) or [a BSD-Style License](deps/avcpp/LICENSE-bsd.txt)

### C++ Requests

This program uses C++ Requests (cpr) library.

Copyright (c) 2017-2021 Huu Nguyen

Copyright (c) 2022 libcpr and many other contributors

[MIT License](deps/cpr/LICENSE)

### Flags.hh

This program uses Flags.hh command line parser header.

Copyright (c) 2015, Song Gao

[BSD-3-Clause license](deps/flags.hh/LICENSE)

### ReaderWriterQueue

This program uses ReaderWriterQueue.

Copyright (c) 2013-2021, Cameron Desrochers

[Simplified BSD License](deps/readerwriterqueue/LICENSE.md)

### nlohmann::json

This program uses JSON for Modern C++ library licensed under the [MIT License](https://opensource.org/licenses/MIT)

Copyright &copy; 2013-2022 [Niels Lohmann](https://nlohmann.me)

### CUDA

This program uses CUDA loader taken from [NVIDIA's CUDA samples](https://github.com/NVIDIA/cuda-samples/tree/e8568c417356f7e66bb9b7130d6be7e55324a519).

Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

[BSD-3-Clause license](deps/cuda_loader/LICENSE)
