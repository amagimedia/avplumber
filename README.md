# avplumber - make your own libav processing graph

avplumber is a graph-based real-time processing framework. Graph can be reconfigured on the fly using a text API. Most nodes are based on FFmpeg's libavcodec, libavformat & libavfilter. You can create entire transcoding & filtering chain in it, replacing FFmpeg in many use cases.

avplumber was created because we were experienced with FFmpeg and wanted to have its features, plus more flexibility. For example, it is possible to:

* encode once and send encoded packets to multiple outputs.
* filter video (using FFmpeg's filter graph syntax) in multiple threads. It is possible since FFmpeg 6.0, but we needed this feature long before its release.
* maintain output timestamps continuity **and** audio-video synchronization even when input timestamps jump.
* insert fallback slate ("we'll be back shortly") when input stream breaks.

Furthermore, it was designed to allow easy prototyping of new video & audio processing blocks (nodes in graph) without writing so much boilerplate code that is needed in case of libavfilter or GStreamer.

So does it replace FFmpeg in all use cases? Not at all. It is targetted at live use - currently it can't seek the input at all. Also, subtitles aren't supported due to limitations of the underlying library - avcpp.


## Quick start

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

### Using as a library

avplumber can be built as a static library: `make static_library` will make `libavplumber.a` which your app or library can link to. [`library_examples/obs-avplumber-source/CMakeLists.txt`](library_examples/obs-avplumber-source/CMakeLists.txt) is an example of CMake integration.

Public API is contained in [`src/avplumber.hpp`](src/avplumber.hpp).

Example: `library_examples/obs-avplumber-source` - source plugin for [OBS](https://github.com/obsproject/obs-studio) supporting video decoder to texture direct VRAM copy.

### Developing custom nodes

See [doc/developing_nodes.md](doc/developing_nodes.md)

## Graph
An avplumber instance consists of a [directed acyclic graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph) of interconnected nodes.

### Edges = queues
Nodes in the graph are connected by edges. Edge is implemented as a queue. `queue.plan_capacity` can be used to change its size. Type of data inside queue is determinated automatically when the queue is created.

Data types:
* [`av::Packet`](https://h4tr3d.github.io/avcpp/classav_1_1Packet.html) - encoded media packet
* [`av::VideoFrame`](https://h4tr3d.github.io/avcpp/classav_1_1VideoFrame.html) - raw video frame
* [`av::AudioSamples`](https://h4tr3d.github.io/avcpp/classav_1_1AudioSamples.html) - raw audio frame (usually 1024 samples of all channels)

Some nodes support multiple input/output types - they work like templates/generics in programming languages (and are implemented this way). If the data type can be deduced from source or sink edges, there is no need to provide it explicitly. But if it can't be, use template syntax in `type` field of the node JSON object:

```node_type<data_type>```

for example:

```split<av::VideoFrame>```

### Topology

Some nodes require that other node implementing specific features (an *interface*) is placed before (up) or after (down) it:

* `input` before `demux`
* `mux` before `output`
* video format metadata source before `enc_video`. It can be `dec_video`, `assume_video_format`, `rescale_video` or `filter_video`
* FPS metadata source before `enc_video`, `extract_timestamps` and `filter_video`. It can be `dec_video`, `force_fps`, `filter_video` or `sentinel_video`
* audio metadata source before `enc_audio` and `sentinel_audio`. It can be `dec_audio`, `assume_audio_format` or `filter_audio`
* time base source before `bsf`, `enc_video`, `enc_audio`, `extract_timestamps`, `filter_video`, `filter_audio`, `sentinel_video`, `sentinel_audio`. It can be `assume_video_format`, `assume_audio_format`, `dec_video`, `dec_audio`, `filter_video`, `filter_audio`, `force_fps`, `packet_relay` or `resample_audio`
* encoder (`enc_video`/`enc_audio`), `bsf` or `packet_relay` before `mux`

## Control methods
avplumber is controlled using text commands on TCP socket, so it can be controlled manually using `netcat` or `telnet`. `--port` argument specifies the port to listen on.

`--script` argument specifies commands to execute on startup.

## Control protocol
Control protocol loosely follows [MVCP](https://www.mltframework.org/docs/mvcp/).

On new connection, server (avplumber) sends a line: `100 VTR READY`

Client issues a command followed by arguments separated by spaces and ending with the new line character. The last argument may contain spaces.

Server (avplumber) responds with line with status code and information:
* `200 OK` - command accepted, empty response
* `201 OK` - command accepted, response will follow and an empty line marks the end of response
* `400 Unknown command: ...`
* `500 ERROR: ...`
* `BYE` and connection close - special response for `bye` command

Commands:

```hello```

Replies with `HELLO`

```version```

Replies with app version and build date

```bye```

Closes the connection

### Nodes management & control

```node.add { ...json object... }```

Add node

```node.add_create { ...json object... }```

Add and create node (without starting it right now)

```node.add_start { ...json object... }```

Add, create and start node

```node.delete name```

Delete node

```node.start name```

Start node

```node.stop name```

Stop node (`auto_restart` action is inhibited)

```node.stop_wait name```

Stop node, return reply when node really stopped

```node.auto_restart name```

Stop node and trigger its `auto_restart` action. This command is probably most useful for restarting input after changing its URL, with confidence that it will be handled the same way as if the previous input stream finished. And unlike `retry group.restart ...`, nothing blocks! See also more experimental `node.interrupt`.

```node.interrupt name```

Stop node *even if it is being constructed right now*. Also, bypass any locks. Currently only `input` node supports this command. After interruption, `auto_restart` action will be triggered.

```node.param.set node_name param_name new_json_value```

Change node parameter. Equivalent in JavaScript: `node['param_name'] = JSON.parse('new_json_value')`. **WARNING:** Node won't accept new parameters until restarted.

```node.param.get node_name```

Get whole node JSON object, for example:
```
node.param.get encode720
{"codec":"libx264","dst":"venc1","group":"out","name":"encode720","options":{"b":"4M","bufsize":"8M","flags":"+cgop","g":25,"level":"3.2","maxrate":"5M","minrate":"3M","muxdelay":0,"muxrate":0,"preset":"ultrafast","profile":"baseline","sc_threshold":0,"x264opts":"no-scenecut"},"src":"vs1o","type":"enc_video"}
OK
```

```node.param.get node_name param_name```

Get single parameter as JSON. Example:

```
node.param.get encode720 group
"out"
OK
```

```node.object.get node_name object_name```

Get object from node. Example:
```
node.object.get input streams
201 OK
[{"codec":"h264","index":0,"type":"V"},{"codec":"h264","index":1,"type":"V"},{"codec":"h264","index":2,"type":"V"},{"codec":"aac","index":3,"type":"A"},{"codec":"aac","index":4,"type":"A"},{"codec":"aac","index":5,"type":"A"}]

node.object.get input programs
201 OK
[{"index":0,"streams":[0,3,4,5]},{"index":1,"streams":[1,3,4,5]},{"index":2,"streams":[2,3,4,5]}]
```

### Queues (edges)

```queue.plan_capacity queue_name capacity```

Plan capacity of queue (which must be created **after** issuing this command). `capacity` is positive integer. **Warning**: [moodycamel::ReaderWriterQueue](https://github.com/cameron314/readerwriterqueue) (the library we use for queues) forces the queue size to be the smallest 2^n-1 (n=integer) larger or equal to specified capacity.

Default capacity can also be changed - use `*` as a queue_name, e.g. `queue.plan_capacity * 7`


```queues.stats```

Print (human-readable) statistics of queues (graph edges). Example:
```
[#...................]  1821.1  videoin
[....................]  243265  aenc1
[....................]  243265  aenc0
[....................]  243265  aenc2
[....................]  243265  mux2
[#####...............]  243265  aenc
[....................]  1821.03  audioin
 ^                      ^         ^
 queue fill             last PTS  queue name
```

```queue.drain queue_name```

Wait until queue is empty.

### Groups

```group.restart group```

```group.stop group```

```group.start group```

### Raw outputs

```output.start output_group```

```output.stop output_group```

### realtime nodes

```realtime.team.reset team```

Manually trigger reset of a realtime team.

### Playback control

```pause team_name now```

Tell all `pause` nodes in a team `team_name` to pause (stop passing packets)

```pause team_name at timestamp```

Tell all `pause` nodes in a team `team_name` to pause (stop passing packets) at specified timestamp.
Timestamp may be expressed in many forms like: `01:02:03` (hh:mm:ss), `01:00.150` (mm:ss.millis), `12000` (time expressed in ms), `2024-10-03T08:12:44.100` (wallclock time, ISO 9601 format with optional milliseconds).

```resume team_name```

Tell all `pause` nodes in a team `team_name` to resume playback

```seek team_name live```

Flush all queues between the `input` nodes and the nodes in a `team_name` team and seek to live video.
Live wideo is a video playback from recording which is still recorded. It is delayed 10 seconds after current time.

```seek team_name now timestamp```

Flush all queues between the `input` nodes and the nodes in a `team_name` team and seek to given timestamp.
Timestamp may be expressed in many forms like: `01:02:03` (hh:mm:ss), `01:00.150` (mm:ss.millis), `12000` (time expressed in ms), `2024-10-03T08:12:44.100` (wallclock time, ISO 9601 format with optional milliseconds).

```seek team_name at timestamp_when timestamp_to```

Flush all queues between the `input` nodes and the nodes in a `team_name` team and seek to given `timestamp_to` when playback time reaches `timestamp_when`.
Multiple `seek at` commands may be specified and it will executed in order of adding.
Timestamp may be expressed in many forms like: `01:02:03` (hh:mm:ss), `01:00.150` (mm:ss.millis), `12000` (time expressed in ms), `2024-10-03T08:12:44.100` (wallclock time, ISO 9601 format with optional milliseconds).

```seek team_name clear```

Clears `seek at` queue.

```speed.set team_name speed```

Set speed of the `speed` nodes belonging to the team `team_name` to `speed` (float).

```speed.get team_name```

Get current playback speed of the team `team_name`.

### Hardware acceleration

```hwaccel.init { "name": "name", "type": "type" }```

Init hardware accelerator which may be used for encoding, decoding or filtering video frames.
* name - identifier, supports global objects syntax (`@`). If accelerator with given name already exists, it isn't touched and no error is returned.
* type - currently only `cuda` is supported

### Statistics

```stats.subscribe { ... json object ... }```

Subscribe to statistics. JSON object structure:
```
{
  "url":"",         // URL to put statistics using HTTP POST. Empty to write them to console.
  "name":"qwerty",  // opaque
  "interval":1,     // refresh interval, in seconds
  "streams":{
    "audio":[
      {
        "q_pre_dec":"audioin",    // name of queue before decoder (kbitrate, speed, AV_diff are taken from it)
        "q_post_dec":"a10",       // name of queue after decoder (fps, width, height, pix_fmt, field_order, samplerate, samplerate_md, channel_layout are taken from it)
        "decoder":"Audio_Decode"  // name of decoder node (codec, type are taken from it)
      }
    ],
    "video":[
      {
        "q_pre_dec":"videoin",
        "q_post_dec":"v05",
        "decoder":"Video_Decode"
      }
    ]
  },
  "sentinel":"Video_Sentinel"  // name of sentinel node (card flag is taken from it)
}
```

### Events

```event.on.node.finished event_name node_name```

When node finishes, signal (wake up) an event `event_name`

```event.wait event_name```

Wait for event `event_name`

### Special commands

<code>retry <i>command arguments ...</i></code>

Repeat command until it succeeds. Intended for startup scripts (`--script`). Not recommended for remote control.

<code>detach <i>command arguments ...</i></code>

Return 200 OK immediately, run command in background thread.


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
-   `start_ts` (string) - start timestamp
-   `seek_table` (string of URL) - file with fast-seek offsets (may be generated by the `output` node)
-   `ts_offsets` (string of URL) - file with timestamp offsets (may be generated by the `sentinel` node)
-   `preseek` (float, seconds, default 0) - how many seconds to preseek back to increase chance of finding a keyframe, when seek to timestamp (`seek.ms` command) is requested
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

### `pause`

Stop passing through frames/packets and resume on request (`pause`, `resume` commands).

1 input, 1 output: anything

-   `team` (string, name of instance-shared object, default `"default"`) - team name that will be used for control
-   `paused` (bool) - sets the team initially in paused state

### `force_fps`

Duplicate and drop frames to achieve requested FPS

1 input, 1 output: `av::VideoFrame`

-   `fps` (string of rational) - target FPS as a string, e.g. `25` or `30000/1001`

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
    -   `false` (default): start all output streams at exact PTS = 10
        seconds (hardcoded in `PTSCorrectorCommon` class)
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

### `picture_buffer_sink`

Take a frame and write it to picture buffer that can be later used by `sentinel_video`.

1 input: `av::VideoFrame`

-   `buffer` (string, name of instance-shared object) - mandatory, picture buffer name
-   `once` (bool) - default true, finish after getting a single frame

### `null_sink`

Discards incoming packets just like `/dev/null`.

1 input: anything

no parameters

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

Seeking is complicated because queues need to be flushed to ensure that user doesn't have to wait for them to drain after requesting a seek. Also, we want to display frame after seek even when the player is paused. That's why seek commands (`seek.ms` and `seek.bytes`) need the name of the downmost node in the graph that limits output speed (in a video player it would be `realtime`). The graph is walked up, passing needed requests to decoder nodes and issuing the actual seek request to the `input` node.

See `examples/video_player.avplumber` for a typical graph with playback control including seeking. Example control commands compatible with it:

* `seek.ms rtsync 30000` - seek to DTS=30s
* `pause p`, `resume p`
* `speed.set s 0.25` - set speed to 4 times slower than realtime
* `speed.set s 2` - set speed to 2 times faster than realtime

### Fast seek

If you want seeking to be as fast as possible, you'll need a specially encoded file. You can make it with avplumber, too.

* Use intra-frame-only codec for `enc_video`
* Specify `seek_table` option of the `output` node

In your application controlling the player, parse the generated seek table and find byte offset corresponding to the timestamp you want to seek to. Then issue the command:

`seek.bytes rtsync BYTES_OFFSET`

Make sure that `preseek` is set to 0 (or unspecified) in the player's `input` node.


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

### How to dump avplumber config from log

```
sed -e 's/^.\+\[control\] Executing: \(.\+\)$/\1/; t; d' < log
```

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
