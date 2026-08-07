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
-   HAVE_NVJPEG=1: enable `nvjpeg_enc`. Requires `HAVE_CUDA=1` plus nvJPEG and CUDA runtime development libraries. Disabled by default.
-   HAVE_GL=1: enable OpenGL & EGL dependency, required by `drm_prime_to_cuda`, `cuda_to_egl_image`
-   HAVE_VAAPI=1: enable VAAPI paths (and implicitly OpenGL/EGL). Links `-lva -lGL -lEGL -lGLESv2`. Requires `libva-dev` and GL/EGL development packages.
-   HAVE_DRM=1: enable DMA-BUF IPC source and DRM-dependent paths. Requires `libdrm-dev`.
-   HAVE_TENSORRT=1: enable TensorRT inference nodes (`cuda_infer_yolo`, `cuda_infer_rtdetr`). Links `-lnvinfer -lnvinfer_plugin`. Optionally set `TENSORRT_ROOT=/path/to/TensorRT`.
-   NEURAL_NET=1: enable retained neural drawing, tracking, inference, scene-cut, OCR, and reframing nodes. `NEURAL_NET_COMMON=1` and `NEURAL_NET_SPECIFIC=1` are supported as compatibility aliases when `NEURAL_NET` is not set explicitly.
-   HAVE_JACK=1: enable `jack_sink`. Links `-ljack`. Requires `libjack-dev`.
-   HAVE_NVCC=1: build CUDA PTX used by CUDA processing nodes (`cuda_to_egl_image`, `cuda_infer_yolo`, `cuda_infer_rtdetr`). Requires `nvcc`.
-   HAVE_SCTE35=1: build SCTE35 libraries and `scte35_parse` node (used for inserting [ads](https://ublockorigin.com/) and switching to regional programs in TV distribution systems)
-   EMBED_IN=obs: [builds nodes and adds fields specific to OBS source plugin](library_examples/obs-avplumber-source/README.md)

Feature gates:
-   `cuda_to_egl_image` builds only when `HAVE_CUDA=1 HAVE_GL=1 HAVE_NVCC=1`.
-   `drm_prime_to_cuda` builds only when `HAVE_CUDA=1 HAVE_GL=1 HAVE_DRM=1`.
-   `nvjpeg_enc` builds only when `HAVE_CUDA=1 HAVE_NVJPEG=1`.
-   TensorRT inference nodes build only when `HAVE_CUDA=1 NEURAL_NET=1 HAVE_TENSORRT=1 HAVE_NVCC=1`.
-   `HAVE_GL` is auto-enabled when `HAVE_VAAPI=1`
-   `scte35_parse` builds only when `HAVE_SCTE35=1`


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
