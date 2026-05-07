# AVP Libfabric Sink Node Spec

Date: 2026-05-05

Status: initial sender implemented and TCP loopback smoke-tested

Scope: first point-to-point AVP fabric sender for NVENC H.264 all-intra
packets. This spec intentionally covers only the sender node. The receiver,
redundancy selector, controller, host agent, and MXL adapter are separate
components.

## Goal

Create an avplumber node that sends already-encoded media packets over
libfabric with enough identity and timing metadata for later frame-perfect
redundancy.

Initial media path:

```text
AVFrame/CUDA
-> h264_nvenc all-intra
-> av::Packet
-> fabric_sink<av::Packet>
-> libfabric
```

Initial target:

```text
1080p60 H.264 all-intra
one encoded packet per video frame
CPU-accessible av::Packet payload
libfabric TCP provider first
libfabric shm provider option for same-host process-to-process tests
AWS EFA/RXM provider later
point-to-point producer-to-consumer media path
fabric_sink connects; fabric_source listens
```

## Non-Goals

The first sink node does not:

```text
select active/standby producer
perform receiver-side redundancy
own controller leases
start or stop graph processes
send raw CUDA frame memory
implement MXL grain semantics
transcode or inspect full H.264 bitstreams
guarantee frame-perfect failover by itself
```

The sink only serializes, timestamps, validates, and transmits packets.

## Node Type

Primary node:

```text
fabric_sink<av::Packet>
```

Alias allowed later:

```text
fabric_sink
```

Input:

```text
av::Packet
```

Output:

```text
none
```

Threading model:

```text
NodeSingleInput<av::Packet>
```

The node should consume packets from one avplumber queue and maintain its own
libfabric connection, send buffer pool, and completion handling.

## Example Graph

```text
node.add { "type": "enc_video", "src": "v_preenc", "dst": "v_encoded", "group": "producer", "name": "H264_Intra_NVENC", "codec": "h264_nvenc", "hwaccel": "@gpu", "options": { "preset": "p1", "tune": "ull", "rc": "constqp", "qp": "23", "g": "1", "bf": "0", "forced-idr": "1", "rc-lookahead": "0", "zerolatency": "1" } }
node.add { "type": "fabric_sink<av::Packet>", "src": "v_encoded", "group": "producer", "name": "VideoFabricOut", "provider": "tcp", "remote_host": "10.0.2.20", "remote_port": "5000", "stream_id": "main", "replica_id": 1, "generation": 1, "media_type": "video", "codec": "h264_intra", "frame_rate": "60/1", "require_keyframes": true, "max_payload_bytes": 4194304, "queue_depth": 64, "backpressure": "fail", "send_timeout_ms": 4, "latency_measure": true }
```

## Parameters

Required parameters:

| Name | Type | Meaning |
| --- | --- | --- |
| `src` | string | Source queue carrying encoded `av::Packet` values |
| `provider` | string | Libfabric provider hint, initially `tcp`; `shm` supported as same-host option |
| `remote_host` | string | Receiver host/IP for client-mode MVP |
| `remote_port` | string/int | Receiver service port |
| `stream_id` | string | Logical stream shared by all redundant replicas |
| `replica_id` | uint32 | Producer instance identity |
| `generation` | uint64 | Controller epoch; static for MVP |
| `media_type` | string | Initially `video` |
| `codec` | string | Initially `h264_intra` |
| `width` | uint32 | Encoded video width; may be discovered from upstream metadata |
| `height` | uint32 | Encoded video height; may be discovered from upstream metadata |
| `pixel_format` | string | Source/encoded frame pixel format, for example `cuda` or `nv12` |

Optional parameters:

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `endpoint_type` | string | `msg` | Libfabric endpoint model; MVP uses message semantics |
| `mode` | string | `connect` | MVP sender connects to receiver |
| `time_base` | rational string | packet timebase | Override packet timestamp timebase if needed |
| `remote_addr_file` | string | empty | Optional binary endpoint address file for providers such as `shm` that need out-of-band address exchange |
| `require_keyframes` | bool | `true` for `h264_intra` | Warn/fail on non-key packets |
| `keyframe_violation` | string | `warn` | `warn`, `drop`, or `fail` |
| `max_payload_bytes` | uint32 | `4194304` | Reject payloads larger than this |
| `queue_depth` | uint32 | `64` | Number of in-flight send buffers |
| `completion_batch` | uint32 | `16` | Max completions reaped per process loop |
| `backpressure` | string | `block` | `fail`, `block`, or `drop_old` |
| `max_queue_ms` | double | `33.334` | Used by `drop_old` |
| `send_timeout_ms` | double | `4` | Max wait for buffer/completion in blocking paths |
| `connect_timeout_ms` | uint32 | `1000` | Initial connection timeout |
| `reconnect` | bool | `false` | MVP should usually let graph restart handle reconnect |
| `latency_measure` | bool | `false` | Log send-path timing summary |
| `latency_warmup_frames` | int | `30` | Warmup frames excluded from timing |
| `latency_report_frames` | int | `300` | Frames before one summary log |
| `payload_crc` | bool | `false` | Compute payload CRC; useful for debugging |
| `header_crc` | bool | `true` | Compute header CRC |
| `stats_interval_ms` | uint32 | `1000` | Periodic stats log/export interval |

Future controller-managed parameters:

```text
connection_id
redundancy_group_id
active_policy
lease_id
controller_url
controller_host
controller_port
heartbeat_interval_ms
```

These should not be required for the first point-to-point node.

## SHM Address Exchange

The TCP provider can use the destination address returned by `fi_getinfo()`
from `remote_host` and `remote_port`.

The `shm` provider needs the receiver endpoint's actual local address. The
debug receiver supports:

```text
tools/fabric_dump_receiver --provider shm --port 5555 --addr-file /tmp/avp_fabric_shm.addr
```

It writes the binary `fi_getname()` address after `fi_enable()`. The sender can
then use:

```text
"provider": "shm",
"remote_host": "127.0.0.1",
"remote_port": "5555",
"remote_addr_file": "/tmp/avp_fabric_shm.addr"
```

This is a test-time stand-in for the future controller/source handshake. A
real `fabric_source` should publish or return its endpoint address through the
connection setup path instead of relying on a filesystem path.

## Frame Identity

The sink must stamp every packet with:

```text
stream_id
replica_id
generation
pts
dts
time_base
duration
```

`pts` is the packet PTS from the encoded `av::Packet`. It is also the
redundancy identity. It must match across redundant producers when they produce
the same visual frame.

Rules:

```text
packet PTS must be valid
packet timebase must be valid
frame_rate may be useful as validation metadata later, but it is not the identity source
generation must be included in receiver selection keys
```

Receiver equivalence key:

```text
stream_id + generation + normalized packet PTS
```

Replica is not part of equivalence. It identifies which producer supplied a
candidate for that frame.

## Wire Protocol

Each media message:

```text
AvpFabricWireHeader
AvpFabricMediaHeader
payload bytes
```

All integer fields must be encoded in network byte order or in a documented
fixed little-endian wire order. Pick one and keep it stable. The recommended
MVP choice is little-endian because both initial hosts are x86_64, but the
header must carry a version so this can be changed only by protocol version.

Fixed wire header:

```cpp
struct AvpFabricWireHeader {
    uint32_t magic;          // "AVPF"
    uint16_t version;        // 1
    uint16_t header_bytes;   // wire + media + extensions
    uint16_t message_type;   // media, heartbeat, eof, config
    uint16_t flags;
    uint32_t header_crc;
    uint32_t payload_crc;
    uint32_t payload_bytes;
};
```

Media header:

```cpp
struct AvpFabricMediaHeader {
    uint64_t stream_id_hash;
    uint32_t replica_id;
    uint64_t generation;

    int64_t pts;
    int64_t dts;
    int32_t time_base_num;
    int32_t time_base_den;

    uint32_t media_type;
    uint32_t codec;
    uint32_t packet_flags;
    uint32_t duration_num;
    uint32_t duration_den;

    uint32_t width;
    uint32_t height;
    uint32_t pixel_format;
    uint32_t real_pixel_format;

    uint64_t sender_wallclock_ns;
};
```

The video format fields are sent on every media frame, not only in a config
message:

```text
width
height
pixel_format
real_pixel_format
```

For CUDA frames, `pixel_format` can be `cuda` while `real_pixel_format`
captures the underlying software layout, for example `nv12`. For H.264 packets
produced by NVENC, the sink should obtain these fields from upstream AVP
metadata where possible. If metadata is not discoverable, the graph must provide
explicit `width`, `height`, and `pixel_format` parameters.

`sender_wallclock_ns` is the sender host wallclock at packet serialization/send
time. It is diagnostic/timeline metadata, not the primary frame identity. Across
hosts it is only as accurate as the host clock synchronization.

Message types:

| Name | Meaning |
| --- | --- |
| `media` | Encoded media payload |
| `heartbeat` | Optional sender liveness/timeline message, preferably sent to controller path |
| `eof` | End-of-stream marker |
| `config` | Optional codec/config side data message |

Packet flags:

```text
keyframe
config
discontinuity
eof
corrupt
late
```

For H.264 all-intra, payload is the encoded packet bytes from `av::Packet`.
Width, height, and pixel format still travel per frame in the media header so a
receiver can validate stream continuity and reconstruct AVP frame metadata
without relying only on out-of-band config.

H.264 payload format for the MVP is passthrough:

```text
payload bytes = av::Packet bytes as emitted by enc_video
packet_format = passthrough
```

The receiver must not infer video shape only from the payload. It should read
the per-frame AVP media header fields first.

## Control And Heartbeat Path

Media remains point-to-point:

```text
producer fabric_sink -> consumer fabric_source
```

Heartbeat/timeline telemetry may use a separate controller path:

```text
producer fabric_sink -> avp-controller/control-server
consumer fabric_source -> avp-controller/control-server
```

This keeps the media path low-latency and avoids requiring the consumer media
connection to receive control messages that are really for orchestration.

Optional controller parameters:

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `controller_host` | string | unset | Controller/control-server host for heartbeat telemetry |
| `controller_port` | string/int | unset | Controller/control-server port |
| `heartbeat_interval_ms` | uint32 | `250` | Periodic heartbeat interval when controller path is enabled |
| `heartbeat_transport` | string | `udp` or `tcp` TBD | Transport for controller telemetry |

If `controller_host` is not configured, the MVP sink does not need to send
heartbeats. The media header still carries enough identity for point-to-point
transport tests.

Heartbeat payload should include:

```text
stream_id
replica_id
generation
connection_state
last_sent_pts
next_expected_pts
time_base
frame_rate
sender_wallclock_ns
frames_sent
bytes_sent
backpressure_events
```

The future controller uses this to detect host/graph health, timeline progress,
and active/standby readiness. Missing-frame deadlines remain receiver/controller
timeline logic, not per-frame media-header fields.

## Libfabric Use

MVP providers:

```text
tcp
shm
```

Provider intent:

```text
tcp:
  first cross-host verification target; easy to inspect with tcpdump

shm:
  same-host process-to-process path through libfabric shared-memory provider;
  should reuse the same AVP wire header and sender logic
```

Recommended libfabric capabilities:

```text
FI_MSG
FI_SEND
FI_RECV only if connection protocol requires it
completion queue for send completions
```

MVP assumptions:

```text
one sender connection to one receiver
no router/server in the media path
sender runs in connect mode
receiver/source runs in listen mode
message boundaries preserved by selected endpoint mode
CPU memory payloads only
copy packet payload into libfabric-owned send buffers
one send buffer held until CQ completion
bounded in-flight buffer pool
```

EFA/RXM is a later validation target. The implementation must not hardcode
TCP-only assumptions into the AVP wire protocol or sender node. Provider-specific
address setup should be isolated in a small fabric transport wrapper so `tcp`,
`shm`, and later `efa` do not fork the node logic.

## Sender State Machine

```text
INIT
  parse parameters
  allocate buffer pool
  prepare static stream identity

CONNECTING
  resolve fabric address
  create fabric/domain/endpoint/CQ
  connect to receiver

RUNNING
  poll/reap send completions
  read packet from source queue
  validate packet
  serialize headers and payload
  post send
  update stats

DRAINING
  send EOF message on input EOF if possible
  wait for outstanding sends up to send_timeout_ms
  close fabric resources

FAILED
  fail node/group or retry according to reconnect policy
```

Shutdown must not hang forever on a blocked fabric operation.

## Backpressure

The node must have explicit behavior when no send buffer is available or
completion queue progress stops.

Supported MVP policies:

| Policy | Behavior |
| --- | --- |
| `fail` | Throw an error after `send_timeout_ms`; best for lab validation |
| `block` | Wait for buffer/completion; can stall live graph |
| `drop_old` | Drop older unsent frames and keep latest within `max_queue_ms` |

Recommended defaults:

```text
first MVP: block
stress/test graphs: fail
live production later: drop_old
```

For redundancy, stale frames are usually worse than missing frames.
For the first point-to-point transport MVP, the selected default is `block` so
the sender preserves frame delivery and exposes congestion as upstream graph
stalling rather than frame loss.

## Validation

Per packet:

```text
pkt.isComplete()
pkt.pts().isValid()
payload_size <= max_payload_bytes
if require_keyframes: pkt.isKeyPacket()
pts monotonic within stream/generation
```

Violations:

```text
missing PTS: fail
payload too large: fail
non-keyframe: keyframe_violation policy
pts regression: fail or mark discontinuity
```

The sender should warn if DTS/PTS are invalid or surprising, but PTS validity
is required for redundancy.

## Latency Measurement

If `latency_measure=true`, collect:

```text
serialize_ms
fi_send_post_ms
send_completion_ms
total_send_ms
```

Definitions:

```text
serialize_ms:
  wallclock to fill header and copy payload into send buffer

fi_send_post_ms:
  wallclock spent in fi_send/fi_sendmsg call

send_completion_ms:
  wallclock from successful send post to CQ completion

total_send_ms:
  wallclock from packet read to CQ completion
```

Summary log:

```text
FABRIC_SINK_LATENCY_SUMMARY name=<node> provider=<provider> samples=<n> serialize_ms_min=<...> serialize_ms_avg=<...> serialize_ms_max=<...> completion_ms_min=<...> completion_ms_avg=<...> completion_ms_max=<...> total_ms_min=<...> total_ms_avg=<...> total_ms_max=<...>
```

## Stats

Expose or log:

```text
connection_state
provider
remote_host
remote_port
stream_id
replica_id
generation
frames_in
frames_sent
frames_dropped
bytes_sent
send_errors
backpressure_events
inflight_sends
available_buffers
last_pts
avg_payload_bytes
max_payload_bytes_seen
```

These stats must be usable by a future host agent/controller.

## EOF And Side Data

On input EOF marker:

```text
send eof message if connected
drain outstanding sends up to timeout
finish node
```

H.264 extradata/SPS/PPS handling needs a receiver-side decision:

```text
Option A: rely on every all-intra packet carrying enough Annex B parameter data
Option B: send codec config message before media
Option C: include codec extradata extension in every keyframe until acknowledged
```

For the first MVP, use the actual packets emitted by avplumber/FFmpeg and
verify the receiver can decode the resulting packet stream. If not, add a
`config` message type before building redundancy features.

## Testing Plan

Unit-style tests:

```text
pts from PTS/timebase/framerate
header serialize/parse roundtrip
payload size validation
keyframe violation policy
backpressure policy selection
```

Local integration:

```text
producer graph writes H.264 all-intra to fabric_sink
test receiver dumps received AVP messages to file
file parser verifies frame count, monotonic pts, CRCs
```

End-to-end integration:

```text
producer: testsrc -> hwupload_cuda -> h264_nvenc -> fabric_sink
consumer: fabric_source -> h264_cuvid -> null_sink/output
verify 60 fps, no dropped frames, valid decoded CUDA frames
measure encoder, fabric, decoder latency separately
```

Failure tests:

```text
receiver not listening
receiver dies during send
network stalls
payload larger than max_payload_bytes
non-keyframe packet arrives when require_keyframes=true
producer EOF
```

## Implementation Estimate

Rough P2P sender only:

```text
2-4 days: rough TCP libfabric sender
1 week: clean shutdown, stats, validation, latency summary
2 weeks: source+sink pair with useful tests and file dump receiver
```

Production redundancy is larger because it needs receiver buffering,
duplicate suppression, generation changes, leases, and controller/agent
coordination.

## Open Questions

1. Should the first wire format use fixed little-endian fields or network byte
   order?
2. Should `fabric_sink` be `av::Packet` only for MVP, or should we immediately
   make it a template that can later accept raw `av::VideoFrame`?
3. Should non-key H.264 packets fail the graph by default, or only warn?
4. Should the first receiver expect Annex B H.264 packets directly, or should
   the sink send an explicit `config` message with codec extradata?
5. Should `generation` be static graph config for the MVP, or should we add a
   tiny local control API immediately so the controller can change it without
   restarting the graph?
6. For first latency tests, should backpressure default to `fail` or `block`?

Decisions so far:

```text
message size: one AVP frame per libfabric message for MVP; no fragmentation
topology: direct point-to-point producer -> consumer
connection mode: fabric_sink connects, fabric_source listens
payload lifetime: copy av::Packet payload into libfabric-owned send buffer
per-frame video metadata: send width, height, pixel_format, real_pixel_format
H.264 packet format: passthrough av::Packet payload bytes
frame identity: av::Packet PTS normalized to common timebase
backpressure default: block
CRC default: header_crc on, payload_crc off
wallclock metadata: include sender_wallclock_ns in every media header
heartbeat path: optional separate controller host/port, not media path by default
providers: tcp for cross-host/tcpdump, shm for same-host process transport
```
