# Fabric Redundancy Edge Cases

These are follow-up tests for the two-producer / one-consumer redundancy MVP.
The current best proof path is NVJPEG active-standby:

```text
examples/redundancy/nvjpeg/active_standby_mpegts/
```

Known good dump:

```text
/home/jp/out_nvjpeg_complicated_restart_rejoin.ts
duration: 53.480s
max packet delta: 0.040s
bad gaps >60ms: 0
ffmpeg decode warnings: none
```

## Priority Tests

### 1. Standby Joins Too Late

Scenario:

```text
A starts active.
B is not running.
Start B as standby.
Kill A before B reaches replica eligibility.
```

Expected behavior:

```text
Consumer should either hold/freeze within configured playout window or declare
no eligible standby. It must not promote an unaligned replica.
```

Risk covered:

```text
False promotion before offset/timeline identity is trustworthy.
```

### 2. Bad Frame Identity From Standby

Scenario:

```text
B sends valid generation and status packets, but pts is far outside
the current active timeline.
```

Expected behavior:

```text
fabric_source should quarantine or ignore the replica until sane frame identity
is observed. It must not advance next_emit_frame to the bad value.
```

Risk covered:

```text
The earlier archived TS failure where a restarted standby pushed the source to
an extreme frame slot.
```

### 3. Repair Cache Too Small

Scenario:

```text
B standby repair_window_frames is smaller than playout_delay_frames plus
network/control delay.
Kill A and request repair for a frame already evicted by B.
```

Expected behavior:

```text
Source should skip only the unrecoverable frame range, log the miss clearly,
and resume at the next available promoted payload.
```

Risk covered:

```text
Silent deadlock waiting for repair payload that can never arrive.
```

### 4. Active Stalls Without Death

Scenario:

```text
A process remains alive but stops sending media.
B continues status only.
Then A resumes after B has been promoted.
```

Expected behavior:

```text
Consumer promotes B after active_timeout_ms and does not flap back to A unless
policy explicitly allows preferred-owner return after stable rejoin.
```

Risk covered:

```text
Flapping between two live replicas.
```

### 5. Bursty Replica Jitter

Scenario:

```text
Inject 500-1500 ms burst jitter on one producer while preserving valid media
PTS/frame identity.
```

Expected behavior:

```text
Selector should keep output cadence stable while the replica remains within
the configured playout/repair window. If not, it should fail predictably with
logged misses.
```

Risk covered:

```text
AWS cross-AZ jitter and delayed TCP delivery.
```

### 6. Dropped Or Duplicated Control UDP

Scenario:

```text
Drop REPAIR.
Drop PROMOTE.
Duplicate REPAIR.
Duplicate PROMOTE.
Deliver PROMOTE before REPAIR.
```

Expected behavior:

```text
Control messages must be idempotent. Duplicate commands should not corrupt
state, and lost commands should be retried or time out cleanly.
```

Risk covered:

```text
UDP control path reliability.
```

### 7. Old TCP Packets After Replica Restart

Scenario:

```text
Replica A restarts with the same replica_id and new generation while old
packets from the previous process are still draining.
```

Expected behavior:

```text
Generation checks must reject stale packets and avoid mixing old/new repair
cache entries.
```

Risk covered:

```text
Generation boundary corruption.
```

### 8. Consumer Restart While Producers Keep Running

Scenario:

```text
A active and B standby are running.
Kill/restart the consumer fabric_source graph.
Do not restart producers.
```

Expected behavior:

```text
Consumer should relearn replica offsets, select the active producer, and become
eligible for failover again without requiring producer restart.
```

Risk covered:

```text
Controller/source process crash recovery.
```

### 9. Active Backpressure

Scenario:

```text
Block or slow the active fabric_sink path so media delivery stalls, while the
process and input decode continue.
```

Expected behavior:

```text
Consumer should distinguish transport/media stall from process death. Promotion
should happen only after the configured timeout and should be logged as a
transport miss.
```

Risk covered:

```text
False promotion caused by receiver or network backpressure.
```

### 10. H.264 Packet-Copy Boundary

Scenario:

```text
Use H.264 intra packet-copy active-standby.
Switch only when the new replica provides a clean IDR access unit with SPS,
PPS, and AUD/header data.
```

Expected behavior:

```text
No decoder corruption after switch. If a clean boundary is unavailable, source
must drop new-replica packets until one appears.
```

Risk covered:

```text
Independent encoder handoff corruption.
```

## Current Notes

NVJPEG is the safer proof path for frame-level redundancy because every payload
is independently decodable before the final consumer-side H.264 encoder.

H.264 packet-copy can still be useful, but it needs explicit access-unit
boundary validation before it should be considered equivalent.

Shutdown remains a separate issue:

```text
The latest NVJPEG media test passed continuity checks, but final teardown still
aborted around decoder shutdown and producers did not exit cleanly on SIGINT.
Fix shutdown independently from media selection.
```
