# Fabric Redundancy Demos

This directory is split by payload codec:

```text
examples/redundancy/h264_intra/
examples/redundancy/nvjpeg/
```

Each codec folder keeps its own active-active, active-standby, and
point-to-point fabric demos where available.

## H.264 Intra

RTMP demos:

```text
examples/redundancy/h264_intra/active_active_sync_bars_rtmp/
examples/redundancy/h264_intra/active_standby_sync_bars_rtmp/
```

MPEG-TS restart/rejoin proof:

```text
examples/redundancy/h264_intra/active_standby_mpegts/
```

Point-to-point smoke test:

```text
examples/redundancy/h264_intra/fabric_tcp/
```

## NVJPEG

MPEG-TS restart/rejoin proofs:

```text
examples/redundancy/nvjpeg/active_active_mpegts/
examples/redundancy/nvjpeg/active_standby_mpegts/
```

Point-to-point smoke test:

```text
examples/redundancy/nvjpeg/fabric_tcp/
```

See `TEST_CASES.md` for the latest measured failover results.
See `EDGE_CASES.md` for the next failure scenarios to test.
