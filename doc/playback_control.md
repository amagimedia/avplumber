# Playback control (experimental)

avplumber was designed for live streams but supports file playback control including seeking, pause, speed change, and reverse.

Reference graph: `examples/video_player.avplumber`

## Seeking

Seeking flushes queues graph-wide so that output responds immediately. The `seek` command takes the name of the **downmost speed-limiting node** (typically `realtime`); avplumber walks up from there, tells decoder nodes to seek, then issues the actual seek on `input_rec`.

```
seek rtsync now 30000    # seek to DTS = 30 s
```

## Pause / resume

```
pause p
resume p
```

(`p` is the `realtime` node name in the example graph)

## Speed control

```
speed.set s 0.25    # 4× slower
speed.set s 2       # 2× faster
speed.set s -1      # 1× reverse
```

## Fast seek

For minimum seek latency, encode the file with avplumber using:
- intra-frame-only codec in `enc_video`
- `seek_table` option on the `output` node

The `seek_table` file maps timestamps to byte offsets. In your player application, look up the desired timestamp in the table and seek to the corresponding byte offset, then issue:

```
seek rtsync now <timestamp>
```

Set `preseek` to `0` (or leave unspecified) in the `input_rec` node.
