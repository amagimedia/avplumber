Unfinished or written-but-untested nodes.

## AI-generated summary:

### `limit_fps`

Drop video frames so the effective output FPS does not exceed a maximum, based on PTS deltas.

1 input, 1 output: `av::VideoFrame`

-   `max_fps` (int/float, required) - maximum frames per second

### `source_switcher`

Switch between multiple input queues, forwarding only the “current” source. If the current source times out or no data arrives within its timeout, the node switches to the next source (round-robin). When switching, it can start/stop node groups.

multi-input, 1 output: anything

Parameters:
-   `src` (list of strings, required) - input queue names
-   `dst` (string, required) - output queue name
-   `src_params` (object, required) - per-source options keyed by the matching `src` name. Each entry supports:
    -   `timeout` (float, seconds) - switch away when no data within this period
    -   `group` (string) - node group to `start` on switch-in and `stop` on switch-out

### `sync_buffer`

Shared-clock buffering for smoothing jitter and aligning streams. Buffers until a small amount of time is enqueued, then outputs paced to wallclock. Multiple `sync_buffer` nodes can share timing through a common group.

1 input, 1 output: anything

-   `sync_group` (string) - name of instance-shared sync group (default `"default"`)

Notes:
-   Internal thresholds (like initial buffered duration) are currently fixed; this node is intended primarily for smoothing and simple A/V alignment.

