# avplumber web-ui

A web UI made in Node & Svelte, for viewing:
* the graph, with grouped navigation and static queue activity indicators
* queues statistics (fill, pps)
* node creation-time parameters
* node objects (`node.object.get`)
* statistics from instance-shared objects (realtime, sentinel)
* stream analysis statistics from `stats.subscribe`

Graph arrows show direction. Blue indicates observed dequeue activity; gray indicates idle or unknown activity. Amber indicates a queue accumulating or not draining across consecutive samples, which may be intentional buffering. Red indicates newly observed drops. The small fill marker shows current occupancy, or the fullest queue in a grouped connection; hover shows rates and affected queue counts. Grouped rates are totals in items/second.

Indicators use the existing queue polling interval (one second by default), with no continuous animation. Paused, disconnected or stale telemetry is neutral. These sampled counters do not prove frame uniqueness, frame-perfect timing or an expected frame rate.

## How to use

```
cd frontend
npm run build
npm start
```

and when starting avplumber, specify `--webui-api` and `--instance-name`. Also, make sure that `--port` is specified, otherwise web UI won't be able to execute necessary commands.

## DiSCLAiMER

I'm not a frontend developer.

Written by AI.
