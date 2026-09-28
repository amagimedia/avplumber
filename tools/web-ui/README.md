# avplumber web-ui

A web UI made in Node & Svelte, for viewing:
* the graph, with grouped navigation and static queue activity indicators
* queues statistics (fill, pps)
* node creation-time parameters
* node objects (`node.object.get`)
* statistics from instance-shared objects (realtime, sentinel)
* stream analysis statistics from `stats.subscribe`

Graph arrows show direction. Blue indicates observed dequeue activity; gray indicates idle or unknown activity. Amber indicates a queue accumulating or not draining across consecutive samples, which may be intentional buffering. Red indicates newly observed drops. The small fill marker shows current occupancy, or the fullest queue in a grouped connection; hover shows rates and affected queue counts. Grouped rates are totals in items/second.

With live queue stats enabled, bold arrows indicate an active frame subscription, independently of traffic. A grouped arrow is bold if any member is subscribed; hover shows the active/total subscription count. Ordinary queues and older instances without subscription telemetry remain thin. This uses the existing queue refresh interval and does not change frame routing.

Indicators use the existing queue polling interval (one second by default), with no continuous animation. Paused, disconnected or stale telemetry is neutral. These sampled counters do not prove frame uniqueness, frame-perfect timing or an expected frame rate.

## How to use

```
cd frontend
npm run build
npm start
```

and when starting avplumber, specify `--webui-api` and `--instance-name`. Also, make sure that `--port` is specified, otherwise web UI won't be able to execute necessary commands.

Selecting an instance updates the URL with `?instance=<id>`; share that URL to
open the same graph. An unavailable instance displays a waiting state instead
of another graph. Use the WebUI server's reachable address when sharing.

Add `expand=<group>[,<group>…]` (e.g. `?instance=<id>&expand=aux_mv2,output`)
to show those groups' nodes individually in the grouped overview, framed in
view. A numbered group also opens its family, so its siblings (`aux_mv0`,
`aux_mv1`, …) appear as separate groups; `expand=aux_mv*` opens every group of
the family. The expansion stays for the page's lifetime; remove the parameter to
get the plain overview back.

## DiSCLAiMER

I'm not a frontend developer.

Written by AI.
