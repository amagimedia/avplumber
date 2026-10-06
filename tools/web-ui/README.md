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

When the avplumber process behind an open instance restarts under the same host:port, the backend notices its lost control connection. This needs the engine to send heartbeats (`--webui-api` / `registerWithWebUI`); an `AVPLUMBER_PORT` or `POST /api/instances` entry is covered only when the engine also heartbeats to the backend from that host:port. The restarted engine's first heartbeat gives the instance a new `generation`, and open tabs reload its graph, keeping the view and `?expand=` groups. The engine can register before its script has built the graph, but it answers control commands only after it is ready, so the reload receives all nodes. Their ON/OFF state is a snapshot: nodes started after the engine became ready (e.g. the mixer's aux buses) can show OFF, so tabs request `nodes.json` once more 5 s after the reload is answered. A dropped connection without a restart also causes one reload, at the next heartbeat (every 25 s by default); while the backend cannot reach the instance's control port at all, a tab that has it selected reloads at every heartbeat. The backend drops an instance created by heartbeats when none arrives for 30 s; with the 25 s interval a restart can exceed this after about 5 s of downtime. Open tabs then show the waiting state, clear the graph and selected node, and load the instance like a fresh selection when it registers again (`?expand=` groups are kept). `AVPLUMBER_PORT` and `POST /api/instances` entries are never dropped. Tabs connected to a restarted backend reload when their WebSocket reconnects, and again if their instance registers after that.

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
