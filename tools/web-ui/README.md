# avplumber web-ui

A web UI made in Node & Svelte, for viewing:
* the graph, with grouped navigation and static queue activity indicators
* queues statistics (fill, pps)
* node creation-time parameters
* node objects (`node.object.get`)
* statistics from instance-shared objects (realtime, sentinel)
* stream analysis statistics from `stats.subscribe`

It is a viewer: it cannot change, start, stop, add or delete anything on an
instance (see [Read-only bridge and network access](#read-only-bridge-and-network-access)).

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

and when starting avplumber, specify `--webui-api` and `--instance-name`. Also, make sure that `--port` is specified, otherwise web UI won't be able to query the graph.

Selecting an instance updates the URL with `?instance=<id>`; share that URL to
open the same graph. An unavailable instance displays a waiting state instead
of another graph. Use the WebUI server's reachable address when sharing.

Add `expand=<group>[,<group>…]` (e.g. `?instance=<id>&expand=aux_mv2,output`)
to show those groups' nodes individually in the grouped overview, framed in
view. A numbered group also opens its family, so its siblings (`aux_mv0`,
`aux_mv1`, …) appear as separate groups; `expand=aux_mv*` opens every group of
the family. The expansion stays for the page's lifetime; remove the parameter to
get the plain overview back.

## Read-only bridge and network access

The backend's `/ws` forwards only these read-only queries to an instance's
control port (`backend/requestGuards.js`): `nodes.json`, `queues.json`,
`queues.stats`, `sync_groups.json`, `correction_groups.json`,
`node.object.get <node> <object>` and `mixer.status <mixer>` (the preview's
cut-latency meter, [doc/mixer_cut_latency.md](../../doc/mixer_cut_latency.md));
for a mixer the instance does not have, `mixer.status` answers
`500 ERROR: unknown mixer: <mixer>` and creates nothing.
Arguments are names of 1 to 128 characters from `A-Z a-z 0-9 _ . @ : -`. Any other command, extra
arguments, or a control character such as a newline anywhere in the message
gets a `500 ERROR` response with an `error` text and never reaches the
instance. The *Messages* panel shows the UI's own notices; there is no command
input.

The backend listens on `127.0.0.1:22222` by default; `WEBUI_PORT` sets the port
and `WEBUI_BIND` the address (`0.0.0.0` for all interfaces; engines on other
hosts need that to register). The startup log prints the address it bound.
A WebSocket handshake is accepted only when its `Host` names `localhost`,
`127.0.0.1`, `::1`, the bind address, or a hostname in the comma-separated
`WEBUI_ALLOWED_HOSTS`, which stops DNS rebinding; set it to the names viewers
type, e.g. `WEBUI_BIND=0.0.0.0 WEBUI_ALLOWED_HOSTS=<host>`. A browser `Origin`
must name the same hostname as `Host` (any port); clients without an `Origin`
header, like the tests, are accepted. A reverse proxy must pass the browser's
`Host` unchanged (nginx: `proxy_set_header Host $http_host;`). Messages over
1 MiB close that connection.

`POST /api/instances`, `/api/instances/heartbeat` and `/api/stats` need
`Content-Type: application/json`, which the engine's REST client sends, and
answer 415 otherwise. Instance ids and stats stream names `__proto__`,
`constructor` and `prototype` are refused with 400.

There is no authentication: anyone who can reach the port can view the graph,
node parameters (including URLs) and the engine log. Expose it only where that
is acceptable.

## Tests

```
node --test tests/ frontend/src
node tests/check-instance-restart.cjs
node tests/check-read-only-bridge.cjs
```

## DiSCLAiMER

I'm not a frontend developer.

Written by AI.
