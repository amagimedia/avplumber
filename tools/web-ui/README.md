# avplumber web-ui

A web UI made in Node & Svelte, in the mixer operator page's look, for viewing:
* the graph as a zoomable map with live queue state
* queue statistics (frames in queue, rates, drops) and the node list
* node creation-time parameters
* node objects (`node.object.get`)
* statistics from instance-shared objects (realtime, sentinel)
* stream analysis statistics from `stats.subscribe`

It is a viewer: it cannot change, start, stop, add or delete anything on an
instance (see [Read-only bridge and network access](#read-only-bridge-and-network-access)).

## Pages

The top bar picks the instance and one of three pages; `#queues` and `#nodes` in the URL open the table pages.

* **Graph**: the graph card and a rail of widgets: *Selected* (the inspector), *Instance* (node, queue and drop
  totals), and *Objects*, *Stats*, *Shared* (instance-shared objects), *Logs* and *Messages* (the UI's own notices).
  Widgets can be dragged between the rail's stacks; *Reset layout* puts them back.
* **Queues**: every queue with its producer and consumer, fill, frames in queue now / average / maximum, capacity,
  rate, enqueued / dequeued / dropped totals and state; a filter and *All / Flowing / Idle / Problems*.
* **Nodes**: every node with its type, group, input and output queue counts and ON/OFF.

A row opens its node in the graph. The tables show at most 400 rows; the filter narrows them.

## Graph

The layout is computed once per `nodes.json`: a node's column is its longest path from a source, every group is a
band of rows, and the groups are gathered into areas (the inputs, the core, a numbered family such as the AUX
buses). Queue samples (`queues.json`, every second) only change colours and numbers, never positions.

The mixer's groups (`pyplumber/mixer`) carry its names (Slot A, Program, Outputs, AUX 3, MV); in other graphs a tile
shows the group's number or id.

Zooming changes the level of detail, by the size of one row on screen:

* **Overview** (under 12 px): each area is a grid of square tiles, one per group, coloured by health; areas are
  joined by summed links labelled with their frames per second and flowing queue count. A dot marks a source that a
  compositor draws now (its `active_inputs` over its input queues), as of the last `nodes.json`: the page fetches it
  when it loads the graph, so the dots do not follow later takes. A band along the bottom of a numbered input's tile
  shows its source kind by its source node: NVDEC (`dec_video`), browser (`ipc_dmabuf_source`) or raw
  (`raw_video_file_cuda`). The chips in the inputs header count them and are the legend: hovering one highlights its
  inputs, a click pins the highlight. Zoomed in, a tile names its kind under the number.
* **Groups** (12–30 px): every node as a dot with its name; the queues from the inputs are summed into one trunk per
  consumer.
* **Nodes** (30–90 px): small node cards and every queue, routed orthogonally, each target with its own lane.
* **Details** (90 px and more): node cards with the type, the rate and the frames in queue of their input.

Lines: green frames, blue packets, amber near full (average fill at 90% of capacity or more), red dropped since the
previous sample; a tile turns red only when a queue feeding one of its nodes lost more than 1% of its frames in that
time. Skips at a non-blocking fan-out (`one_to_many_nb`, latest frame wins) are by design and stay green.
Idle queues (nothing flows, e.g. an unsubscribed AUX tap) are not drawn; the Queues page lists them.

Clicking a node, a group, a connection or a queue selects it and highlights its path up- and downstream through flowing
queues; the inspector shows its queues (frames in queue now / average / maximum of capacity), which compositors draw
it (*Drawn by*) or which sources a compositor draws (*Drawing*), and its parameters.

Pinch or scroll to zoom, two-finger scroll or drag to pan. Keys: `1`–`4` Overview, Groups, Nodes, Details; `0` fit;
`+` / `-` zoom; `/` search; `Shift+F` fullscreen (the inspector then floats over the graph); `Esc` clears a pinned
source kind, else leaves fullscreen or clears the selection.

The graph is drawn with WebGL (sigma.js). A browser without it (GPU disabled, some remote desktops) shows a notice in
the graph card instead; search, the other widgets and the Queues and Nodes pages still work.

When the avplumber process behind an open instance restarts under the same host:port, the backend notices its lost control connection. This needs the engine to send heartbeats (`--webui-api` / `registerWithWebUI`); an `AVPLUMBER_PORT` or `POST /api/instances` entry is covered only when the engine also heartbeats to the backend from that host:port. The restarted engine's first heartbeat gives the instance a new `generation`, and open tabs reload its graph, keeping the view. The engine can register before its script has built the graph, but it answers control commands only after it is ready, so the reload receives all nodes. Their ON/OFF state is a snapshot: nodes started after the engine became ready (e.g. the mixer's aux buses) can show OFF, so tabs request `nodes.json` once more 5 s after the reload is answered. A dropped connection without a restart also causes one reload, at the next heartbeat (every 25 s by default); while the backend cannot reach the instance's control port at all, a tab that has it selected reloads at every heartbeat. The backend drops an instance created by heartbeats when none arrives for 30 s; with the 25 s interval a restart can exceed this after about 5 s of downtime. Open tabs then show the waiting state, clear the graph and selected node, and load the instance like a fresh selection when it registers again. `AVPLUMBER_PORT` and `POST /api/instances` entries are never dropped. Tabs connected to a restarted backend reload when their WebSocket reconnects, and again if their instance registers after that.

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

### Behind a reverse proxy

The page uses relative URLs for its assets, `api/instances` and the `ws` bridge, so it also works under a path
prefix. With nginx, strip the prefix, pass the WebSocket upgrade and the browser's `Host`, and set
`WEBUI_ALLOWED_HOSTS` to the hostname viewers type. Viewers only send GET requests (the WebSocket upgrade is one), so
the proxy can refuse the rest; engines then register on the loopback port directly
(`docker-compose/mixer/deploy/l4/nginx.conf.template` does this):

```
# http level:
map $http_upgrade $connection_upgrade { default upgrade; '' close; }

# server level:
location = /graph { return 301 /graph/; }
location /graph/ {
    limit_except GET { deny all; }
    proxy_pass http://127.0.0.1:22222/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection $connection_upgrade;
    proxy_set_header Host $http_host;
}
```

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
npm test                                  # node --test tests/ frontend/src
node tests/check-instance-restart.cjs
node tests/check-read-only-bridge.cjs
```

The graph model test also checks a recorded graph when `AVP_GRAPH_DUMP` names a dump of an engine's replies, a JSON
file `{"nodes.json": {"status", "body"}, "queues.json": {...}, "queues.stats": {...}}`:

```
AVP_GRAPH_DUMP=<dump.json> node --test frontend/src/graph/model.test.mjs
```

`tests/fakeEngine.cjs` replays such a dump as an engine's control port, advancing the queue counters by their rates;
`--drop` makes the named queues lose a tenth of their frames. With it the UI runs without an engine:

```
node tests/fakeEngine.cjs <dump.json> --port 5001 [--drop <queue>,<queue>]
AVPLUMBER_PORT=5001 npm start             # then open http://127.0.0.1:22222/
```

A browser check, after `npm run build`, against a Chrome started with `--remote-debugging-port=9222` (`CDP_URL`
sets another address): instance switching, late replies, deep links and a browser without WebGL, served under a path
prefix.

```
node tests/check-instance-switching.cjs
```

## DiSCLAiMER

I'm not a frontend developer.

Written by AI.
