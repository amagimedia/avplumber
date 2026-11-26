# REST API

An optional HTTP gateway to a running avplumber instance, described by OpenAPI 3.1.
It is Python code (`pyplumber/rest/`) that talks to the engine over its
[control port](control_protocol.md). The engine does not link, load or know it:
a build or deployment that does not start the gateway is unchanged, and the
gateway's dependencies (`pyplumber/rest/requirements.txt`: FastAPI, pydantic,
uvicorn) are imported by nothing else in `pyplumber`.

```
HTTP client ──► gateway (pyplumber.rest) ──► control port ──► avplumber
                 /api/v1/health
                 /api/v1/engine/…      core, always served
                 /api/v1/mixer/…       pyplumber.mixer.rest, when the instance runs a mixer
                 /api/v1/<yours>/…     routers of the embedding application
```

* [doc/api/openapi.json](api/openapi.json) is the document of the core and the
  mixer router, for client generators. A test fails when it differs from the code;
  regenerate it with `python -m pyplumber.rest.export`.
* A running gateway serves the document of what it mounted (application routers
  included) at `/api/v1/openapi.json`, and Swagger UI for trying requests at
  `/api/v1/docs`. Swagger UI loads its script and style sheet from a CDN, so the
  browser needs internet access; the JSON document does not.

## Running

```
pip install -r pyplumber/rest/requirements.txt
python -m pyplumber.rest --port 20200                  # the engine only
python -m pyplumber.rest --port 7777 --mixer mixer     # and the mixer named "mixer"
python -m pyplumber.rest --fake                        # no engine: an in-memory one with a mixer
```

`--host`/`--port` are the engine's control address (default `127.0.0.1:7777`),
`--bind`/`--http-port` the address the API is served on (default
`127.0.0.1:7682`). `--mixer NAME` is the name the mixer was created with
(`mixer.status NAME`); `pyplumber.mixer` applications use `mixer`.

`--fake` answers from memory (`pyplumber/rest/fake.py`): four scenes, two keys,
two AUX buses (`mv` follows the layout, `monitor` is external), two audio
sources, a ducker. A take is on Program at once, and a key is on or off without
a fade. Use it to write a client away from a mixer host.

### Who may call

The control port has no authentication, and neither has the gateway by default,
so it binds to `127.0.0.1`. To serve other hosts:

```
AVP_REST_TOKEN=<token> python -m pyplumber.rest --bind 0.0.0.0 --token-env AVP_REST_TOKEN --mixer mixer
```

Every route except `/api/v1/health`, `/api/v1/openapi.json` and `/api/v1/docs`
then requires `Authorization: Bearer <token>`. The gateway speaks plain HTTP:
put it behind a TLS-terminating proxy when the network between client and
gateway is not trusted. Binding to another address without a token is allowed
and logged as a warning; restrict access in the deployment.

A browser request made by another site's page (an `Origin` header naming another
host, or `Sec-Fetch-Site: cross-site`) is refused with 403, and request bodies
are JSON only. The gateway sends no CORS headers: a web application calls it
from its own origin, through a reverse proxy.

### Embedding

An application that already holds a `MixerBridge` (the mixer's control page
does) serves the gateway on a thread, so the page and the API share one ordering
of takes and one cached state:

```python
from pyplumber.mixer.rest import mixer_router
from pyplumber.rest import create_app, serve_in_thread

app = create_app(bridge, [mixer_router(bridge, on_aux_change=setup.remember_aux), my_router], token=token)
server = serve_in_thread(app, "127.0.0.1", 7682)
...
server.close()
```

`create_app(client, routers, token=None)` takes any object with
`command(line, timeout=None) -> str | None` as `client` (`ControlClient`).
Routers are `fastapi.APIRouter`s, mounted under `/api/v1`. They are additive: a
router that defines an operation the core or an earlier router serves makes
`create_app` raise `ValueError`, so a client written against the core and the
mixer router finds the same routes in every application. Application routers
get the token check, the cross-origin check and the error handlers; they raise
`pyplumber.rest.ApiError(status, code, message)` for answers of their own.

An application that starts the engine itself passes `starting`, a function that
returns a reason while it does and `None` otherwise, to `create_app` and to
`mixer_router`. An engine building its graph accepts the control connection and
answers nothing, so each request would wait for its timeout behind the ones
before it. With `starting`, `/engine` and `/mixer` answer 503
`engine_not_ready` with the reason and health reports it, at once; routers that
do not need the engine (the application's setup) stay served.

## Errors

Every error has one shape. Branch on the HTTP status and `code`; `message` is
for people.

```json
{"error": {"code": "engine_rejected", "message": "mixer: unknown scene 'cam9'"}}
```

| Status | `code` | Meaning |
| --- | --- | --- |
| 401 | `unauthorized` | The gateway requires a token; the request has none or a wrong one. |
| 403 | `cross_origin` | A browser sent the request for another site's page. |
| 404 | `not_found` | No such route in this gateway (e.g. `/mixer` without `--mixer`). |
| 404 | `not_available` | The engine has no such command: e.g. keys in a show without a keyer. |
| 409 | `engine_rejected` | The engine refused the command; `message` is its `500 ERROR` text. The control protocol does not classify refusals, so an unknown node or scene is a 409 too. |
| 409 | `conflict` | The resource changed since the revision the request names (`PUT /mixer/aux/{bus}/scenes`). Read it again. |
| 422 | `invalid_request` | The request does not match the schema; `details` lists `loc`, `type` and `msg` per field. Nothing was sent to the engine. |
| 500 | `internal_error` | The gateway failed; the cause is in its log, not in the answer. |
| 502 | `bad_engine_reply` | The engine's reply is not JSON, or lacks a field the route's schema promises (an engine of another version). |
| 503 | `engine_unreachable` | No control connection. The gateway reconnects on the next request, so it survives an engine restart. |
| 503 | `engine_not_ready` | The engine is starting and does not take this command yet; `message` says what it is doing. Try again. |
| 504 | `engine_timeout` | The engine did not answer in time. The command may still have run: read the state before repeating a take. |

`GET /api/v1/health` is always 200 while the gateway runs; its `status` is
`degraded` and `engine.reachable` false when the engine does not answer or is
starting (`engine.error` says which).

Request bodies are strict: an unknown field, a number as a string, or `1` for a
boolean is a 422. Replies carry the fields the schema names plus whatever else the
engine reports (the schema marks those objects `additionalProperties: true`).

## Engine

Read-only in this version. Commands that change the graph (`node.add`,
`node.object.set`, `group.*`, …) open files and URLs on the engine's host; they
stay on the control port.

| Operation | Control command |
| --- | --- |
| `GET /engine/version` | `version` |
| `GET /engine/nodes` | `nodes.json` |
| `GET /engine/nodes/{name}` | `node.param.get` |
| `GET /engine/nodes/{name}/objects/{object}` | `node.object.get` |
| `GET /engine/queues` | `queues.json` |

Node and object names are limited to `[A-Za-z0-9_.@:-]`, 1 to 128 characters,
because they are written into a control-protocol line.

## Mixer

`pyplumber.mixer.rest.mixer_router(bridge)`. The request and reply fields are in
the OpenAPI document; [mixer.md](mixer.md) describes what the commands do.

| Operation | Control command |
| --- | --- |
| `GET /mixer/status` | `mixer.status` |
| `GET /mixer/scenes` | `mixer.scenes` |
| `GET /mixer/settings` | `mixer.settings` |
| `PUT /mixer/preview` | `mixer.preview` |
| `POST /mixer/takes` | `mixer.cut`, `mixer.fade`, `mixer.wipe` or `mixer.move`, by `type` |
| `POST /mixer/transition/interrupt` | `mixer.interrupt` |
| `PUT /mixer/lever` | `mixer.lever` |
| `GET /mixer/keys` | `mixer.dsk_status` |
| `PATCH /mixer/keys` | `mixer.dsk` with `keys` |
| `PATCH /mixer/keys/{key}` | `mixer.dsk` with `key` |
| `PUT /mixer/keys/rate` | `mixer.dsk` with `all_rate` |
| `GET`, `PUT /mixer/ftb` | `mixer.ftb_status`, `mixer.ftb` |
| `GET /mixer/aux` | `mixer.aux_status` |
| `PUT /mixer/aux/{bus}/scenes` | `mixer.aux` |
| `PUT /mixer/aux/{bus}/layout` | `mixer.aux_layout` |
| `PUT /mixer/aux/{bus}/page` | `mixer.aux_page` |
| `PUT /mixer/aux/{bus}/composition` | `mixer.aux_composition` (external-mode buses) |
| `PUT /mixer/sources/{id}` | `mixer.source.set` |
| `POST /mixer/sources/{id}/prepare` | `mixer.source.prepare` |
| `POST /mixer/sources/{id}/media` | `mixer.source.media` |
| `POST /mixer/sources/{id}/resume_at` | `mixer.source.resume_at` |
| `GET`, `PATCH /mixer/audio/gains` | `mixer.audio_gain_status`, `mixer.audio_gain` |
| `GET /mixer/audio/processors` | `mixer.audio_processor_status` |
| `PATCH /mixer/audio/processors/{output}/{id}` | `mixer.audio_processor` |

`mixer.status`, `mixer.scenes` and the takes are the engine's; the key, fade to
black, AUX, source, audio and settings commands are registered by `pyplumber.mixer`, so
a mixer built without it answers those routes with 404 `not_available`.

While a take is on its way to Program (`transition` in the status is not
`idle`, some frames even for a cut), `PUT /mixer/preview` is refused with 409
`engine_rejected`, "transition already in progress": wait for `idle`.

Takes reach the mixer one at a time, in arrival order. A take that is still
waiting when a newer one arrives is not sent; its answer is
`{"superseded": true}`. The answer to a take says the mixer accepted it, not
that it is on Program: poll `GET /mixer/status` for `pgm_scene`.

```
curl -s http://127.0.0.1:7682/api/v1/mixer/status
curl -s -X PUT  http://127.0.0.1:7682/api/v1/mixer/preview -H 'Content-Type: application/json' -d '{"scene": "cam2"}'
curl -s -X POST http://127.0.0.1:7682/api/v1/mixer/takes   -H 'Content-Type: application/json' -d '{"type": "fade", "scene": "cam2", "duration_sec": 1.5}'
curl -s -X PATCH http://127.0.0.1:7682/api/v1/mixer/keys/logo -H 'Content-Type: application/json' -d '{"on": true}'
```

## Testing

* Swagger UI at `/api/v1/docs`: every operation with its schema and a form to
  send it. With a token, use its Authorize button.
* `python -m pyplumber.rest --fake`: the API without an engine or a GPU.
* `tools/rest_smoke.py [--url http://<host>:7682] [--token-env VAR] [--read-only]`:
  standard library only. It reads the engine and the mixer, previews and cuts to
  another scene, checks that the mixer reports each, and puts the original Program
  and Preview back. The cut is visible on Program: pass `--read-only` on a show
  that is on air.
* `pytest tests/rest` runs the gateway over the fake engine, without a network or
  a GPU. It is skipped where FastAPI is not installed.
