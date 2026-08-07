# Docker Compose Runner

This directory owns shared demo orchestration. It keeps Janus, web UI, DMA
browser, and AVPlumber runners in one stack.

The generic `runner` service runs any mounted `.avplumber` script. Application
and sport-specific stacks belong in their respective downstream repositories.

## Layout

- `docker-compose.yml` - shared support services and selectable runners.
- `run-demo.sh` - small wrapper around `docker compose`.
- `.env.example` - all common knobs in one place.
- `demos/*.env.example` - starting points for common demo choices.
- `scripts/run-avplumber-script.sh` - entrypoint for a custom `.avplumber` graph.
- `images/` - Dockerfiles for the shared support services.

## Custom Script

Copy and edit an env file:

```sh
cp docker-compose/demos/custom-script.env.example docker-compose/demos/custom-script.env
```

Set host paths in the env file:

- `AVP_SCRIPT_FILE` is the host `.avplumber` script to run.
- `AVP_MEDIA_DIR` is mounted read-only at `/media`.
- `AVP_MODELS_DIR` is mounted read-only at `/models`.
- `AVP_ARTIFACT_DIR` is mounted read-write at `/artifacts`.

Run it:

```sh
docker-compose/run-demo.sh --env docker-compose/demos/custom-script.env --profile script up
```

## Support Services Only

```sh
docker-compose/run-demo.sh --env docker-compose/.env up -d
```

The web UI defaults to `http://127.0.0.1:22222`, Janus HTTP to
`http://127.0.0.1:8088/janus`, and the preview server to
`http://127.0.0.1:8080`.
