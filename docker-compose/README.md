# Docker Compose Runner

This directory owns demo orchestration only. It keeps Janus, web UI, DMA browser,
and avplumber runners in one stack without making auto-mixer or neural-demo own
the compose file.

The generic `runner` service runs any mounted `.avplumber` script. Existing
`neural-demo` and `auto-mixer` entrypoints are still available as separate
profiles for the full basketball and talkshow demos.

## Layout

- `docker-compose.yml` - shared support services and selectable runners.
- `run-demo.sh` - small wrapper around `docker compose`.
- `.env.example` - all common knobs in one place.
- `demos/*.env.example` - starting points for common demo choices.
- `scripts/run-avplumber-script.sh` - entrypoint for a custom `.avplumber` graph.

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

## Full Basketball Demo

```sh
cp docker-compose/demos/basketball-full.env.example docker-compose/demos/basketball-full.env
docker-compose/run-demo.sh --env docker-compose/demos/basketball-full.env --profile neural up
```

Set `NEURAL_DEMO_INPUT` to the host media file. The container sees it at
`/run/avp/input/input`.

## TrackNet Ball Script

```sh
cp docker-compose/demos/tracknet-ball.env.example docker-compose/demos/tracknet-ball.env
docker-compose/run-demo.sh --env docker-compose/demos/tracknet-ball.env --profile script up
```

This is still the generic script runner. The env file only points at the
mounted graph, media directory, model directory, and artifact directory.

## Talkshow / Auto Mixer Demo

```sh
cp docker-compose/demos/talkshow-6cam.env.example docker-compose/demos/talkshow-6cam.env
docker-compose/run-demo.sh --env docker-compose/demos/talkshow-6cam.env --profile auto-mixer up
```

Set `MEDIA_INPUT_DIR` to the host directory containing the six talkshow inputs.
The container sees them under `/media-inputs`, so `AUTO_MIXER_INPUTS` should use
that path.

## Support Services Only

```sh
docker-compose/run-demo.sh --env docker-compose/.env up -d
```

The web UI defaults to `http://127.0.0.1:22222`, Janus HTTP to
`http://127.0.0.1:8088/janus`, and the preview server to
`http://127.0.0.1:8080`.
