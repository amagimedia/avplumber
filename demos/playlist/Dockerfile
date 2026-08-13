# syntax=docker/dockerfile:1

ARG AVP_BASE_IMAGE=avplumber-python:latest

FROM alpine:3.21 AS fixtures

RUN apk add --no-cache bash ffmpeg font-dejavu

WORKDIR /fixtures
COPY test-media/generate.sh ./generate.sh
RUN ./generate.sh


FROM python:3.13-alpine AS python-dependencies

COPY requirements.txt /requirements.txt
RUN python3 -m pip install --no-cache-dir \
        --target /opt/playlist-python \
        --requirement /requirements.txt


FROM ${AVP_BASE_IMAGE} AS runtime

ENV PYTHONPATH=/opt/playlist-python:${PYTHONPATH}
WORKDIR /opt/avplumber/demos/playlist

COPY --from=python-dependencies /opt/playlist-python /opt/playlist-python
COPY player.py playlist.py playlist_app.py regression.py requirements.txt README.md ./
COPY test-media/generate.sh ./test-media/generate.sh
COPY --from=fixtures /fixtures/*.mp4 ./test-media/

RUN python3 -c 'from importlib.util import find_spec; from pathlib import Path; import textual; from player import FIXTURE_NAMES; root = Path("test-media"); paths = [root / name for name in FIXTURE_NAMES]; assert find_spec("pyplumber") is not None; assert len(paths) == 5 and all(path.is_file() for path in paths), paths'

ENTRYPOINT ["python3", "player.py"]
CMD ["--janus-host", "127.0.0.1", "--janus-video-port", "5004", "--log-file", "/tmp/playlist-demo.log"]
