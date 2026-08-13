"""Persistent playlist fixture image contract."""

from pathlib import Path


DEMO_DIR = Path(__file__).parents[1]
DOCKERFILE = DEMO_DIR / "Dockerfile"


def test_image_generates_fixtures_in_a_disposable_stage():
    source = DOCKERFILE.read_text()
    assert "FROM alpine:3.21 AS fixtures" in source
    assert "COPY test-media/generate.sh ./generate.sh" in source
    assert "RUN ./generate.sh" in source
    assert "COPY --from=fixtures /fixtures/*.mp4 ./test-media/" in source
    assert "COPY test-media/*.mp4" not in source


def test_runtime_uses_configurable_avplumber_base_and_packaged_dependencies():
    source = DOCKERFILE.read_text()
    assert "ARG AVP_BASE_IMAGE" in source
    assert "FROM ${AVP_BASE_IMAGE} AS runtime" in source
    assert "COPY --from=python-dependencies /opt/playlist-python" in source
    assert "ENV PYTHONPATH=/opt/playlist-python:${PYTHONPATH}" in source
    assert 'find_spec("pyplumber")' in source
    assert "import pyplumber" not in source
    assert 'ENTRYPOINT ["python3", "player.py"]' in source
    assert '"--janus-host", "127.0.0.1"' in source
    assert '"--janus-video-port", "5004"' in source
    assert '"--log-file", "/tmp/playlist-demo.log"' in source


def test_build_context_excludes_local_artifacts_and_readme_documents_image():
    ignored = (DEMO_DIR / ".dockerignore").read_text().splitlines()
    for pattern in ("test-media/*.mp4", "test-media/*.tmp", "__pycache__/",
                    ".pytest_cache/"):
        assert pattern in ignored

    readme = (DEMO_DIR / "README.md").read_text()
    assert "AVP_BASE_IMAGE=<cuda-python-avplumber-image>" in readme
    assert "docker build" in readme
    assert "--gpus all" in readme
    assert "--network host" in readme
