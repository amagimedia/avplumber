"""CPU-native BSF lifecycle regression; subprocess deadlines bound shutdown hangs."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


@pytest.mark.parametrize("case", [
    "unopened", "encoder_failure", "parse_failure", "init_failure", "initialized",
])
def test_bsf_shutdown(case, tmp_path):
    if importlib.util.find_spec("_avplumber") is None:
        pytest.skip("requires the native avplumber extension")
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), case, str(tmp_path / "output.nut")],
            capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(f"BSF {case} exceeded shutdown deadline: {error.stdout!r}{error.stderr!r}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PASS {case}" in result.stdout


def _run(case, output):
    from pyplumber import AVPlumber
    from pyplumber.node import AssumeVideoFormat, Bsf, EncVideo, ForceFPS, Mux, Output

    avp = AVPlumber()
    filters = {
        "parse_failure": "not_a_bitstream_filter",
        "init_failure": "h264_mp4toannexb",  # Rejects the MPEG-2 codec after allocation.
    }
    if case != "unopened":
        for cls, params in (
            (AssumeVideoFormat, {"name": "format", "src": "raw", "dst": "sized",
                                "width": 64, "height": 64}),
            (ForceFPS, {"name": "rate", "src": "sized", "dst": "frames", "fps": "25/1"}),
            (EncVideo, {"name": "encoder", "src": "frames", "dst": "encoded",
                        "codec": "mpeg2video", "options": {"qmin": "invalid_test_value"}
                        if case == "encoder_failure" else {}}),
        ):
            avp.addNode(cls(params), early_create=True)
    avp.addNode(Bsf({
        "name": "bsf", "src": "encoded", "dst": "filtered",
        "bsf": filters.get(case, "dump_extra=freq=keyframe"),
    }), early_create=True)
    if case != "unopened":
        avp.addNode(Mux({"name": "mux", "src": ["filtered"], "dst": "muxed"}),
                    early_create=True)
        failure = None
        try:
            avp.addNode(Output({"name": "output", "src": "muxed", "url": output}),
                        early_create=True)
        except RuntimeError as error:
            failure = str(error)
        if case == "initialized":
            assert failure is None, failure
        else:
            assert failure is not None, f"{case} did not fail initialization"
            expected = {
                "encoder_failure": "Invalid argument",
                "parse_failure": "Couldn't create BSF context",
                "init_failure": "Couldn't initialize BSF context",
            }[case]
            assert expected in failure, failure
            print(f"EXPECTED {case}: {failure}", flush=True)
    if case == "init_failure":
        errors = []
        avp.on_exception = lambda *error: errors.append(error)
        avp.node("bsf").start()
        deadline = time.monotonic() + 3
        while not errors and time.monotonic() < deadline:
            time.sleep(0.01)
        assert any("BSF context is not initialized" in error[-1] for error in errors), errors
        avp.node("bsf").join()
    # stop() flushes a created node even when no worker thread was started.
    # Both first-time cleanup and a repeated stop must complete normally.
    avp.node("bsf").stop(False)
    avp.node("bsf").stop(False)
    avp.shutdown()
    print(f"PASS {case}", flush=True)


if __name__ == "__main__":
    import traceback
    try:
        _run(*sys.argv[1:])
    except BaseException:
        traceback.print_exc()
        # Do not reenter the failing graph's destructor after a test assertion.
        os._exit(1)
