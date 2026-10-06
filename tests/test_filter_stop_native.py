"""Stop a filter waiting for an input's first frame, with another input queued."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import pytest


def test_filter_stop_with_pending_input():
    if importlib.util.find_spec('_avplumber') is None or not shutil.which('ffmpeg'):
        pytest.skip('requires the native extension and ffmpeg')
    result = subprocess.run([sys.executable, str(Path(__file__).resolve())],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'did not finish after stop' not in result.stdout + result.stderr
    assert 'PASS filter stop' in result.stdout


def run_case():
    from pyplumber import AVPlumber
    from pyplumber.node import NodeBase

    with tempfile.TemporaryDirectory() as directory:
        source = str(Path(directory) / 'frames.mkv')
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                        'color=size=32x32:rate=25', '-frames:v', '1000',
                        '-c:v', 'ffv1', source], check=True)
        avp = AVPlumber()
        errors = []
        avp.on_exception = lambda *error: errors.append(error)
        graph = [
            dict(type='input', name='input', url=source, dst='packets'),
            dict(type='demux', name='demux', src='packets', routing={'v:0': 'video'}),
            dict(type='dec_video', name='decoder', src='video', dst='frames'),
            dict(type='filter_video', name='filter', src=['frames', 'missing'],
                 dst='output', graph='[in0][in1]hstack[out]', defer_preliminary_init=True),
        ]
        for params in graph:
            avp.addNode(NodeBase(params), early_create=True)
        for params in reversed(graph):
            avp.node(params['name']).start()
        frames = avp.getEdge('frames', 'VideoFrame')
        deadline = time.monotonic() + 5
        while frames.occupied < 2:
            assert time.monotonic() < deadline, errors
            time.sleep(0.01)
        # The first pad is available, but the second never supplies parameters.
        # Stop must wake that wait and flush, without processing/waiting again.
        time.sleep(0.1)
        avp.node('filter').stopAndWait()
        assert not errors, errors
        avp.shutdown()
    print('PASS filter stop', flush=True)


if __name__ == '__main__':
    run_case()
