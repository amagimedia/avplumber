"""Filter commands from another thread while the node filters frames, as lever moves send transition_cuda's alpha."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time

import pytest


def test_filter_command_beside_filtering(run_script):
    if importlib.util.find_spec('_avplumber') is None or not shutil.which('ffmpeg'):
        pytest.skip('requires the native extension and ffmpeg')
    result = run_script(Path(__file__).resolve(), timeout=60)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert 'PASS filter command' in result.stdout


def run_case():
    from pyplumber import AVPlumber
    from pyplumber.node import NodeBase

    with tempfile.TemporaryDirectory() as directory:
        source = str(Path(directory) / 'frames.mkv')
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=size=32x32:rate=25',
                        '-frames:v', '50000', '-c:v', 'ffv1', source], check=True)
        avp = AVPlumber()
        errors = []
        avp.on_exception = lambda *error: errors.append(error)
        graph = [
            dict(type='input', name='input', url=source, dst='packets'),
            dict(type='demux', name='demux', src='packets', routing={'v:0': 'video'}),
            dict(type='dec_video', name='decoder', src='video', dst='frames'),
            # hue evaluates h for every frame and a command re-parses it, as transition_cuda does with alpha.
            dict(type='filter_video', name='filter', src='frames', dst='output', graph='hue=h=t*10', threads=1),
        ]
        for params in graph:
            avp.addNode(NodeBase(params), early_create=True)
        output = avp.getEdge('output', 'VideoFrame')
        for params in reversed(graph):
            avp.node(params['name']).start()
        output.get(5000)   # the filter graph exists

        stop = threading.Event()

        def send_commands():
            for i in range(1 << 62):
                if stop.is_set():
                    return
                avp.executeCommandsFromString(
                    f'node.object.set filter filter_command {{"target":"hue","command":"h","argument":"t*{i % 50}"}}')

        sender = threading.Thread(target=send_commands)
        sender.start()
        frames, deadline = 0, time.monotonic() + 3
        try:
            while time.monotonic() < deadline:
                output.get(5000)
                frames += 1
        finally:
            stop.set()
            sender.join()
        assert frames > 1000, frames
        assert not errors, errors
        avp.shutdown()
    print(f'PASS filter command ({frames} frames)', flush=True)


if __name__ == '__main__':
    run_case()
