"""NVIDIA regression: an uncached wipe must render its fading tail after input EOF.

Provide a black 640x360 video and a two-second white RGBA wipe that fades to
transparent over its final second (alpha: 255*clip(min(T/0.8,(1.92-T)/0.8),0,1),
QTRLE/ARGB at 25 or 60 fps). Uses the existing GPU graph with CPU readback
only on the pixel-assertion output, without changing a running demo.
"""
import argparse
import json
import time

import numpy as np

from check_transition_recovery import recovery_graph


def check_tail(avp, mixer, output, errors, wipe):
    def sample(seconds):
        levels = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                frame = output.get(50)
            except ValueError as error:
                if str(error) == 'get: timeout':
                    continue
                raise
            luma = np.frombuffer(frame.data[0], np.uint8).reshape(frame.height, frame.linesize[0])
            levels.append(float(luma[:, :frame.width].mean()))
        assert not errors, errors
        return levels

    baseline = sample(.5)
    assert baseline and max(abs(level - 16) for level in baseline) < 2, 'input must be black'
    mixer.wipe('full1', wipe, duration_sec=2)
    levels = np.array(sample(3.5))
    opaque = np.flatnonzero(levels > 220)
    assert len(opaque) >= 4, 'wipe never fully covered the program'
    tail = levels[opaque[-1] + 1:]
    visible = tail[tail > 18]
    assert len(visible) >= 10, 'wipe exit animation was truncated'
    assert visible[-1] < 35, f'wipe switched off at luma {visible[-1]:.1f}, before fading out'
    assert np.max(np.abs(np.diff(np.r_[235, tail]))) < 25, 'wipe exit contains an abrupt alpha jump'
    assert np.all(np.abs(levels[-12:] - 16) < 2), 'wipe did not return to the black program'
    mixer.wipe('full0', wipe, duration_sec=2)
    assert max(sample(.5)) > 30, 'second wipe did not start'
    mixer.cut('full1')
    cancelled = sample(.5)
    assert len(cancelled) >= 8 and max(abs(level - 16) for level in cancelled[-8:]) < 2, \
        'waiting for the tail blocked a replacement cut'
    result = {'visible_exit_frames': len(visible), 'last_visible_luma': float(visible[-1]),
              'largest_exit_luma_step': float(np.max(np.abs(np.diff(np.r_[235, tail]))))}
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('black_video')
    parser.add_argument('wipe')
    args = parser.parse_args()
    with recovery_graph([args.black_video, args.black_video]) as graph:
        check_tail(*graph, args.wipe)


if __name__ == '__main__':
    main()
