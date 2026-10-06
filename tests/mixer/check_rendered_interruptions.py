"""NVIDIA integration: immediate takes, transition replacements and live endpoints.

Uses numbered CUDA sources and downloads only the assertion output. It checks
rendered source identities and motion after each take, without future deadlines.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import time

import numpy as np

from frame_codes import read_code


def assert_frozen(reference, frames):
    if len(frames) < 8:
        raise AssertionError("fewer than eight retained output frames")
    for index, image in enumerate(frames):
        if not np.array_equal(image, reference):
            raise AssertionError(f"retained picture differs at frame {index}")


def check_graph(avp, mixer, output, errors, wipe_file, wipe_seconds):
    last_pts = None
    def receive(timeout=50):
        nonlocal last_pts
        try:
            frame = output.get(timeout)
        except ValueError as error:
            if str(error) == 'get: timeout':
                return None
            raise
        assert (frame.width, frame.height) == (640, 360)
        pts = Fraction(frame.pts.timestamp * frame.pts.timebase.num, frame.pts.timebase.den)
        assert last_pts is None or pts >= last_pts, ('output timestamp went backwards', last_pts, pts)
        last_pts = pts
        image = np.frombuffer(frame.data[0], np.uint8).reshape(360, frame.linesize[0])[:, :640].copy()
        return time.monotonic(), image

    def collect(seconds, *, allow_wipe_cancel=False):
        frames = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if sample := receive():
                frames.append(sample)
        if allow_wipe_cancel:
            # NodeGroup reports NotReallyError through the same callback as
            # failures when a take cancels an in-progress decoder-group start.
            # Accept only that notification, only during deliberate cancellation;
            # the caller still checks live pixels and the replacement endpoint.
            cancelled = ('recovery_wipe', 'NodeGroup',
                         'Error while changing state: Another start of the group requested')
            errors[:] = [error for error in errors if error != cancelled]
        assert not errors, errors
        return frames

    def take(kind, scene):
        if kind == 'cut':
            mixer.cut(scene)
        elif kind == 'fade':
            mixer.fade(scene, duration_sec=1.2)
        else:
            mixer.wipe(scene, wipe_file, duration_sec=wipe_seconds)

    def live_scene(source, seconds=0.4):
        frames = collect(seconds)
        codes = [read_code(image) for _, image in frames[-12:]]
        assert len(codes) >= 8 and all(code and code[0] == source for code in codes), codes
        assert len({code[1] for code in codes}) >= 6, 'settled program does not advance'
        return frames[-1][1]

    # Observe both real endpoints; they also establish that the test inputs run.
    collect(0.5)
    mixer.cut('full1')
    collect(0.5)
    endpoint1 = live_scene(1)
    mixer.cut('full0')
    collect(0.5)
    endpoint0 = live_scene(0)
    results = []
    for first in ('cut', 'fade', 'media_wipe'):
        for second in ('cut', 'fade', 'media_wipe'):
            for phase in (('before', 'after') if first == 'media_wipe' else (None,)):
                mixer.cut('full0')
                collect(0.5)
                live_scene(0)
                take(first, 'full1')
                if first == 'media_wipe':
                    # Use visible pixels to distinguish the two sides of the
                    # media wipe, not an assumed decoder startup duration.
                    endpoint = endpoint0 if phase == 'before' else endpoint1
                    deadline = time.monotonic() + wipe_seconds + 3
                    while time.monotonic() < deadline:
                        sample = receive()
                        if sample is None:
                            continue
                        equal_fraction = float(np.mean(np.abs(
                            sample[1].astype(np.int16) - endpoint.astype(np.int16)) <= 1))
                        if 0.25 < equal_fraction < 0.75:
                            break
                    else:
                        raise AssertionError(f'could not observe partial media wipe {phase} midpoint')
                else:
                    sample = collect(0.55 if first == 'fade' else 0.1)[-1]
                started = time.monotonic()
                take(second, 'full0')
                acknowledged = time.monotonic()
                following = collect(max(2.0, wipe_seconds + 1), allow_wipe_cancel=first == 'media_wipe')
                assert len(following) >= 60, 'replacement stalled program output'
                live_scene(0)
                # A cancelled transition cannot reappear after its former end.
                live_scene(0)
                result = {'first': first, 'second': second, 'wipe_phase': phase,
                          'checked_frames': len(following),
                          'command_ms': (acknowledged - started) * 1000}
                results.append(result)
                print(json.dumps(result), flush=True)
    # An explicit interruption retains a blended picture until a replacement
    # arrives. Check pixels, then prove that the replacement restores motion.
    mixer.fade('full1', duration_sec=1.2)
    collect(0.55)
    avp.executeCommandsFromString('mixer.interrupt {"mixer":"recovery"}')
    retained = collect(0.5)
    assert_frozen(retained[-12][1], [image for _, image in retained[-12:]])
    for index in range(20):
        mixer.cut(f'full{index % 2}')
    collect(0.5)
    live_scene(1)
    results.append({'explicit_interrupt': 'retained blend', 'burst_cuts': 20, 'final_scene': 'full1'})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs=2)
    parser.add_argument('--wipe-file', required=True)
    parser.add_argument('--wipe-seconds', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--webui')
    parser.add_argument('--port', type=int, default=18779)
    args = parser.parse_args()
    if not np.isfinite(args.wipe_seconds) or args.wipe_seconds <= 0:
        parser.error('--wipe-seconds must be finite and positive')
    from check_transition_recovery import recovery_graph
    with recovery_graph(args.inputs, webui=args.webui, port=args.port) as graph:
        results = check_graph(*graph, args.wipe_file, args.wipe_seconds)
    args.output.write_text(json.dumps({'passed': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
