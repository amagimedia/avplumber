"""NVIDIA integration: interrupting a take keeps the right picture on air.

Interrupting a crossfade retains the actual last output, blend included.
Interrupting a media wipe returns to the live program instead: the wipe graphic
belongs to the cancelled transition, and freezing it painted the graphic into
the program, so every further wipe composited over the last one and they stacked.

Use the two numbered 640x360/60 fixtures from frame_codes.py and a transparent
media wipe. The owned native mixer downloads its final output for assertions;
no running demo, reconstructed TUI image or status-only oracle is involved.
Replacement takes start 750 ms in the future so their retained picture can be
checked before they become visible. Immediate-take responsiveness is measured
separately by measure_click_latency.cjs.
Every luma pixel is compared exactly; this test does not compare chroma.

The reference is the last output observed immediately before the command, never
a matching frame selected afterwards. If the producer advances across that
boundary, the attempt fails as inconclusive rather than claiming an exact match.
"""
import argparse
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
    def receive(timeout=50):
        try:
            frame = output.get(timeout)
        except ValueError as error:
            if str(error) == 'get: timeout':
                return None
            raise
        assert (frame.width, frame.height) == (640, 360)
        image = np.frombuffer(frame.data[0], np.uint8).reshape(360, frame.linesize[0])[:, :640].copy()
        return time.monotonic(), image

    def collect(seconds):
        frames = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if sample := receive():
                frames.append(sample)
        assert not errors, errors
        return frames

    def take(kind, scene, at=-1):
        if kind == 'cut':
            mixer.cut(scene, start_pts_ms=at)
        elif kind == 'fade':
            mixer.fade(scene, duration_sec=1.2, start_pts_ms=at)
        else:
            mixer.wipe(scene, wipe_file, duration_sec=wipe_seconds, start_pts_ms=at)

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
                at = time.monotonic_ns() // 1000000 + 1500 if first == 'cut' else -1
                take(first, 'full1', at)
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
                # Empty the readback queue without waiting for a future frame.
                while newer := receive(0):
                    sample = newer
                received, reference = sample
                if first != 'cut':
                    for endpoint in (endpoint0, endpoint1):
                        difference = np.abs(reference.astype(np.int16) - endpoint.astype(np.int16)).mean()
                        assert difference > 8, 'reference is an endpoint, not a partial blend/overlay'
                started = time.monotonic()
                assert started - received < 0.008, 'reference collection was delayed; retry on an idle host'
                take(second, 'full0', time.monotonic_ns() // 1000000 + 750)
                acknowledged = time.monotonic()
                assert acknowledged - started < 0.008, 'command crossed capture boundary; correspondence uncertain'
                # Check every following output, including the boundary. If an
                # unseen in-flight frame advanced during the command, this
                # fails rather than selecting a convenient later reference.
                following = collect(0.5)
                held = [image for _, image in following]
                if first == 'media_wipe':
                    expected = 0 if phase == 'before' else 1
                    codes = [read_code(image) for image in held[-12:]]
                    assert len(codes) >= 8 and all(code and code[0] == expected for code in codes), codes
                    assert len({code[1] for code in codes}) >= 6, 'program did not resume after the interruption'
                else:
                    assert_frozen(reference, held)
                # Let both new and cancelled callbacks expire, then inspect
                # actual source IDs and progression, not only mixer.status.
                collect(max(2.0, wipe_seconds + 1))
                live_scene(0)
                result = {'first': first, 'second': second, 'wipe_phase': phase,
                          'checked_held_frames': len(held),
                          'command_ms': (acknowledged - started) * 1000}
                results.append(result)
                print(json.dumps(result), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs=2)
    parser.add_argument('--wipe-file', required=True)
    parser.add_argument('--wipe-seconds', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.wipe_seconds) or args.wipe_seconds <= 0:
        parser.error('--wipe-seconds must be finite and positive')
    from check_transition_recovery import recovery_graph
    with recovery_graph(args.inputs) as graph:
        results = check_graph(*graph, args.wipe_file, args.wipe_seconds)
    args.output.write_text(json.dumps({'passed': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
