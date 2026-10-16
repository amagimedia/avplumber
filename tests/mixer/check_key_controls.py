"""Short NVIDIA check of rendered key tally and frame-synchronized key operations.

Uses the isolated 640x360 recovery graph, never the running demo's control port.
Pass two frame_codes.py clips. Status is sampled while draining actual output frames.
"""
import argparse
import json
import time

from check_transition_recovery import recovery_graph


def check(avp, mixer, output, errors):
    node = avp.node('test_keyer')

    def send(key, value):
        avp.executeCommandsFromString(f'node.object.set test_keyer {key} ' + json.dumps(value))

    def wait(predicate, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            output.get(100)
            rows = node.getObject('keys')
            assert not errors, errors
            if predicate(rows):
                return rows
        raise AssertionError(('key state did not arrive', rows))

    wait(lambda rows: all(r['phase'] == 'on' for r in rows))
    send('key_changes', {'inputs': 6, 'active_inputs': 1, 'duration_ms': 1000})
    rows = wait(lambda rows: all(r['phase'] == 'fading-out' and .3 < r['opacity'] < .7 for r in rows))
    assert all(not r['on'] and r['on_air'] and 0 < r['opacity'] < 1 for r in rows), rows
    assert rows[0]['opacity'] == rows[1]['opacity'] and rows[0]['pts_ns'] == rows[1]['pts_ns'], rows
    # Reverse one key while the other continues out; the latter must not restart.
    before = rows[1]['opacity']
    send('key_changes', {'inputs': 2, 'active_inputs': 3, 'duration_ms': 1000})
    rows = wait(lambda rows: rows[0]['phase'] == 'fading-in' and rows[1]['phase'] == 'fading-out')
    assert rows[1]['opacity'] < before, rows
    # Explicit cut to an already requested target completes that key's envelope.
    send('key_changes', {'inputs': 2, 'active_inputs': 3, 'duration_ms': 0})
    rows = wait(lambda rows: rows[0]['phase'] == 'on')
    assert rows[0]['opacity'] == 1, rows
    wait(lambda rows: rows[1]['phase'] == 'off')
    # Preparation does not alter air. Only a successfully started background take fires it.
    send('prepare_inputs', {'inputs': 6, 'duration_ms': 1000})
    rows = wait(lambda rows: all(r['next_take'] for r in rows))
    assert rows[0]['on'] and not rows[1]['on'], rows
    mixer.preview('full1')
    rows = wait(lambda rows: all(r['next_take'] for r in rows))
    assert rows[0]['phase'] == 'on' and rows[1]['phase'] == 'off', rows
    mixer.fade('full1', duration_sec=1)
    rows = wait(lambda rows: rows[0]['phase'] == 'fading-out' and rows[1]['phase'] == 'fading-in')
    assert abs(rows[0]['opacity'] + rows[1]['opacity'] - 1) < 1e-6, rows
    mixer.cut('full1')
    rows = wait(lambda rows: rows[0]['phase'] == 'off' and rows[1]['phase'] == 'on')
    assert not rows[0]['on_air'] and rows[1]['on_air'], rows
    # The persistent selection toggles again on the next cut, once, with no fade.
    mixer.cut('full0')
    wait(lambda rows: rows[0]['phase'] == 'on' and rows[1]['phase'] == 'off')
    print('PASS: rendered tally, atomic group, reversal, forced completion, preparation, fade and cut')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs=2)
    args = parser.parse_args()
    with recovery_graph(args.inputs, keyer=True) as graph:
        check(*graph)
