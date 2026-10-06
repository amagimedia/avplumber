"""Remote CUDA test of native-rate inputs into a 60 FPS mixer, with pixel frame IDs.

Runs the production mixer input chain; --baseline restores input force_fps.
Downloads are observation branches, never part of the GPU path to the encoders.
Outputs include raw observations so cadence claims can be independently checked.
"""
import argparse
from contextlib import nullcontext
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import threading
import time
from unittest.mock import patch

import numpy as np
from pyplumber.mixer.cli import GraphOptions, build_application, load_avp_api
from pyplumber.mixer.gui import MixerBridge
from pyplumber.mixer.inputs import _pace

RATES = ('25', '30', '30000/1001', '469/20', '50', '24000/1001', '60000/1001', '120')
W, H, FPS = 320, 180, 60


def fixture(root, rate):
    path = root / (rate.replace('/', '-') + '.mp4')
    count = round(float(Fraction(rate)) * 12)
    if not path.exists():
        # Each 20-pixel stripe is one bit of the source frame number.
        graph = f"nullsrc=s={W}x{H}:r={rate},geq=lum='32+176*mod(floor(N/pow(2,floor(X/20))),2)':cb=128:cr=128"
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', graph, '-frames:v', str(count),
                        '-c:v', 'libx264', '-threads', '2', '-preset', 'ultrafast', '-crf', '0',
                        '-pix_fmt', 'yuv420p', '-color_trc', 'bt709', '-color_primaries', 'bt709',
                        '-colorspace', 'bt709', str(path)], check=True)
    return path, count


def analyze(rows, counts):
    pts = np.array([r['pts'] for r in rows])
    ids = np.array([r['ids'] for r in rows])
    tick_gaps = np.rint(np.diff(pts) * FPS).astype(int)
    ages = [(r['arrival'] - r['pts']) * 1000 for r in rows]
    result = {'output_age_p95_ms': float(np.percentile(ages, 95)), 'frames': len(rows), 'seconds': float(pts[-1] - pts[0]),
              'tick_gaps': {str(k): int(v) for k, v in zip(*np.unique(tick_gaps, return_counts=True))}, 'sources': []}
    for i, (rate, count) in enumerate(zip(RATES, counts)):
        delta = np.diff(ids[:, i])
        delta[delta < -count / 2] += count
        unwrapped = np.r_[0, np.cumsum(delta)]
        phase = unwrapped / float(Fraction(rate)) - (pts - pts[0])
        observed = float(unwrapped[-1] / (pts[-1] - pts[0]))
        result['sources'].append({'rate': rate, 'observed_source_fps': observed,
                                  'repeats': int(np.sum(delta == 0)), 'backwards': int(np.sum(delta < 0)),
                                  'max_advance': int(delta.max()), 'phase_range_ms': float(np.ptp(phase) * 1000),
                                  'advance_histogram': {str(k): int(v) for k, v in zip(*np.unique(delta, return_counts=True))}})
    return result


def verify(report, rows, samples, *, strict):
    checks = {}
    for name, stream in report['streams'].items():
        checks[name + '/steady_ticks'] = stream['tick_gaps'] == {'1': stream['frames'] - 1}
        checks[name + '/output_age'] = stream['output_age_p95_ms'] < 100
        if strict:
            for source in stream['sources']:
                rate = float(Fraction(source['rate']))
                checks[name + '/' + source['rate']] = (
                    source['backwards'] == 0 and source['max_advance'] <= int(np.ceil(rate / FPS))
                    and source['phase_range_ms'] <= 1000 / rate + .1
                    and abs(source['observed_source_fps'] - rate) <= 2 / stream['seconds'])
    checks['bounded_queues'] = all(0 <= q['occupied'] <= q['capacity'] for sample in samples for q in sample['queues'])
    aux = {round(r['pts'] * FPS): r for r in rows['aux_mv_sdr']}
    for index, event in enumerate(report['events']):
        when = event['time']
        if 'cut' in event:
            offset = 1 if event['cut'] == 'rotate' else 0
            selected = [r for r in rows['scale_pgm'] if when + .2 < r['arrival'] < when + .45]
            matched = sum(all(r['ids'][(i + offset) % 8] == aux[round(r['pts'] * FPS)]['ids'][i]
                              for i in range(8)) for r in selected if round(r['pts'] * FPS) in aux)
            # The old normalized AUX path can select different frames; strict parity
            # applies to the native timestamp path under test.
            checks[f'cut_{index}'] = len(selected) >= 13 and (not strict or matched >= .9 * len(selected))
        elif 'stall' in event or 'resume' in event:
            stalled = 'stall' in event
            for name, data in rows.items():
                low, high = (.25, .85) if stalled else (1., 2.)
                selected = [r for r in data if when + low < r['arrival'] < when + high]
                ids = {r['ids'][0] for r in selected}
                checks[f'{name}/{index}'] = (len(selected) >= (high - low) * FPS - 2 and
                    (len(ids) == 1 and len({r['ids'][1] for r in selected}) >= 16 if stalled else len(ids) >= 24))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--seconds', type=float, default=40)
    parser.add_argument('--webui', default='http://127.0.0.1:22222')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    mode = 'normalized' if args.baseline else 'native'
    assets = [fixture(args.root, rate) for rate in RATES]
    tiles = [{'x': i % 4 * W, 'y': i // 4 * H, 'w': W, 'h': H} for i in range(len(RATES))]
    sources = [{'id': f's{i}', 'kind': 'video', 'path': str(p), 'color': 'sdr', 'loop': True,
                'decoder_params': {'pixel_format': 'cuarray'}} for i, (p, _) in enumerate(assets)]
    scenes = [{'id': label, 'items': [{'source': f's{i}', 'dst': tiles[(i + offset) % 8], 'fit': 'stretch'}
                                    for i in range(8)]} for label, offset in [('grid', 0), ('rotate', 1)]]
    config = {'canvas': {'width': W * 4, 'height': H * 2, 'fps': FPS}, 'sources': sources,
              'scenes': scenes, 'initial_scene': 'grid',
              'renditions': [{'id': 'pgm', 'target': 'janus', 'port': 15404}],
              'aux_buses': [{'id': 'mv', 'full_rate': True,
                             'layout': {'cells': [{'role': 'source', 'source': i, **tiles[i]} for i in range(8)]},
                             'renditions': [{'id': 'monitor', 'port': 15408}]}]}
    path = args.root / (mode + '-config.json')
    path.write_text(json.dumps(config))
    api = load_avp_api()
    factory = api.FilterVideo
    taps = []
    def filter_video(params):
        name = params['name']
        if name in ('scale_pgm', 'aux_mv_sdr'):
            taps.append((name, params['src'], params['group'], params['hwaccel']))
            params = {**params, 'src': name + '_tap_forward'}
        return factory(params)
    api.FilterVideo = filter_video
    with patch('pyplumber.mixer.cli.load_avp_api', return_value=api), \
         (patch('pyplumber.mixer.inputs._pace', lambda *a, **k: _pace(*a, **{**k, 'native_rate': False}))
          if args.baseline else nullcontext()):
        app = build_application(GraphOptions(config=str(path), janus_output=True,
                                             remote_control_port=18778, prewarm_cut_scenes=('*',)))
    assert len(taps) == 2, taps
    rows = {name: [] for name, *_ in taps}
    stop = threading.Event()
    errors = []
    app.avp.on_exception = lambda *e: errors.append(tuple(map(str, e)))
    threads = []
    def drain(name, edge):
        try:
            while not stop.is_set():
                f = edge.tryGet(100)
                if f is None or f.pts.timestamp == -(1 << 63):
                    continue
                plane = np.frombuffer(f.data[0], np.uint8).reshape(f.height, f.linesize[0])
                ids = [sum(int(plane[t['y'] + 30, t['x'] + bit * 20 + 10] > 120) << bit for bit in range(16)) for t in tiles]
                rows[name].append({'arrival': time.monotonic(), 'pts': f.pts.timestamp * f.pts.timebase.num / f.pts.timebase.den, 'ids': ids})
        except Exception as e:
            errors.append(str(e))
    for name, src, group, hw in taps:
        app.avp.addNode(api.Split({'name': name + '_tap', 'src': src,
                                  'dst': [name + '_tap_forward', name + '_tap_read'], 'group': group}))
        app.avp.addNode(factory({'name': name + '_download', 'src': name + '_tap_read', 'dst': name + '_cpu',
                                'graph': 'scale_cuda=passthrough=0,hwdownload,format=nv12', 'hwaccel': hw, 'threads': 1, 'group': group}))
        t = threading.Thread(target=drain, args=(name, app.avp.getEdge(name + '_cpu', 'VideoFrame')), daemon=True)
        t.start(); threads.append(t)
    app.avp.registerWithWebUI(args.webui, 'mixed-fps-' + mode, '')
    samples, events = [], []
    bridge = MixerBridge('127.0.0.1', 18778, 'mixer')
    try:
        app.start()
        nodes = json.loads(bridge.command('nodes.json'))
        input_fps_nodes = [n['name'] for n in nodes if n['type'] == 'force_fps' and n['params'].get('group', '').startswith('input_')]
        assert len(input_fps_nodes) == (8 if args.baseline else 0), input_fps_nodes
        time.sleep(3)
        begin = time.monotonic()
        while time.monotonic() - begin < args.seconds:
            assert not errors, errors
            samples.append({'time': time.monotonic(), 'queues': json.loads(bridge.command('queues.json')),
                            'status': bridge.status()})
            app.avp.heartbeat()
            time.sleep(1)
        end = time.monotonic()
        for scene in ('rotate', 'grid', 'rotate', 'grid'):
            app.mixer.cut(scene); events.append({'time': time.monotonic(), 'cut': scene}); time.sleep(.5)
        node = app.avp.node('realtime_0')
        node.stopAndWait(); events.append({'time': time.monotonic(), 'stall': 0})
        time.sleep(1)
        node.start(); events.append({'time': time.monotonic(), 'resume': 0})
        time.sleep(3)
        report = {'input_force_fps_nodes': input_fps_nodes, 'mode': mode, 'errors': errors, 'steady_begin': begin, 'steady_end': end, 'events': events,
                  'streams': {name: analyze([r for r in data if begin <= r['arrival'] <= end], [n for _, n in assets])
                              for name, data in rows.items()}}
        report['checks'] = verify(report, rows, samples, strict=not args.baseline)
        (args.root / (mode + '-report.json')).write_text(json.dumps(report, indent=2))
        (args.root / (mode + '-queues.json')).write_text(json.dumps(samples))
        for name, data in rows.items():
            (args.root / (mode + '-' + name + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in data))
        print(json.dumps(report, indent=2), flush=True)
        assert not errors, errors
        assert all(report['checks'].values()), report['checks']
    finally:
        app.stop()
        stop.set()
        for t in threads: t.join(2)


if __name__ == '__main__':
    main()
