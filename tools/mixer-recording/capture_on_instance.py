"""Record 20 seconds of the 192-input 30 fps show, including two manual T-bar strokes.

Run on the mixer host: python3 capture_on_instance.py > capture.json.
Janus records the existing Program and Program/Preview RTP inputs before Internet transport.
The report contains private paths and full state: publish only a sanitized summary.
"""
import json
import threading
import time
import urllib.request
import uuid

base = 'http://127.0.0.1:17681'
janus = 'http://127.0.0.1:8088/janus'
prefix = '/tmp/avplumber-native-' + str(int(time.time()))
FPS, DURATION = 30, 20
report = {'files': {}, 'states': [], 'statuses': [], 'events': [], 'fps': FPS, 'frames': FPS * DURATION}


def request(url, body=None):
    payload = None if body is None else json.dumps(body).encode()
    with urllib.request.urlopen(urllib.request.Request(url, payload, {'Content-Type': 'application/json'}), timeout=8) as response:
        return json.load(response)


def call(url, body):
    data = request(url, {**body, 'transaction': uuid.uuid4().hex})
    if data.get('janus') != 'success':
        raise RuntimeError(data)
    return data


def command(body):
    if body['command'] not in ('dsk', 'cut', 'preview', 'lever'):
        raise ValueError(body)
    started = time.time()
    result = request(base + '/api/command', body)
    if not result.get('ok'):
        raise RuntimeError(result)
    report['events'].append({'wall': started, 'done': time.time(), 'command': body})


def snapshot():
    wall = time.time()
    state = request(base + '/api/state')
    report['states'].append({'wall': wall, 'received': time.time(), 'state': state})
    return state


def counters(state):
    return {'playout': state['status']['playout'],
            'aux_missed': sum(bus['playout']['missed_deadlines'] for bus in state['aux_buses']),
            'aux_drops': sum(bus['output_drops'] for bus in state['aux_buses'])}


original = snapshot()
settings = original['settings']
assert settings['source_count'] == 192 and settings['canvas']['fps'] == FPS, 'Restore the 192-input 30 fps show first'
assert len(original['aux_buses']) == 22 and len(original['scenes']) == 256
assert not original['status'].get('transition_manual'), 'A manual transition is already in progress'
report['original'] = original
report['counters_before'] = counters(original)
session = janus + '/' + str(call(janus, {'janus': 'create'})['data']['id'])
handle = session + '/' + str(call(session, {'janus': 'attach', 'plugin': 'janus.plugin.streaming'})['data']['id'])


def streaming(body):
    data = call(handle, {'janus': 'message', 'body': body})['plugindata']['data']
    if 'error' in data:
        raise RuntimeError(data)
    return data


stop = threading.Event()
poll_errors = []


def poll():
    tick = time.monotonic()
    count = 0
    try:
        while not stop.is_set():
            wall = time.time()
            status = request(base + '/api/status')
            report['statuses'].append({'wall': wall, 'received': time.time(), 'status': status})
            if count % FPS == 0:
                snapshot()
            count += 1
            tick += 1 / FPS
            stop.wait(max(0, tick - time.monotonic()))
    except Exception as exc:
        poll_errors.append(str(exc))


client, seq = uuid.uuid4().hex, 0


def position(value):
    global seq
    seq += 1
    command({'command': 'lever', 'position': value, 'duration_sec': 1, 'curve': 'linear',
             'color': None, 'wipe_file': None, 'client': client, 'seq': seq, 't_ms': time.monotonic() * 1000})


started, worker = [], None
try:
    for mountpoint, name in ((1, 'pgm'), (5008, 'preview')):
        info = streaming({'request': 'info', 'id': mountpoint})['info']
        report[name + '_info'] = info
        if info.get('recording') or any(media.get('recording') for media in info['media']):
            raise RuntimeError('Stream is already being recorded')
        media = next(media for media in info['media'] if media['type'] == 'video')
        path = prefix + '-' + name
        streaming({'request': 'recording', 'action': 'start', 'id': mountpoint,
                   'media': [{'mid': media['mid'], 'filename': path}]})
        started.append((mountpoint, media['mid']))
        report['files'][name] = path + '.mjr'
    command({'command': 'dsk', 'keys': [key['id'] for key in original['dsk']], 'on': True, 'fade_seconds': 0})
    position(0)
    command({'command': 'cut', 'scene': 'grid_16_080'})
    time.sleep(0.3)
    command({'command': 'preview', 'scene': 'grid_4_048'})
    # Preroll includes complete keyframes and settled keys.
    time.sleep(5)
    worker = threading.Thread(target=poll)
    worker.start()
    report['demo_start_wall'] = time.time()
    start = time.monotonic()

    def wait(at):
        time.sleep(max(0, start + at - time.monotonic()))

    for at, scene in ((1, 'random_180'), (3, 'grid_64_112'), (5, 'pip_128')):
        wait(at)
        command({'command': 'cut', 'scene': scene})
    wait(6)
    command({'command': 'preview', 'scene': 'alpha_overlay_224'})
    # Half a stroke, a one-second hold, then completion. Reverse on the next scene.
    for at, duration, first, last in ((7, 2, 0, 0.5), (10, 2, 0.5, 1), (14, 3, 1, 0)):
        if at == 14:
            wait(13)
            command({'command': 'preview', 'scene': 'grid_8_064'})
        for frame in range(duration * FPS + 1):
            wait(at + frame / FPS)
            position(first + (last - first) * frame / (duration * FPS))
    wait(18)
    command({'command': 'cut', 'scene': 'random_179'})
    wait(DURATION)
    report['demo_end_wall'] = time.time()
    snapshot()
    time.sleep(2)
finally:
    stop.set()
    if worker:
        worker.join()
    errors = list(poll_errors)
    for mountpoint, media in started:
        try:
            streaming({'request': 'recording', 'action': 'stop', 'id': mountpoint, 'media': [{'mid': media}]})
        except Exception as exc:
            errors.append(str(exc))
    # A partial stroke must end before restoring Preview; cuts settle on the next video frame.
    restore = [{'command': 'cut', 'scene': original['status']['pgm_scene']},
               {'command': 'preview', 'scene': original['status']['pvw_scene']},
               *[{'command': 'dsk', 'key': key['id'], 'on': key['on'], 'fade_seconds': 0} for key in original['dsk']]]
    for body in restore:
        try:
            command(body)
            time.sleep(0.2)
        except Exception as exc:
            errors.append(str(exc))
    report['restored'] = snapshot()
    report['counters_after'] = counters(report['restored'])
    call(session, {'janus': 'destroy'})
    report['cleanup_errors'] = errors
    print(json.dumps(report), flush=True)
    if errors:
        raise RuntimeError(errors)
