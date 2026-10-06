"""An application can supply its own inputs without importing demo setup policy."""
import json
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

from pyplumber.mixer.gui import GpuStats, HostStats, MixerBridge, SetupController, serve


def test_public_components_do_not_import_demo_or_native_runtime():
    code = """
import sys
from pyplumber.mixer.gui import GpuStats, HostStats, MixerBridge, SetupController, serve
from pyplumber.mixer.gui.setup_runtime import SetupRuntime
assert not any(name == 'demos' or name.startswith('demos.') for name in sys.modules)
assert '_avplumber' not in sys.modules
"""
    subprocess.run([sys.executable, '-c', code], check=True,
                   cwd=Path(__file__).resolve().parents[2])


def test_application_owned_inputs_and_shared_encoder_components(http_server):
    class ProductSetup:
        settings = None
        saved_aux = None

        def page(self):
            return b'<main>Product inputs</main><script src="/setup.js"></script>'

        def status(self):
            return dict(phase='running', message='ready', revision=42)

        def apply(self, settings):
            self.settings = settings

        def remember_aux(self, bus_id, fields):
            self.saved_aux = (bus_id, fields)

    class Bridge:
        def state(self):
            return dict(mixer='product', settings={'sources': []})

        def take(self, request):
            return {'revision': 3}

    class EncoderMetrics:
        def snapshot(self):
            return [dict(index=0, encoder_sessions=2, encoder_fps=120.0,
                         encoder_mpix_s=248.832)]

        def __bool__(self):
            return False  # a caller's provider must not be replaced by truthiness

    setup: SetupController = ProductSetup()
    gpu = EncoderMetrics()
    server = http_server(Bridge(), setup=setup, gpu=gpu)
    base = f'http://127.0.0.1:{server.server_port}'

    def post(path, body):
        request = Request(base + path, data=json.dumps(body).encode(),
                          headers={'Content-Type': 'application/json'})
        with urlopen(request) as response:
            return response.status, json.load(response)

    with urlopen(base + '/setup') as response:
        assert b'Product inputs' in response.read()
    with urlopen(base + '/setup.js') as response:
        assert b'encodeRow' in response.read()
    settings = {'inputs': [{'id': 'camera_a', 'kind': 'product_receiver'}]}
    status, _ = post('/api/setup', settings)
    assert status == 202 and setup.settings == settings
    for _ in range(2):
        with urlopen(base + '/api/state') as response:
            state = json.load(response)
        assert state['gpus'] == gpu.snapshot()
        assert state['setup_revision'] == 42
    status, _ = post('/api/command', {'command': 'aux', 'bus': 'monitor', 'scene': 'camera_a'})
    assert status == 200 and setup.saved_aux == ('monitor', {'revision': 3})
