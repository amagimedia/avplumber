// Linux shared-texture probe. Run in an isolated browser container, never in a live worker.
// It opens its own loopback REST service and socket directory, and connects no media consumer.
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { app, BrowserWindow } = require('electron');
const { Application } = require('../../dist/main/Application');

function integer(name, fallback, min, max) {
  const value = Number(process.env[name] ?? fallback);
  if (!Number.isSafeInteger(value) || value < min || value > max)
    throw new Error(`Invalid ${name}`);
  return value;
}

if (process.platform !== 'linux')
  throw new Error('Run the shared-texture probe on a Linux GPU host');
const startAtMs = integer('BROWSER_SYNC_START_MS', NaN, Date.now() + 2000, Date.now() + 120000);
const durationMs = integer('BROWSER_SYNC_DURATION_MS', 15000, 5000, 120000);
const fps = integer('BROWSER_SYNC_FPS', 30, 1, 240);
const epochMs = integer('BROWSER_SYNC_EPOCH_MS', 0, 0, Number.MAX_SAFE_INTEGER);
const reloadAfterMs = integer('BROWSER_SYNC_RELOAD_MS', 0, 0, durationMs - 2000);
const port = integer('BROWSER_SYNC_PORT', 19099, 1024, 65535);
const socketDir = fs.mkdtempSync(path.join(os.tmpdir(), 'browser-sync-probe-'));
app.setPath('userData', path.join(socketDir, 'profile'));
const service = new Application({
  ...process.env,
  DMA_BROWSER_REST_HOST: '127.0.0.1',
  DMA_BROWSER_REST_PORT: String(port),
  DMA_BROWSER_SOCKET_DIR: socketDir,
  DMA_BROWSER_AUTO_OPEN_URL: '',
  DMA_BROWSER_MAX_WINDOWS: '1',
  DMA_BROWSER_ALLOWED_DIMS: '640x480',
  DMA_BROWSER_STAGGER_FRAMES: '0',
});
service.applyHwAccelSwitches();
const endAtMs = startAtMs + durationMs;
const result = {
  kind: 'browser-sync-probe-v1',
  label: process.env.BROWSER_SYNC_LABEL ?? 'replica',
  fps,
  epochMs,
  phase: 0,
  startAtMs,
  endAtMs,
  paints: [],
  samples: [],
  status: null,
  reloadedAtMs: null,
  errors: [],
};
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
let finishing = false;
const watchdog = setTimeout(
  () => finish(new Error('probe deadline exceeded')),
  endAtMs - Date.now() + 15000,
);

async function request(method, route, body) {
  const response = await fetch(`http://127.0.0.1:${port}${route}`, {
    method,
    headers: { 'Content-Type': 'application/json' },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    signal: AbortSignal.timeout(10000),
  });
  if (!response.ok) throw new Error(`${route}: ${response.status} ${await response.text()}`);
  return response.json();
}

async function finish(error) {
  if (finishing) return;
  finishing = true;
  if (error) result.errors.push(String(error));
  clearTimeout(watchdog);
  // Exit even if shutdown gets stuck; this process owns only its test window and sockets.
  const forceExit = setTimeout(() => app.exit(1), 5000);
  try {
    await service.shutdown();
  } catch (err) {
    result.errors.push(String(err));
  }
  fs.rmSync(socketDir, { recursive: true, force: true });
  console.log(JSON.stringify(result));
  clearTimeout(forceExit);
  app.exit(result.errors.length ? 1 : 0);
}

app.on('window-all-closed', () => {});
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => finish(new Error(signal)));
app
  .whenReady()
  .then(async () => {
    await service.start();
    const config = {
      id: 'sync_probe',
      url: pathToFileURL(path.resolve(__dirname, '../fixtures/sync.html')).href,
      width: 640,
      height: 480,
      fps,
      audio: false,
      sync: { epochMs, phase: 0, loopAnimations: true },
    };
    const opened = await request('POST', '/window/open', config);
    if (JSON.stringify(opened.sync) !== JSON.stringify(config.sync))
      throw new Error('sync settings not applied');
    const win = BrowserWindow.getAllWindows()[0];
    if (!win) throw new Error('probe window missing');
    win.webContents.on('paint', () => {
      const unixMs = Date.now();
      if (unixMs >= startAtMs && unixMs < endAtMs) {
        result.paints.push({ unixMs, monoNs: process.hrtime.bigint().toString() });
      }
    });
    win.webContents.on('render-process-gone', (_event, details) =>
      result.errors.push(`renderer: ${details.reason}`),
    );
    if (Date.now() >= startAtMs) throw new Error('window was not ready before the common start');
    await sleep(startAtMs - Date.now());
    while (Date.now() < endAtMs && !finishing) {
      if (
        reloadAfterMs &&
        result.reloadedAtMs === null &&
        Date.now() >= startAtMs + reloadAfterMs
      ) {
        result.reloadedAtMs = Date.now();
        await request('POST', '/window/refresh', { id: config.id });
      }
      try {
        const sample = await win.webContents.executeJavaScript(`(() => {
        const now = Date.now();
        return { unixMs: now, loops: document.getAnimations()
          .filter(a => a.effect?.getTiming().iterations === Infinity)
          .map(a => ({ name: a.animationName || a.effect.target.id, elapsedMs: a.currentTime })) };
      })()`);
        result.samples.push(sample);
      } catch (error) {
        // Navigation can destroy an evaluation context; retain evidence without inventing a sample.
        result.samples.push({ unixMs: Date.now(), error: String(error) });
      }
      await sleep(200);
    }
    result.status = await request('GET', '/status');
    const status = result.status.windows[0];
    if (!status?.syncStatus?.applications) result.errors.push('no successful phase application');
    // No consumer is attached. These drops confirm valid native textures reached FD broadcast;
    // other drop reasons identify a failure earlier in the capture path.
    if (!(status?.stats?.droppedReasons?.no_consumer > 0))
      result.errors.push('no valid shared textures');
    if (!result.paints.length) result.errors.push('no paints in measurement interval');
    await finish();
  })
  .catch(finish);
