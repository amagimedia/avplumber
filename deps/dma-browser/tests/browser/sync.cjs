// CPU/DOM smoke test; no fdpass, CUDA, DMA-BUF or offscreen capture qualification.
// Run after npm run build: node_modules/.bin/electron tests/browser/sync.cjs
const assert = require('node:assert/strict');
const path = require('node:path');
const { app, BrowserWindow } = require('electron');

app.disableHardwareAcceleration();
const windows = [];
const epochMs = Date.now() - 10000;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const timeout = setTimeout(() => {
  console.error('browser sync smoke timed out');
  app.exit(1);
}, 20000);

async function open(sync) {
  const win = new BrowserWindow({
    show: false,
    width: 640,
    height: 480,
    webPreferences: {
      offscreen: true,
      backgroundThrottling: false,
      sandbox: false,
      preload: path.resolve(__dirname, '../../dist/preload/index.js'),
      additionalArguments: sync ? [`--loop-sync-epoch-ms=${epochMs}`] : [],
    },
  });
  windows.push(win);
  win.webContents.on('preload-error', (_event, _path, error) => {
    throw error;
  });
  await win.loadFile(path.resolve(__dirname, '../fixtures/sync.html'));
  return win;
}

function sample(win) {
  return win.webContents.executeJavaScript(`(() => {
    const now = Date.now();
    const doc = document;
    const child = doc.getElementById('child').contentDocument;
    const animations = [...doc.getAnimations(), ...child.getAnimations()];
    return {
      now, age: performance.now(),
      animations: animations.map(a => ({
        current: a.currentTime,
        iterations: a.effect.getTiming().iterations === Infinity ? 'infinite' : 'finite',
        delay: a.effect.getTiming().delay,
      })),
    };
  })()`);
}

function checkAligned(value) {
  assert.equal(value.animations.filter((a) => a.iterations === 'infinite').length, 3);
  for (const a of value.animations) {
    const expected = a.iterations === 'infinite' ? value.now - epochMs : value.age;
    // CPU browser smoke: verifies clock origin, not production frame accuracy.
    assert.ok(Math.abs(a.current - expected) < 150, JSON.stringify({ a, expected }));
  }
  assert.ok(
    value.animations.some((a) => a.delay === -500),
    'CSS phase delay was preserved',
  );
}

app
  .whenReady()
  .then(async () => {
    app.dock?.hide();
    const a = await open(true);
    await sleep(350);
    const b = await open(true);
    const ordinary = await open(false);
    await sleep(1000);
    checkAligned(await sample(a));
    checkAligned(await sample(b));
    const baseline = await sample(ordinary);
    assert.ok(
      baseline.animations.every((anim) => anim.current < 5000),
      'opt-out page was rephased',
    );
    await b.reload();
    await sleep(1200);
    checkAligned(await sample(b));
    console.log(
      'PASS: independent document origins, CSS/WAAPI/iframe loops, reload, finite animations, opt-out',
    );
    windows.forEach((win) => win.destroy());
    clearTimeout(timeout);
    app.exit(0);
  })
  .catch((error) => {
    console.error(error);
    clearTimeout(timeout);
    app.exit(1);
  });
