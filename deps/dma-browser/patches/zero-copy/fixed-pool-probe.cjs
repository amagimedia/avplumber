const assert = require('node:assert/strict');
const fs = require('node:fs');
const { setTimeout: delay } = require('node:timers/promises');
const { app, BrowserWindow } = require('electron');

// Keep the process alive to release leases after the window is destroyed.
app.on('window-all-closed', () => {});

const requested = process.env.AVP_PROBE_POOL_SIZE;
const capacity = requested === undefined ? 11 : Number(requested);
assert.ok(Number.isInteger(capacity) && capacity >= 2 && capacity <= 16);
// With no override, exercise the compiled default. Otherwise select at launch.
app.commandLine.appendSwitch('enable-features',
  `AvpZeroCopyOffscreenExport${requested === undefined ? '' : ':buffers/' + capacity},`
  + 'RenderableMappableSharedImageForceScanout');
app.commandLine.appendSwitch('disable-features',
  'Vulkan,VulkanFromANGLE,DefaultANGLEVulkan,DrmOverlayManager');
app.commandLine.appendSwitch('use-gl', 'angle');
app.commandLine.appendSwitch('use-angle', 'gl-egl');
app.commandLine.appendSwitch('use-webgpu-adapter', 'opengles');
app.commandLine.appendSwitch('enable-gpu');
app.commandLine.appendSwitch('ignore-gpu-blocklist');
app.commandLine.appendSwitch('disable-hardware-overlays');
app.commandLine.appendSwitch('run-all-compositor-stages-before-draw');
app.commandLine.appendSwitch('force-device-scale-factor', '1');
app.commandLine.appendSwitch('enable-logging', 'stderr');

const seen = new Set();
const held = new Map();
let retaining = false;
let failure;
let paints = 0;
let bitmapPaints = 0;
let window;
const timeout = setTimeout(() => {
  console.error('FIXED_POOL ' + JSON.stringify({ passed: false, error: 'Timeout' }));
  app.exit(2);
}, 30000);

async function until(predicate, action, label) {
  const end = Date.now() + 5000;
  while (!predicate()) {
    if (failure) throw failure;
    assert.ok(Date.now() < end, label + ' timed out');
    action();
    await delay(50);
  }
  if (failure) throw failure;
}

app.whenReady().then(async () => {
  window = new BrowserWindow({
    width: 1920, height: 1080, show: false, frame: false, transparent: true,
    enableLargerThanScreen: true,
    webPreferences: {
      backgroundThrottling: false, sandbox: false,
      offscreen: { useSharedTexture: true },
    },
  });
  // Native window creation can clamp to the Wayland display's work area.
  window.setContentSize(1920, 1080);
  const wc = window.webContents;
  wc.setFrameRate(60);
  wc.on('paint', (event) => {
    const texture = event.texture;
    if (!texture) { bitmapPaints++; return; }
    try {
      const info = texture.textureInfo;
      assert.equal(info.codedSize.width, 1920);
      assert.equal(info.codedSize.height, 1080);
      const handle = info.handle?.nativePixmap;
      const fd = (info.planes ?? handle?.planes)?.[0]?.fd;
      assert.equal(typeof fd, 'number', 'Native DMA-BUF is required');
      const stat = fs.fstatSync(fd, { bigint: true });
      const id = `${stat.dev}:${stat.ino}`;
      assert.ok(!held.has(id), 'Producer reused a buffer still held by the consumer');
      seen.add(id);
      assert.ok(seen.size <= capacity, 'Backing allocation changed after warm-up');
      paints++;
      if (retaining) held.set(id, texture);
      else texture.release();
    } catch (error) {
      failure = error;
      texture.release();
    }
  });
  await window.loadURL('data:text/html,<body style="margin:0;background:transparent">'
    + '<div style="width:50%;height:50vh;background:%23ff550080">Fixed pool</div></body>');

  await until(() => seen.size === capacity, () => wc.invalidate(), 'Static warm-up');
  const warm = paints;
  await until(() => paints >= warm + 2 * capacity, () => wc.invalidate(), 'FIFO reuse');
  assert.equal(seen.size, capacity);
  console.log('FIXED_POOL warm ' + JSON.stringify({ capacity, paints, uniqueBuffers: seen.size }));

  retaining = true;
  await until(() => held.size === capacity, () => wc.invalidate(), 'Holding the full pool');
  const full = paints;
  for (let i = 0; i < 8; i++) { wc.invalidate(); await delay(50); }
  assert.equal(paints, full, 'Capture must stop while the full pool is held');

  const [returnedId, returned] = held.entries().next().value;
  held.delete(returnedId);
  returned.release();
  await until(() => held.size === capacity, () => wc.invalidate(), 'Single-buffer return');
  assert.ok(held.has(returnedId), 'Producer must reuse the only returned buffer');
  assert.equal(paints, full + 1);

  retaining = false;
  for (const texture of held.values()) texture.release();
  held.clear();
  const resumed = paints;
  await until(() => paints >= resumed + 2 * capacity, () => wc.invalidate(), 'Batched return');
  assert.equal(seen.size, capacity);
  assert.equal(bitmapPaints, 0, 'Static invalidation must remain on the GPU path');

  retaining = true;
  await until(() => held.size >= 2, () => wc.invalidate(), 'Close with leases');
  window.destroy();
  await delay(100);
  for (const texture of held.values()) texture.release();
  held.clear();
  if (failure) throw failure;
  clearTimeout(timeout);
  console.log('FIXED_POOL ' + JSON.stringify({ passed: true, capacity, paints,
    uniqueBuffers: seen.size, bitmapPaints, pressureSkipped: true }));
  app.exit(0);
}).catch((error) => {
  clearTimeout(timeout);
  console.error('FIXED_POOL ' + JSON.stringify({ passed: false, capacity, error: error.stack,
    paints, uniqueBuffers: seen.size, held: held.size, bitmapPaints }));
  for (const texture of held.values()) texture.release();
  if (window && !window.isDestroyed()) window.destroy();
  app.exit(1);
});
