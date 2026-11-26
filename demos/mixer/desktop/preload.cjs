// Keep native navigation out of Chromium's history and focus handling.
const { contextBridge, ipcRenderer, sharedTexture } = require('electron');
document.addEventListener('click', event => {
  const anchor = event.composedPath().find(el => el.tagName === 'A' && el.href);
  if (!anchor || event.defaultPrevented || event.button !== 0) return;
  const target = new URL(anchor.href);
  if (target.origin !== location.origin || target.pathname === location.pathname) return;
  event.preventDefault();
  ipcRenderer.send('open-mixer-window', target.href);
}, true);
contextBridge.exposeInMainWorld('mixerDesktop', true);

let feeds = new Map();
const rates = new WeakMap();
function update() {
  const next = new Map();
  if (!document.hidden) {
    for (const canvas of document.querySelectorAll('canvas[data-feed]')) {
      const id = canvas.dataset.feed;
      if (!id) continue;
      if (!next.has(id)) next.set(id, []);
      next.get(id).push(canvas);
    }
  }
  const changed = next.size !== feeds.size || [...next.keys()].some(id => !feeds.has(id));
  feeds = next;
  if (changed) ipcRenderer.send('monitor-feeds', [...feeds.keys()]);
}

sharedTexture.setSharedTextureReceiver(async ({ importedSharedTexture }, id) => {
  let frame;
  try {
    frame = importedSharedTexture.getVideoFrame();
    for (const canvas of feeds.get(id) || []) {
      if (canvas.width !== frame.displayWidth) canvas.width = frame.displayWidth;
      if (canvas.height !== frame.displayHeight) canvas.height = frame.displayHeight;
      canvas.getContext('2d', { alpha: false, desynchronized: true }).drawImage(frame, 0, 0);
      const now = performance.now();
      let rate = rates.get(canvas);
      if (!rate || rate.id !== id) {
        rate = { id, start: now, frames: 0 };
        rates.set(canvas, rate);
        canvas.dataset.status = 'Local GPU monitor';
      }
      rate.frames++;
      if (now - rate.start >= 1000) {
        canvas.dataset.status = `Local GPU · ${(rate.frames * 1000 / (now - rate.start)).toFixed(1)} fps`;
        rate.start = now;
        rate.frames = 0;
      }
    }
  } finally {
    frame?.close();
    importedSharedTexture.release();
  }
});

ipcRenderer.on('monitor-status', (_event, id, reason) => {
  for (const canvas of feeds.get(id) || []) {
    if (canvas.dataset.status === reason) continue;
    rates.delete(canvas);
    canvas.dataset.status = reason;
    canvas.getContext('2d', { alpha: false, desynchronized: true })?.clearRect(0, 0, canvas.width, canvas.height);
  }
});
addEventListener('DOMContentLoaded', () => {
  new MutationObserver(records => {
    if (records.some(r => r.type === 'attributes' || [...r.addedNodes, ...r.removedNodes].some(
      n => n.nodeType === 1 && (n.matches('canvas') || n.querySelector('canvas'))))) update();
  }).observe(document.documentElement, { childList: true, subtree: true, attributes: true, attributeFilter: ['data-feed'] });
  document.addEventListener('visibilitychange', update);
  update();
});
