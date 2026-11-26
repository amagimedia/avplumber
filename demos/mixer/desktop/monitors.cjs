// One subscription per visible bus. Windows showing the same output share an
// import. The exporter may reuse its image only after Chromium's GPU is done.
const fs = require('node:fs');
const path = require('node:path');
const { ipcMain, sharedTexture } = require('electron');

function registerMonitors(directory, allowed) {
  const windows = new Map(), feeds = new Map();
  const native = process.platform === 'linux' ? require('./receiver.node') : null;
  let timer;
  function status(feed, message) {
    for (const [win, names] of windows) if (names.has(feed.id)) win.send('monitor-status', feed.id, message);
  }
  function disconnect(feed) {
    if (feed.socket !== null) fs.closeSync(feed.socket);
    feed.socket = null;
    feed.retry = Date.now() + 1000;
    status(feed, 'Waiting for local video');
  }
  function reconcile() {
    const active = new Set([...windows.values()].flatMap(ids => [...ids]));
    for (const [id, feed] of feeds) if (!active.has(id)) {
      disconnect(feed);
      feeds.delete(id);
    }
    for (const id of active) if (!feeds.has(id)) feeds.set(id, { id, socket: null, retry: 0, epoch: 0 });
    if (feeds.size && !timer) timer = setTimeout(tick, 0);
    if (!feeds.size) { clearTimeout(timer); timer = undefined; }
  }
  ipcMain.on('monitor-feeds', (event, names) => {
    const win = event.sender;
    if (event.senderFrame !== win.mainFrame || !allowed(win) || !Array.isArray(names) ||
        names.length > 256 || names.some(n => typeof n !== 'string' || !/^[a-zA-Z0-9_-]{1,80}$/.test(n))) return;
    if (!windows.has(win)) {
      const clear = () => { windows.delete(win); reconcile(); };
      win.once('destroyed', clear);
      win.on('render-process-gone', () => { windows.set(win, new Set()); reconcile(); });
      win.on('did-start-navigation', event => {
        if (event.isMainFrame && !event.isSameDocument) { windows.set(win, new Set()); reconcile(); }
      });
    }
    windows.set(win, new Set(names));
    reconcile();
  });
  function tick() {
    timer = undefined;
    for (const feed of feeds.values()) {
      if (feed.socket === null) {
        if (Date.now() < feed.retry) continue;
        const fd = native?.connectFeed(path.join(directory, feed.id + '.sock')) ?? -1;
        if (fd < 0) { feed.retry = Date.now() + 1000; status(feed, 'Waiting for local video'); continue; }
        feed.socket = fd;
        feed.epoch++;
        feed.lastFrame = Date.now();
      }
      const header = Buffer.alloc(48);
      const fd = native.receive(feed.socket, header);
      if (fd === -1) {
        if (feed.lastFrame && Date.now() - feed.lastFrame >= 2000) {
          status(feed, 'Waiting for local video');
          feed.lastFrame = 0;
        }
        continue;
      }
      if (fd < 0) { disconnect(feed); continue; }
      feed.lastFrame = Date.now();
      const epoch = feed.epoch;
      const ack = () => {
        if (feed.socket === null || feed.epoch !== epoch) return;
        const bytes = Buffer.alloc(16);
        bytes.write('ACK1');
        header.copy(bytes, 8, 40, 48);
        try {
          if (fs.writeSync(feed.socket, bytes) !== bytes.length) disconnect(feed);
        } catch { disconnect(feed); }
      };
      let texture;
      try {
        const width = header.readUInt32LE(0), height = header.readUInt32LE(4);
        const stride = header.readUInt32LE(8), fourcc = header.readUInt32LE(12);
        const offset = Number(header.readBigUInt64LE(24));
        const size = fs.fstatSync(fd).size;
        if (!width || !height || width > 16384 || height > 16384 || stride < width * 4 ||
            !Number.isSafeInteger(offset) || !size || offset >= size) throw new Error('Invalid texture layout');
        const pixelFormat = ({ 0x52474241: 'rgba', 0x42475241: 'bgra' })[fourcc];
        if (!pixelFormat) throw new Error('Unsupported local texture format');
        texture = sharedTexture.importSharedTexture({ textureInfo: {
          pixelFormat, codedSize: { width, height }, timestamp: Number(header.readBigUInt64LE(32) / 1000n),
          handle: { nativePixmap: { modifier: header.readBigUInt64LE(16).toString(),
            planes: [{ fd, stride, offset, size }], supportsZeroCopyWebGpuImport: false } }
        }, allReferencesReleased: ack });
        const sends = [];
        for (const [win, names] of windows) {
          if (!names.has(feed.id) || win.isDestroyed()) continue;
          sends.push(sharedTexture.sendSharedTexture({ frame: win.mainFrame, importedSharedTexture: texture }, feed.id)
            .catch(error => {
              if (!win.isDestroyed() && windows.get(win)?.has(feed.id)) win.send('monitor-status', feed.id, error.message);
            }));
        }
        Promise.allSettled(sends).finally(() => texture.release());
      } catch (error) {
        if (texture) texture.release(); else ack();
        status(feed, error.message);
      } finally {
        fs.closeSync(fd); // Electron duplicates the native-pixmap plane on import.
      }
    }
    if (feeds.size) timer = setTimeout(tick, [...feeds.values()].some(f => f.socket !== null) ? 4 : 250);
  }
}
module.exports = { registerMonitors };
