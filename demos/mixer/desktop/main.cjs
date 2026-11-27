// The desktop uses the same documents and keyboard handlers as the browser.
// Each tool has one window. Navigation between tools focuses that window.
const { app, BrowserWindow, Menu, nativeTheme, net, session, ipcMain } = require('electron');
const path = require('node:path');

const port = Number(process.env.MIXER_PUBLIC_PORT || 7681);
if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Invalid MIXER_PUBLIC_PORT');
const origin = `http://127.0.0.1:${port}`;
const views = {
  mixer: { title: 'Mixer', route: '/', width: 1440, height: 900 },
  setup: { title: 'Mixer Setup', route: '/setup/', width: 980, height: 850 },
  graph: { title: 'Processing Graph', route: '/graph/', width: 1280, height: 820 },
  wall: { title: 'Outputs', route: '/wall', width: 1280, height: 800 },
  levels: { title: 'Audio Levels', route: '/levels', width: 1040, height: 760 },
  matrix: { title: 'Audio Matrix', route: '/matrix', width: 1200, height: 800 }
};
const windows = new Map();
let startupSetup, quitting = false;

function destination(address) {
  if (!URL.canParse(address)) return;
  const url = new URL(address);
  if (url.origin !== origin || url.username || url.password) return;
  const pathname = url.pathname.replace(/\/$/, '') || '/';
  return Object.keys(views).find(key => pathname === (views[key].route.replace(/\/$/, '') || '/'));
}

function focus(win) {
  if (win.isMinimized()) win.restore();
  win.show();
  win.focus();
}

function openView(key, address = origin + views[key].route) {
  const view = views[key];
  const existing = windows.get(key);
  if (existing) {
    focus(existing);
    return existing;
  }
  const win = new BrowserWindow({
    title: view.title, width: view.width, height: view.height,
    backgroundColor: '#15191c', show: false,
    webPreferences: { sandbox: true, contextIsolation: true, nodeIntegration: false,
      preload: path.join(__dirname, 'preload.cjs') }
  });
  windows.set(key, win);
  // Keep menu accelerators available in tool windows, including return to Mixer.
  if (key !== 'mixer' && process.platform !== 'darwin') win.setMenuBarVisibility(false);
  if (key === 'mixer') startupSetup?.close();
  win.once('ready-to-show', () => win.show());
  win.on('page-title-updated', event => event.preventDefault());
  let retry;
  win.on('closed', () => {
    clearTimeout(retry);
    windows.delete(key);
  });

  win.webContents.setWindowOpenHandler(({ url }) => {
    const target = destination(url);
    if (target) openView(target, url);
    return { action: 'deny' };
  });
  function route(event) {
    if (!event.isMainFrame) return;
    const url = event.url;
    const target = destination(url);
    if (target === key) return;
    event.preventDefault();
    if (target) openView(target, url);
  }
  win.webContents.on('will-navigate', route);
  win.webContents.on('will-redirect', route);

  // On a fresh Compose start nginx may already accept connections while a
  // service is still coming up. Retry both connection failures and proxy errors.
  function unavailable() {
    if (win.isDestroyed() || retry) return;
    void win.loadFile(path.join(__dirname, 'waiting.html')).catch(() => {});
    retry = setTimeout(() => {
      retry = undefined;
      load();
    }, 2000);
  }
  function load() {
    void win.loadURL(address).catch(error => {
      if (error.code !== 'ERR_ABORTED') unavailable();
    });
  }
  win.webContents.on('did-navigate', (_event, _url, status) => {
    if (status >= 400) unavailable();
  });
  load();
  return win;
}

async function initialWindow() {
  let phase;
  try {
    const status = await net.fetch(origin + '/api/setup', { signal: AbortSignal.timeout(2000) });
    if (status.ok) phase = (await status.json()).phase;
  } catch { /* The stack may still be starting. Setup reports its progress. */ }
  if (windows.size) return;
  if (phase === 'running') {
    openView('mixer');
  } else {
    startupSetup = openView('setup');
    startupSetup.once('close', () => {
      startupSetup = undefined;
      if (!quitting && !windows.has('mixer')) openView('mixer');
    });
  }
}

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('before-quit', () => { quitting = true; });
  app.on('second-instance', () => {
    const win = windows.get('mixer') || windows.get('setup') || windows.values().next().value;
    if (win) focus(win);
  });
  app.whenReady().then(() => {
    nativeTheme.themeSource = 'dark';
    // A desktop app opened against an older web UI must never start its network player.
    session.defaultSession.webRequest.onBeforeRequest({ urls: [origin + '/preview/*', 'http://127.0.0.1:8080/*'] },
      (_details, done) => done({ cancel: true }));
    ipcMain.on('open-mixer-window', (event, url) => {
      if (event.senderFrame !== event.sender.mainFrame || !destination(event.sender.getURL())) return;
      const key = typeof url === 'string' && destination(url);
      if (key) openView(key, url);
    });
    const directory = process.env.MIXER_MONITOR_DIR || path.resolve(__dirname, '../../../media/desktop/.desktop-monitors');
    require('./monitors.cjs').registerMonitors(directory, contents =>
      ['mixer', 'wall'].some(key => windows.get(key)?.webContents === contents && destination(contents.getURL()) === key));
    const menu = [
      { label: 'Mixer', submenu: [
        { label: 'Setup…', click: () => openView('setup') },
        { type: 'separator' }, { role: 'quit', label: 'Quit Desktop App' }
      ] },
      { role: 'editMenu' },
      { label: 'Window', submenu: [
        { id: 'show-mixer', label: 'Mixer Controls', accelerator: 'CmdOrCtrl+1', click: () => openView('mixer') },
        { label: 'Processing Graph', click: () => openView('graph') },
        { label: 'Outputs', click: () => openView('wall') },
        { label: 'Audio Levels', click: () => openView('levels') },
        { label: 'Audio Matrix', click: () => openView('matrix') },
        { type: 'separator' }, { role: 'togglefullscreen' }, { role: 'close' }
      ] }
    ];
    if (process.platform === 'darwin') menu.unshift({ role: 'appMenu' });
    Menu.setApplicationMenu(Menu.buildFromTemplate(menu));
    return initialWindow();
  });
  app.on('window-all-closed', () => app.quit());
}
