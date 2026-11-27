// Run with the demo's Electron installed and Playwright available on NODE_PATH.
// The loopback fixture serves the actual Setup and Mixer pages. No show is controlled.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const os = require('node:os');
const path = require('node:path');
const { bigShow } = require('./page_fixture.cjs');
const { _electron: electron } = require('playwright');

const demo = path.resolve(__dirname, '../../demos/mixer');
const assets = path.resolve(__dirname, '../../pyplumber/mixer/gui/assets');
const executablePath = require(path.join(demo, 'desktop/node_modules/electron'));

(async () => {
  let phase = 'starting', available = false, previewRequests = 0;
  const { state } = bigShow();
  const commands = [];
  const server = http.createServer((req, res) => {
    res.setHeader('Content-Type', 'text/html; charset=utf-8');
    const { pathname } = new URL(req.url, 'http://127.0.0.1');
    if (pathname === '/api/setup') return res.end(JSON.stringify({ phase, message: phase }));
    if (!available) { res.writeHead(503); return res.end(); }
    if (pathname === '/api/state') return res.end(JSON.stringify(state));
    if (pathname === '/api/command') {
      let body = '';
      req.on('data', chunk => { body += chunk; });
      return req.on('end', () => { commands.push(JSON.parse(body)); res.end('{}'); });
    }
    if (pathname === '/setup/') return res.end(fs.readFileSync(path.join(demo, 'setup.html')));
    if (pathname === '/graph/') return res.end('<label>Find node<input></label>');
    if (pathname === '/setup.js' || pathname === '/outputs.js') {
      res.setHeader('Content-Type', 'text/javascript');
      return res.end(fs.readFileSync(path.join(assets, pathname.slice(1))));
    }
    if (pathname === '/') {
      const config = { setup: true, graph_url: '/graph/', preview_base: '/preview/' };
      return res.end(fs.readFileSync(path.join(assets, 'index.html'), 'utf8')
        .replace('type="application/json">{}</script>', 'type="application/json">' + JSON.stringify(config) + '</script>'));
    }
    if (pathname === '/preview/') previewRequests++;
    res.writeHead(404);
    res.end();
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const port = server.address().port;
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'avp-desktop-'));
  const launch = () => electron.launch({ executablePath, args: ['--user-data-dir=' + profile, path.join(demo, 'desktop')],
    env: { ...process.env, MIXER_PUBLIC_PORT: String(port) } });
  let app;
  try {
    app = await launch();
    if (process.env.DEBUG_DESKTOP) {
      app.process().stderr.pipe(process.stderr);
      app.on('window', page => { console.log('opened', page.url()); page.on('close', () => console.log('closed', page.url())); });
    }
    const setup = await app.firstWindow();
    await setup.getByRole('heading', { name: 'Connecting to the desktop mixer…' }).waitFor();
    assert.equal(await app.evaluate(({ nativeTheme }) => nativeTheme.themeSource), 'dark');
    available = true;
    await setup.waitForURL('**/setup/');
    phase = 'running';
    const [mixer] = await Promise.all([app.waitForEvent('window'), setup.locator('#open-controls').click({ noWaitAfter: true })]);
    await mixer.locator('#graph-link').waitFor();
    await mixer.locator('canvas[data-feed="aux_pvw"]').waitFor();
    const cut = mixer.waitForResponse(r => r.url().endsWith('/api/command'));
    await mixer.keyboard.press('c');
    await cut;
    assert.equal(commands.at(-1).command, 'cut');
    await mixer.locator('#setup-link').click({ noWaitAfter: true });
    assert.equal(app.windows().length, 2);
    assert.match(mixer.url(), new RegExp(':' + port + '/$'));
    await mixer.bringToFront();
    const [graph] = await Promise.all([app.waitForEvent('window'), mixer.locator('#graph-link').click()]);
    await graph.getByRole('textbox').fill('compositor');
    await mixer.bringToFront();
    await mixer.locator('#graph-link').click();
    assert.equal(app.windows().length, 3);
    assert.equal(await graph.getByRole('textbox').inputValue(), 'compositor');
    assert.equal(await mixer.evaluate(() => typeof require), 'undefined');
    if (process.platform === 'linux') {
      const menuBars = await app.evaluate(({ BrowserWindow }) => Object.fromEntries(
        BrowserWindow.getAllWindows().map(w => [w.getTitle(), w.isMenuBarVisible()])));
      assert.deepEqual(menuBars, { 'Mixer Setup': false, 'Processing Graph': false, 'Mixer': true });
    }
    // The Window menu's shortcut can recover the console while its tools stay open.
    await mixer.close();
    const [reopened] = await Promise.all([app.waitForEvent('window'), app.evaluate(({ Menu }) => {
      const action = Menu.getApplicationMenu().getMenuItemById('show-mixer');
      if (action.accelerator !== 'CmdOrCtrl+1') throw new Error('Missing Mixer Controls shortcut');
      action.click();
    })]);
    await reopened.locator('#graph-link').waitFor();
    assert.equal(app.windows().length, 3);
    assert.equal(await graph.getByRole('textbox').inputValue(), 'compositor');
    await app.close();
    app = await launch();
    const live = await app.firstWindow();
    await live.locator('#graph-link').waitFor();
    assert.equal(app.windows().length, 1);
    assert.equal(previewRequests, 0, 'Electron must never load the WebRTC player');
    console.log('Desktop startup, reconnect, window routing and viewer isolation passed');
  } finally {
    if (app) await app.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmSync(profile, { recursive: true, force: true });
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
