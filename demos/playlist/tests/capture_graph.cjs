// Capture the AVPlumber web UI processing graph as PNG, grouped and ungrouped,
// with Electron offscreen rendering (same approach as the other demos).
//
//   WEBUI_URL=http://127.0.0.1:22222 CAPTURE_DIR=/capture NODE_HINT=input_pl0 \
//     electron --no-sandbox capture_graph.cjs [probe]
//
// `probe` prints the UI's buttons, labels and body text so selectors can be
// checked before a capture.  Otherwise writes <dir>/playlist-graph.png (full
// native graph) and <dir>/playlist-graph-grouped.png (grouped overview).
const {app, BrowserWindow} = require('electron');
const fs = require('fs');
app.disableHardwareAcceleration();
const url = process.env.WEBUI_URL || 'http://127.0.0.1:22222';
const dir = process.env.CAPTURE_DIR || '/capture';
const hint = process.env.NODE_HINT || 'input_pl0';
const probe = process.argv.includes('probe');
const sleep = ms => new Promise(r => setTimeout(r, ms));

const layout = {root: {type: 'column', content: [
  {type: 'stack', size: '72%', content: [{type: 'component', componentType: 'graph', title: 'Playlist graph', componentState: {}}]},
  {type: 'stack', size: '28%', content: [{type: 'component', componentType: 'queues', title: 'Live queue statistics', componentState: {}}]},
]}};

async function clickText(win, texts) {
  return win.webContents.executeJavaScript(`(() => {
    const wanted = ${JSON.stringify(texts)};
    const els = Array.from(document.querySelectorAll('button, label, [role=button], input[type=checkbox]'));
    for (const el of els) {
      const text = (el.textContent || el.getAttribute('aria-label') || el.title || '').trim();
      if (wanted.some(w => text === w || text.startsWith(w))) { el.click(); return text; }
    }
    return null;
  })()`);
}

app.whenReady().then(async () => {
  try {
    const win = new BrowserWindow({width: 6000, height: 2600, show: false,
      webPreferences: {offscreen: true, backgroundThrottling: false}});
    await win.loadURL(url);
    await win.webContents.executeJavaScript(
      `localStorage.setItem('avplumber.webui.dock.layout.v1', ${JSON.stringify(JSON.stringify(layout))})`);
    win.reload();
    await sleep(4000);
    if (probe) {
      const dump = await win.webContents.executeJavaScript(`JSON.stringify({
        buttons: Array.from(document.querySelectorAll('button, [role=button]')).map(b => b.textContent.trim()).filter(Boolean),
        labels: Array.from(document.querySelectorAll('label')).map(l => l.textContent.trim()).filter(Boolean),
        checkboxes: Array.from(document.querySelectorAll('input[type=checkbox]')).map(c => (c.id || '') + ':' + (c.closest('label')?.textContent.trim() || '') + ':' + c.checked),
        text: document.body.innerText.slice(0, 1500)})`);
      console.log(dump);
      app.exit(0); return;
    }
    await clickText(win, ['Expand / restore graph']);
    await clickText(win, ['Refresh graph']);
    await clickText(win, ['Refresh queues']);
    await sleep(10000);
    const text = await win.webContents.executeJavaScript('document.body.innerText');
    if (!text.includes(hint)) throw Error(`graph did not load (no ${hint} in page)`);
    fs.mkdirSync(dir, {recursive: true});
    // Whatever state the UI starts in, capture it and its toggled counterpart.
    const grouped = await win.webContents.executeJavaScript(
      `Array.from(document.querySelectorAll('input[type=checkbox]')).some(c => /grouped/i.test(c.closest('label')?.textContent || '') && c.checked)`);
    const first = grouped ? 'playlist-graph-grouped.png' : 'playlist-graph.png';
    const second = grouped ? 'playlist-graph.png' : 'playlist-graph-grouped.png';
    fs.writeFileSync(`${dir}/${first}`, (await win.webContents.capturePage()).toPNG());
    const toggled = await clickText(win, ['Grouped overview', 'Grouped']);
    if (toggled === null) throw Error('no Grouped overview toggle found');
    await sleep(6000);
    fs.writeFileSync(`${dir}/${second}`, (await win.webContents.capturePage()).toPNG());
    console.log(JSON.stringify({result: 'CAPTURED', files: [first, second], grouped_first: grouped}));
    app.exit(0);
  } catch (e) { console.error(e); app.exit(1); }
});
