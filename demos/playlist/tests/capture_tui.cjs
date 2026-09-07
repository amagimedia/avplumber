// Offscreen Electron capture of the TUI served by ttyd, as JPEG frames at a
// fixed rate, for compose_recording.sh.  Runs inside the demo browser image
// (same as the mixer recording):
//
//   CAPTURE_URL=http://127.0.0.1:7681 CAPTURE_DURATION=60 CAPTURE_DIR=/capture \
//     electron --no-sandbox capture_tui.cjs
//
// Writes <dir>/tui/NNNNN.jpg, <dir>/tui.json (frame timings, start wallclock)
// and exits.  Keyboard actions for the recording are driven separately over
// the control port (see record_demo.py), so the capture only watches.
const {app, BrowserWindow} = require('electron');
const fs = require('fs');
app.disableHardwareAcceleration();
const url = process.env.CAPTURE_URL || 'http://127.0.0.1:7681';
const duration = Number(process.env.CAPTURE_DURATION || 60);
const fps = Number(process.env.CAPTURE_FPS || 30);
const dir = process.env.CAPTURE_DIR || '/capture';
const width = Number(process.env.CAPTURE_WIDTH || 1000), height = Number(process.env.CAPTURE_HEIGHT || 900);
const sleep = ms => new Promise(r => setTimeout(r, ms));

app.whenReady().then(async () => {
  try {
    const win = new BrowserWindow({width, height, show: false,
      webPreferences: {offscreen: true, backgroundThrottling: false}});
    let paint;
    win.webContents.setFrameRate(fps);
    win.webContents.on('paint', (_e, _r, img) => paint = img);
    await win.loadURL(url);
    for (let n = 0; n < 60; n++) {           // ttyd draws on a canvas: wait for the xterm element
      const ready = await win.webContents.executeJavaScript(
        '!!document.querySelector(".xterm-screen, .xterm, canvas")');
      if (ready) break;
      if (n === 59) throw Error('TUI did not render');
      await sleep(500);
    }
    await sleep(3000);                        // let Textual paint its first frame
    fs.mkdirSync(`${dir}/tui`, {recursive: true});
    const start = performance.now(), startWallclock = Date.now(), timings = [];
    for (let frame = 0; performance.now() - start < duration * 1000; frame++) {
      await sleep(Math.max(0, start + frame * 1000 / fps - performance.now()));
      fs.writeFileSync(`${dir}/tui/${String(frame).padStart(5, '0')}.jpg`,
        (paint || await win.webContents.capturePage()).toJPEG(92));
      timings.push(performance.now() - start);
    }
    fs.writeFileSync(`${dir}/tui.json`, JSON.stringify({url, fps, startWallclock, timings}, null, 2));
    console.log(JSON.stringify({result: 'CAPTURED', frames: timings.length, seconds: timings.at(-1) / 1000}));
    app.exit(0);
  } catch (e) { console.error(e); app.exit(1); }
});
