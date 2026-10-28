// Capture the live browser display. Both viewers must decode before any show control changes.
// Requires Playwright, Chrome with the stream codecs, FFmpeg and a dedicated 1920x1080 X display.
const fs = require('fs'), path = require('path'), {spawn} = require('child_process');
const {chromium} = require('playwright');
const url = process.env.MIXER_UI_URL || 'http://127.0.0.1:7681/';
const out = path.resolve(process.argv[2] || 'ui-capture');
const FPS = 30, SECONDS = 20;
fs.mkdirSync(out, {recursive: true});
const report = {fps: FPS, seconds: SECONDS, method: 'Live Chrome display captured by FFmpeg x11grab; real mouse-operated T-bar and live WebRTC viewers.', events: []};
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
(async () => {
  const browser = await chromium.launch({headless: false,
    ...(process.env.CHROME_EXECUTABLE ? {executablePath: process.env.CHROME_EXECUTABLE} : {channel: 'chrome'}),
    args: ['--no-sandbox', '--start-fullscreen', '--window-position=0,0', '--window-size=1920,1080', '--autoplay-policy=no-user-gesture-required', ...JSON.parse(process.env.CHROME_ARGS || '[]')]});
  const context = await browser.newContext({viewport: null,
    ...(process.env.MIXER_HTTP_USER ? {httpCredentials: {username: process.env.MIXER_HTTP_USER, password: process.env.MIXER_HTTP_PASSWORD || ''}} : {})});
  await context.addInitScript(() => {
    localStorage.setItem('avplumber.mixer.viewers', JSON.stringify(['mv', 'program']));
    localStorage.setItem('avplumber.mixer.tbarUpright', 'true');
  });
  const page = await context.newPage();
  let original, recorder, changed = false;
  const state = async () => {
    const response = await context.request.get(new URL('/api/state', url).href);
    if (!response.ok()) throw Error(`State unavailable: HTTP ${response.status()}`);
    return response.json();
  };
  const command = async body => {
    const response = await context.request.post(new URL('/api/command', url).href, {data: body});
    if (!response.ok()) throw Error(`Restore failed: ${await response.text()}`);
  };
  const videos = async () => {
    const results = [];
    for (const frame of page.frames()) {
      const samples = await frame.evaluate(() => [...document.querySelectorAll('video')].map(v => ({
        width: v.videoWidth, height: v.videoHeight, time: v.currentTime,
        frames: v.getVideoPlaybackQuality().totalVideoFrames, dropped: v.getVideoPlaybackQuality().droppedVideoFrames,
      })));
      for (const sample of samples) results.push({url: frame.url(), ...sample});
    }
    return results;
  };
  const selectScene = async scene => {
    const group = scene.match(/^(random|grid|fullscreen|pip|alpha_overlay)_/)[1];
    await page.locator(`#chips [data-focus="${group}"]`).click();
    for (let i = 0; i < 10; i++) {
      const tile = page.locator(`[data-scene="${scene}"]`);
      if (await tile.count()) { await tile.click(); report.events.push({at: Date.now(), scene}); return; }
      await page.locator('#pg-next').click();
    }
    throw Error(`Scene not found: ${scene}`);
  };
  const direct = async on => {
    const button = page.locator('#direct');
    if ((await button.getAttribute('aria-pressed') === 'true') !== on) await button.click();
  };
  const stroke = async (from, to, duration) => {
    const track = await page.locator('.tbar .track').boundingBox();
    const grip = await page.locator('.tbar .grip').boundingBox();
    const x = grip.x + grip.width / 2;
    await page.mouse.move(x, grip.y + grip.height / 2);
    await page.mouse.down();
    const start = Date.now();
    report.events.push({at: start, stroke: [from, to], duration_ms: duration});
    for (;;) {
      const fraction = Math.min(1, (Date.now() - start) / duration);
      const position = from + (to - from) * fraction;
      await page.mouse.move(x, track.y + track.height - 18 - position * (track.height - 36));
      if (fraction === 1) break;
      await sleep(16);
    }
    await page.mouse.up();
  };
  try {
    await page.goto(url);
    const windowSession = await context.newCDPSession(page);
    const {windowId} = await windowSession.send('Browser.getWindowForTarget');
    await windowSession.send('Browser.setWindowBounds', {windowId, bounds: {windowState: 'fullscreen'}});
    await sleep(1000);
    await page.waitForFunction(() => document.querySelector('#counts')?.textContent.includes('192'));
    original = await state();
    if (original.settings.source_count !== 192 || original.settings.canvas.fps !== FPS) throw Error('Expected the 192-input 30 fps show');
    const first = await videos();
    await sleep(5000);
    const ready = await videos();
    if (ready.length !== 2 || ready.some((v, i) => v.width !== 1920 || v.height !== 1080 || v.frames - (first[i]?.frames || 0) < 60))
      throw Error('Both live viewers must decode before capture');
    let size = await page.evaluate(() => [innerWidth, innerHeight]);
    if (size[0] !== 1920 || size[1] !== 1080) {
      await page.evaluate(() => document.documentElement.requestFullscreen()); await sleep(1000);
      size = await page.evaluate(() => [innerWidth, innerHeight]);
    }
    if (size[0] !== 1920 || size[1] !== 1080) throw Error(`Unexpected fullscreen size: ${size}`);
    if (original.status.transition_manual) throw Error('Another controller is moving the lever');
    const origin = original.status.lever.position;
    if (![0, 1].includes(origin)) throw Error('Park the T-bar at an endpoint first');
    report.original = original;
    changed = true;
    await direct(true);
    await selectScene('random_180');
    for (const key of original.dsk) {
      if (!key.on) await page.locator('#keys button').filter({hasText: ({lower_third:'Lower third',ticker:'Ticker',bug_left:'Bug left',bug_right:'Bug right'})[key.id]}).first().click();
    }
    await page.locator('#fade').click();
    await sleep(2000);
    report.before = await videos();
    report.state_before = await state();
    await page.screenshot({path: path.join(out, 'before.png')});
    const log = fs.openSync(path.join(out, 'ffmpeg.log'), 'w');
    recorder = spawn('ffmpeg', ['-hide_banner', '-y', '-f', 'x11grab', '-framerate', String(FPS),
      '-video_size', '1920x1080', '-i', process.env.DISPLAY || ':99', '-t', String(SECONDS),
      '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18', '-threads', '2', '-pix_fmt', 'yuv420p',
      '-movflags', '+faststart', path.join(out, 'live-ui.mp4')], {stdio: ['ignore', 'ignore', log]});
    const finished = new Promise((resolve, reject) => {recorder.on('error', reject); recorder.on('exit', code => code === 0 ? resolve() : reject(Error(`FFmpeg exited ${code}`)));});
    // FFmpeg opens X11 before controls start; the first second shows the settled layout.
    await sleep(500);
    const start = Date.now(); report.started = start;
    const at = async seconds => {
      await sleep(Math.max(0, start + seconds * 1000 - Date.now()));
      if ((await state()).setup_revision !== original.setup_revision) throw Error('Demo changed during capture; leave the new deployment untouched');
    };
    await at(1); await selectScene('random_181');
    await at(2); await direct(false); await selectScene('pip_128');
    await at(4); await stroke(origin, 0.5, 2000);
    await at(7); await stroke(0.5, 1 - origin, 2000);
    await at(10); await selectScene('random_180');
    await at(12); await stroke(1 - origin, origin, 4000);
    await at(18); await page.mouse.move(20, 20);
    await finished;
    report.after = await videos(); report.state_after = await state();
    if (report.after.some((v, i) => v.frames - report.before[i].frames < 400)) throw Error('A viewer stopped advancing during capture');
    report.ok = true;
  } finally {
    if (recorder && recorder.exitCode === null) recorder.kill('SIGINT');
    if (changed && (await state()).setup_revision === original.setup_revision) {
      await command({command:'cut', scene: original.status.pgm_scene}); await sleep(1000);
      await command({command:'preview', scene: original.status.pvw_scene});
      for (const key of original.dsk) await command({command:'dsk', key:key.id, on:key.on, fade_seconds:0});
    }
    fs.writeFileSync(path.join(out, 'capture.json'), JSON.stringify(report, null, 2));
    await browser.close();
  }
})().catch(error => {console.error(error); process.exit(1);});
