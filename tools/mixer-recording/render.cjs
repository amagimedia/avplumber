// Replay host-recorded pictures and sampled mixer status in the matching real control page.
const fs = require('fs'), http = require('http');
const {chromium} = require('playwright');
const root = process.cwd();
const capture = JSON.parse(fs.readFileSync(root + '/capture.json'));
const report = JSON.parse(fs.readFileSync(root + '/report.json'));
const FPS = capture.fps, FRAMES = capture.frames;
const latest = (samples, wall) => samples.findLast(s => s.wall <= wall) || samples[0];
let current = latest(capture.states, capture.demo_start_wall).state;
const server = http.createServer((req, res) => {
  const path = req.url.split('?')[0];
  let data, type = 'text/html';
  if (path === '/') data = fs.readFileSync(root + '/ui.html');
  else if (path === '/outputs.js') { data = fs.readFileSync(root + '/outputs.js'); type = 'text/javascript'; }
  else if (path === '/api/state') { data = JSON.stringify(current); type = 'application/json'; }
  else if (/^\/frames\/(pgm|preview)\/\d{3,}\.png$/.test(path)) { data = fs.readFileSync(root + path); type = 'image/png'; }
  else data = '<!doctype html><body style="margin:0;background:#000">';
  res.writeHead(200, {'Content-Type': type}); res.end(data);
});
(async () => {
  await new Promise(r => server.listen(48774, '127.0.0.1', r));
  fs.mkdirSync(root + '/ui-frames', {recursive: true});
  const browser = await chromium.launch();
  const page = await browser.newPage({viewport: {width: 1920, height: 1080}, deviceScaleFactor: 1});
  page.on('pageerror', e => { throw e; });
  await page.addInitScript(() => { localStorage.setItem('avplumber.mixer.viewers', JSON.stringify(['mv', 'program'])); window.setInterval = () => 0; });
  await page.goto('http://127.0.0.1:48774/');
  await page.waitForFunction(() => document.querySelectorAll('.viewer').length === 2 && document.querySelector('#counts').textContent.includes('192'));
  await page.evaluate((fps) => {
    // No wall-clock animation or further polling may advance an offline frame.
    window.requestAnimationFrame = () => 0;
    refresh = async () => {};
    for (const v of viewers) {
      v.frame.style.display = 'none'; v.img = document.createElement('img');
      v.img.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;object-fit:fill'; v.fit.insertBefore(v.img, v.ovl);
      v.statsRow.innerHTML = `<span>INSTANCE RECORDING</span><span>${fps} FPS</span>`;
    }
    directTouched = tookTransition = true;
    turnTbar(true);
    selectTransition('fade');
  }, FPS);
  const check = [];
  for (let n = 0; n < FRAMES; n++) {
    const wall = report.pgm.first_frame_wall + n / FPS;
    const status = latest(capture.statuses, wall).status;
    current = {...latest(capture.states, wall).state, status};
    // The control selection follows the recorded commands, not an inferred image-change ranking.
    const event = capture.events.findLast(e => e.wall <= wall && ['cut', 'preview'].includes(e.command.command));
    const manual = capture.events.some(e => e.wall <= wall && e.wall >= capture.demo_start_wall && e.command.command === 'preview');
    const drawn = await page.evaluate(async ({snapshot, selectedScene, manual, n, fps}) => {
      state = snapshot; selected = selectedScene;
      setDirect(!manual);
      mix = gripBack = parking = null;
      lever.kind = 'crossfade';
      lever.sync(state.status.lever, true);
      render(); renderLever();
      for (const [i, v] of viewers.entries()) {
        v.frame.style.display = 'none'; v.statsRow.innerHTML = `<span>INSTANCE RECORDING</span><span>${fps} FPS</span>`;
        v.img.src = `/frames/${i === 0 ? 'preview' : 'pgm'}/${String(n + 1).padStart(3, '0')}.png`;
      }
      await Promise.all(viewers.map(v => v.img.decode()));
      return {position: gripShown, progress: document.querySelector('.tbar .pct').textContent};
    }, {snapshot: current, selectedScene: event.command.scene, manual, n, fps: FPS});
    check.push({frame: n, ...drawn, sampled_position: status.lever.position});
    await page.screenshot({path: root + `/ui-frames/${String(n).padStart(3, '0')}.png`, animations: 'disabled'});
    if (n % 60 === 0) console.log('Rendered', n, '/', FRAMES);
  }
  fs.writeFileSync(root + '/render-check.json', JSON.stringify(check));
  await browser.close(); server.close(); console.log('Rendered', FRAMES, 'frames');
})().catch(e => { console.error(e); process.exit(1); });
