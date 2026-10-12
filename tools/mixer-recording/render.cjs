// Replays the capture in the real control page: each output frame is the page drawn with the
// recorded mixer state and the two recorded pictures of that instant. Run in a Playwright container
// with the capture directory as the working directory.
const fs = require('fs'), http = require('http');
const {chromium} = require('playwright');
const root = process.cwd(), FPS = 30, FRAMES = 10 * FPS;
const capture = JSON.parse(fs.readFileSync(root + '/capture.json'));
const report = JSON.parse(fs.readFileSync(root + '/report.json'));
const scenes = ['grid_16_080', 'random_180', 'grid_64_112', 'pip_128', 'alpha_overlay_224', 'grid_8_064', 'random_179'];
const cuts = [[0, scenes[0]], ...report.cut_frames.map((frame, i) => [frame, scenes[i + 1]])];
let current = capture.states.filter(s => s.wall < capture.demo_start_wall).at(-1).state;
const server = http.createServer((req, res) => {
  const path = req.url.split('?')[0];
  let data, type = 'text/html';
  if (path === '/') data = fs.readFileSync(root + '/ui.html');
  else if (path === '/outputs.js') { data = fs.readFileSync(root + '/outputs.js'); type = 'text/javascript'; }
  else if (path === '/api/state') { data = JSON.stringify(current); type = 'application/json'; }
  else if (/^\/frames\/(pgm|preview)\/\d{3}\.png$/.test(path)) { data = fs.readFileSync(root + path); type = 'image/png'; }
  else data = '<!doctype html><body style="margin:0;background:#000">';
  res.writeHead(200, {'Content-Type': type}); res.end(data);
});
(async () => {
  await new Promise(r => server.listen(48774, '127.0.0.1', r));
  fs.mkdirSync(root + '/ui-frames', {recursive: true});
  const browser = await chromium.launch();
  const page = await browser.newPage({viewport: {width: 1920, height: 1080}, deviceScaleFactor: 1});
  page.on('pageerror', e => console.log('page error:', e.message));
  await page.addInitScript(() => { localStorage.setItem('avplumber.mixer.viewers', JSON.stringify(['mv', 'program'])); window.setInterval = () => 0; });
  await page.goto('http://127.0.0.1:48774/');
  await page.waitForFunction(() => document.querySelectorAll('.viewer').length === 2 && document.querySelector('#counts').textContent.includes('192'));
  await page.evaluate((fps) => {
    for (const v of viewers) {
      v.frame.style.display = 'none'; v.img = document.createElement('img');
      v.img.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;object-fit:fill'; v.screen.insertBefore(v.img, v.ovl);
      v.statsRow.innerHTML = `<span>INSTANCE RECORDING</span><span>${fps} FPS</span>`;
    }
    directTouched = true; setDirect(true);
  }, FPS);
  let prior = null;
  for (let n = 0; n < FRAMES; n++) {
    const scene = cuts.filter(c => c[0] <= n).at(-1)[1];
    const snapshot = capture.states.find(s => s.state.status.pgm_scene === scene && s.wall >= capture.demo_start_wall - 6.5) || capture.states.at(-2);
    current = snapshot.state;
    await page.evaluate(async ({snapshot, scene, n, fps}) => {
      if (snapshot) { state = snapshot; selected = scene; render(); }
      for (const [i, v] of viewers.entries()) {
        v.frame.style.display = 'none'; v.statsRow.innerHTML = `<span>INSTANCE RECORDING</span><span>${fps} FPS</span>`;
        v.img.src = `/frames/${i === 0 ? 'preview' : 'pgm'}/${String(n + 1).padStart(3, '0')}.png`;
      }
      await Promise.all(viewers.map(v => v.img.decode()));
    }, {snapshot: prior === scene ? null : current, scene, n, fps: FPS});
    prior = scene;
    await page.screenshot({path: root + `/ui-frames/${String(n).padStart(3, '0')}.png`, animations: 'disabled'});
    if (n % 60 === 0) console.log('Rendered', n, '/', FRAMES);
  }
  await browser.close(); server.close(); console.log('Rendered', FRAMES, 'frames');
})().catch(e => { console.error(e); process.exit(1); });
