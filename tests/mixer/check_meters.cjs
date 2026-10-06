// Playwright; all requests are intercepted, so this never controls a live mixer.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');

(async () => {
  const browser = await chromium.launch(process.env.CHROMIUM ? {executablePath: process.env.CHROMIUM} : {});
  try {
    const page = await browser.newPage({viewport: {width: 1920, height: 1080}}), errors = [];
    page.on('pageerror', e => errors.push(e.message));
    const price = {hourly_usd: 1.2};
    const gpu = {index: 0, gpu: 49, decoder: 66, encoder: 75, memory_used_mib: 18432, memory_total_mib: 24576,
      power_draw_w: 50, power_limit_w: 72, encoder_sessions: 30, encoder_fps: 750, encoder_mpix_s: 1555.2};
    const cuts = [74, 90, 106].map((ms, i) => ({id: i + 1, ms, state: 'measured'}));
    const aux = Array.from({length: 30}, (_, i) => ({id: `aux${i}`, label: `Aux ${i}`, layout: {preset: 'cells'},
      layouts: [], cells: [], canvas: {w: 1920, h: 1080}, fps: 25, running: true,
      playout: {frames: 100, missed_deadlines: i === 29 ? 2 : 0}, output_drops: i === 29 ? 1 : 0}));
    const sample = {setup_revision: 1, gpus: [gpu], host: {load1: 12, vcpus: 16, cpu_pct: 55,
      mixer_cpu_pct: 600, thread_pct: 40, thread_name: 'mixer_comp_a'},
      status: {pgm_scene: 'a', pvw_scene: 'b', transition: 'idle', cut_latency: {direct: {...cuts[2], recent: cuts}},
        playout: {A: {missed_deadlines: 1}, B: {missed_deadlines: 0}}, wipe_cache: {bytes: 200000000, budget_bytes: 600000000}},
      scenes: ['a', 'b'], aux_buses: aux,
      settings: {source_count: 192, source_counts: {video: 120, browser: 40, nv12: 32},
        canvas: {width: 1920, height: 1080, fps: 25, working_format: 'nv12'}, preview_codecs: ['h264'],
        aux_buses: aux.map(a => a.id), preview_outputs: aux.map((a, i) => ({bus: a.id, label: a.label,
          codec: 'h264', mountpoint: 5000 + 4 * i, fps: 25}))}};
    const root = path.join(__dirname, '../../pyplumber/mixer/gui/assets');
    await page.route('**/*', route => {
      const url = new URL(route.request().url());
      if (url.pathname === '/api/state') return route.fulfill({json: sample});
      if (url.pathname === '/outputs.js') return route.fulfill({contentType: 'text/javascript', body: fs.readFileSync(path.join(root, 'outputs.js'), 'utf8')});
      if (url.pathname === '/') return route.fulfill({contentType: 'text/html', body: fs.readFileSync(path.join(root, 'index.html'), 'utf8')
        .replace('<script id="config" type="application/json">{}</script>',
          `<script id="config" type="application/json">${JSON.stringify({compute_price: price, preview_base: '/preview/'})}</script>`)});
      return route.fulfill({contentType: 'text/html', body: ''});
    });
    await page.goto('http://mixer.test/');
    await page.waitForFunction(() => document.querySelector('#power b').textContent === '50 / 72 W');
    const values = id => page.locator(`#${id}-tip dd`).allTextContents();
    assert.deepEqual(await values('power'), ['3.84', '1555.2', '31.10', '$0.00625']);
    assert.deepEqual(await values('gpu-vram'), ['10.67', '6.00 GiB']);
    assert.deepEqual(await values('gpu-enc'), ['30', '750']);
    assert.deepEqual(await values('cpu'), ['6.00', '32.0']);
    assert.deepEqual(await values('lat'), ['106.0 ms', '3', '1']);
    await page.evaluate(() => { observeCuts(); observeCuts(); renderLatency(); });
    assert.match(await page.locator('#lat-tip').textContent(), /3 cuts/, 'repeated samples are not new cuts');
    for (const width of [1920, 1600, 1440, 1280]) {
      await page.setViewportSize({width, height: 1080});
      const box = await page.locator('#top').evaluate(e => ({width: e.clientWidth, scroll: e.scrollWidth, height: e.offsetHeight}));
      assert(box.height < 35, `header must be one row at ${width}px`);
      assert(box.scroll <= box.width + 1, `header must fit ${width}px: ${JSON.stringify(box)}`);
      if (process.env.METER_SCREENSHOTS) await page.screenshot({path: path.join(process.env.METER_SCREENSHOTS, `meters-${width}.png`)});
      for (const id of ['power', 'gpu-vram', 'gpu-enc', 'cpu', 'lat']) {
        await page.locator(`#${id}`).hover();
        const tip = await page.locator(`#${id}-tip`).boundingBox();
        assert(tip && tip.x >= 0 && tip.x + tip.width <= width, `${id} tooltip must fit`);
      }
    }
    await page.setViewportSize({width: 1920, height: 1080});
    await page.keyboard.press('Escape');
    const selector = page.locator('.viewer > header > .dd').first();
    await selector.locator('.dd-btn').click();
    const menu = selector.locator('.dd-menu');
    assert.equal(await menu.locator('[role="option"]').count(), 31, 'program plus every AUX');
    assert(await menu.evaluate(e => e.scrollHeight > e.clientHeight), 'long output list scrolls');
    await page.keyboard.press('End');
    assert.match(await page.locator(':focus').textContent(), /Aux 29/);
    assert(await menu.evaluate(e => e.scrollTop > 0));
    await page.keyboard.press('Enter');
    assert.match(await selector.locator('.dd-btn').textContent(), /Aux 29/);
    await selector.locator('.dd-btn').click();
    await page.keyboard.press('Home');
    assert.match(await page.locator(':focus').textContent(), /Program/);
    await page.keyboard.press('Escape');
    // Missing readings remain unknown; all-board ratios must not silently use one board.
    await page.evaluate(() => {
      state.gpus.push({...state.gpus[0], index: 1, power_draw_w: null, encoder_mpix_s: null});
      state.host.mixer_cpu_pct = null;
      renderMeters(); renderHost();
    });
    assert.deepEqual(await values('power'), ['—', '—', '—', '$0.00625']);
    assert.deepEqual(await values('cpu'), ['—', '—']);
    await page.evaluate(() => {
      state.setup_revision = 2;
      state.status.cut_latency = {direct: {id: 0, state: 'waiting', recent: []}};
      observeCuts(); renderLatency();
    });
    assert.equal((await values('lat'))[0], '—', 'restart clears prior cut history');
    await page.setViewportSize({width: 390, height: 844});
    await page.locator('#power').focus();
    const tip = await page.locator('#power-tip').boundingBox();
    assert(tip.x >= 0 && tip.x + tip.width <= 390);
    assert.deepEqual(errors, []);
    console.log('Compact meters, efficiency values, latency counters and 30-AUX selector passed');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exit(1); });
