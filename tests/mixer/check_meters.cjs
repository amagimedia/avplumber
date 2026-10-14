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
      mixer_cpu_pct: 600, thread_pct: 40, thread_name: 'mixer_comp_a',
      thread_count: 1472, thread_kinds: [['input_N', 152, 140.5, 3.3], ['upload_N', 47, 135, 3.3],
        ['mixer_comp_a', 1, 40, 40], ['EventLoop', 5, 37.2, 8.2], ['mixer_color_sdr', 105, 17.2, 0.4],
        ['other', 1161, 230.1, 38]].map(([name, count, pct, max_pct]) =>
          ({name, count, pct, avg_pct: Math.round(pct / count * 10) / 10, max_pct}))},
      chromium: {total_pct: 148.2, cores_used: 1.48, vcpus: 16,
        processes: [['GPU process', 2, 80.4, 52.1], ['renderer', 24, 52, 6.3], ['browser', 2, 12.6, 9]]
          .map(([name, count, pct, max_pct]) => ({name, count, pct, max_pct})),
        gpu_threads: [['viz compositor', 2, 38.5, 24], ['GPU main', 2, 30.1, 19.8], ['IO', 2, 6.2, 3.5], ['other', 40, 5.6, 1.1]]
          .map(([name, count, pct, max_pct]) => ({name, count, pct, max_pct})),
        busiest_cores: [{cpu: 3, pct: 62, host_pct: 91.5}, {cpu: 7, pct: 40.2, host_pct: 70}]},
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
    // THREAD shows the busiest thread. Its tip is one table of the mixer's threads by kind, the
    // largest total first and "other" last: count, total, mean and busiest thread. Under it
    // Chromium's processes, its GPU process's threads and its busiest cores.
    assert.equal(await page.locator('#thread b').textContent(), '40%');
    const tables = () => page.locator('#thread-tip table').evaluateAll((all) => all.map((t) =>
      [...t.rows].map((r) => [...r.cells].map((c) => c.textContent))));
    assert.deepEqual(await tables(), [
      [['Mixer threads', 'n', 'total', 'avg', 'max'],
        ['input_N', '152', '141%', '0.9%', '3.3%'], ['upload_N', '47', '135%', '2.9%', '3.3%'],
        ['mixer_comp_a', '1', '40%', '40%', '40%'], ['EventLoop', '5', '37%', '7.4%', '8.2%'],
        ['mixer_color_sdr', '105', '17%', '0.2%', '0.4%'], ['other', '1161', '230%', '0.2%', '38%']],
      [['Chromium · 1.48/16 cores', 'n', 'total', 'avg', 'max'],
        ['GPU process', '2', '80%', '40%', '52%'], ['renderer', '24', '52%', '2.2%', '6.3%'],
        ['browser', '2', '13%', '6.3%', '9.0%']],
      [['GPU process threads', 'n', 'total', 'avg', 'max'],
        ['viz compositor', '2', '39%', '19%', '24%'], ['GPU main', '2', '30%', '15%', '20%'],
        ['IO', '2', '6.2%', '3.1%', '3.5%'], ['other', '40', '5.6%', '0.1%', '1.1%']],
      [["Chromium's busiest cores", 'Chromium', 'host'], ['cpu3', '62%', '92%'], ['cpu7', '40%', '70%']]]);
    // A server that names only the busiest thread, and no browser figures: one table with that thread.
    await page.evaluate(() => { const {thread_kinds: all} = state.host, chromium = state.chromium;
      state.host.thread_kinds = undefined; state.chromium = null; renderHost();
      window.oneThread = [...document.querySelectorAll('#thread-tip table')].map((t) => t.textContent);
      state.host.thread_kinds = all; state.chromium = chromium; renderHost(); });
    assert.deepEqual(await page.evaluate(() => window.oneThread), ['Mixer threadsntotalavgmaxmixer_comp_a140%40%40%']);
    assert.deepEqual(await values('lat'), ['106.0 ms', '3', '1']);
    // No player is open: the bar shows the median of the mixer's own last three cuts.
    assert.equal(await page.locator('#lat b').textContent(), '90ms');
    await page.evaluate(() => { observeCuts(); observeCuts(); renderLatency(); });
    assert.match(await page.locator('#lat-tip').textContent(), /3 cuts/, 'repeated samples are not new cuts');
    // A server that counts the last ten minutes: its figures replace the totals since the start.
    assert.match(await page.locator('#lat-tip').textContent(), /Deadline misses · total/);
    await page.evaluate(() => { state.recent = {window_s: 600, covered_s: 754, missed_deadlines: 0, output_drops: 2}; renderLatency(); });
    assert.deepEqual((await values('lat')).slice(1), ['0', '2']);
    assert.match(await page.locator('#lat-tip').textContent(), /Deadline misses · last 10 min.*AUX output drops · last 10 min/s);
    await page.evaluate(() => { state.recent.covered_s = 40; renderLatency(); });
    assert.match(await page.locator('#lat-tip').textContent(), /Deadline misses · last 40 s/);
    await page.evaluate(() => { state.recent = null; renderLatency(); });
    assert.deepEqual(await values('lat'), ['106.0 ms', '3', '1']);
    for (const width of [1920, 1600, 1440, 1280]) {
      await page.setViewportSize({width, height: 1080});
      const box = await page.locator('#top').evaluate(e => ({width: e.clientWidth, scroll: e.scrollWidth, height: e.offsetHeight}));
      assert(box.height < 35, `header must be one row at ${width}px`);
      assert(box.scroll <= box.width + 1, `header must fit ${width}px: ${JSON.stringify(box)}`);
      if (process.env.METER_SCREENSHOTS) await page.screenshot({path: path.join(process.env.METER_SCREENSHOTS, `meters-${width}.png`)});
      for (const id of ['power', 'gpu-vram', 'gpu-enc', 'cpu', 'thread', 'lat']) {
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
    assert.equal(await page.locator('#lat b').textContent(), '—', 'and the bar has no cut to show');
    await page.setViewportSize({width: 390, height: 844});
    await page.locator('#power').focus();
    const tip = await page.locator('#power-tip').boundingBox();
    assert(tip.x >= 0 && tip.x + tip.width <= 390);
    assert.deepEqual(errors, []);
    console.log('Compact meters, efficiency values, latency counters and 30-AUX selector passed');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exit(1); });
