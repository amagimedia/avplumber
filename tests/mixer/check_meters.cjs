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
          .map(([name, count, pct, max_pct]) => ({name, count, pct, max_pct}))},
      status: {pgm_scene: 'a', pvw_scene: 'b', transition: 'idle', cut_latency: {direct: {...cuts[2], recent: cuts}},
        playout: {A: {missed_deadlines: 1}, B: {missed_deadlines: 0}}, wipe_cache: {bytes: 200000000, budget_bytes: 600000000}},
      scenes: ['a', 'b'], aux_buses: aux,
      settings: {source_count: 192, source_counts: {video: 120, browser: 40, nv12: 32},
        canvas: {width: 1920, height: 1080, fps: 30, working_format: 'nv12'}, preview_codecs: ['h264'],
        aux_buses: aux.map(a => a.id), preview_outputs: aux.map((a, i) => ({bus: a.id, label: a.label,
          codec: 'h264', mountpoint: 5000 + 4 * i, fps: 25}))}};
    const root = path.join(__dirname, '../../pyplumber/mixer/gui/assets'), commands = [];
    await page.route('**/*', route => {
      const url = new URL(route.request().url());
      if (url.pathname === '/api/state') return route.fulfill({json: sample});
      if (url.pathname === '/api/command') {
        const command = route.request().postDataJSON();
        commands.push(command);
        if (command.command === 'preview') sample.status.pvw_scene = command.scene;
        if (command.command === 'dsk' && command.prepare) sample.dsk.forEach(key => { key.next_take = command.keys.includes(key.id); });
        else if (command.command === 'dsk' && command.keys)
          sample.dsk.filter(key => command.keys.includes(key.id)).forEach(key => { key.on = command.toggle ? !key.on : command.on; });
        else if (command.command === 'dsk') sample.dsk.find(key => key.id === command.key).on = command.on;
        if (command.command === 'ftb') sample.ftb = {on: !sample.ftb.on, phase: sample.ftb.on ? 'off' : 'on'};
        return route.fulfill({json: {ok: true}});
      }
      if (url.pathname === '/outputs.js') return route.fulfill({contentType: 'text/javascript', body: fs.readFileSync(path.join(root, 'outputs.js'), 'utf8')});
      if (url.pathname === '/') return route.fulfill({contentType: 'text/html', body: fs.readFileSync(path.join(root, 'index.html'), 'utf8')
        .replace('<script id="config" type="application/json">{}</script>',
          `<script id="config" type="application/json">${JSON.stringify({compute_price: price, preview_base: '/preview/'})}</script>`)});
      return route.fulfill({contentType: 'text/html', body: ''});
    });
    await page.goto('http://mixer.test/');
    await page.waitForFunction(() => document.querySelector('#power b').textContent === '50 / 72 W');
    const values = id => page.locator(`#${id}-tip dd`).allTextContents();
    // One hover per meter: a meter with its big tip has no title of its own or of a group around it,
    // which the browser would show stacked over the tip.
    const titledTips = () => page.evaluate(() => [...document.querySelectorAll('.has-tip')]
      .filter((m) => m.closest('[title]')).map((m) => m.id));
    assert.deepEqual(await titledTips(), []);
    assert.deepEqual(await page.evaluate(() => ['thread', 'lat', 'cpu', 'power', 'gpu-enc', 'gpu-vram']
      .filter((id) => !document.getElementById(id).classList.contains('has-tip'))), []);
    assert.deepEqual(await values('power'), ['3.84', '1555.2', '31.10', '$0.00625']);
    assert.deepEqual(await values('gpu-vram'), ['10.67', '6.00 GiB']);
    assert.deepEqual(await values('gpu-enc'), ['30', '750']);
    assert.deepEqual(await values('cpu'), ['6.00', '32.0']);
    // THREAD shows the busiest thread. Its tip is one table of the mixer's threads by kind, the
    // largest total first and "other" last: count, total, mean and busiest thread. Under it
    // Chromium's cores' worth in its header, its processes and its GPU process's threads.
    assert.equal(await page.locator('#thread b').textContent(), '40%');
    const tables = () => page.locator('#thread-tip table').evaluateAll((all) => all.map((t) =>
      [...t.rows].map((r) => [...r.cells].map((c) => c.textContent))));
    assert.deepEqual(await tables(), [
      [['Thread kind', 'n', 'total', 'avg', 'max'],
        ['input_N', '152', '141%', '0.9%', '3.3%'], ['upload_N', '47', '135%', '2.9%', '3.3%'],
        ['mixer_comp_a', '1', '40%', '40%', '40%'], ['EventLoop', '5', '37%', '7.4%', '8.2%'],
        ['mixer_color_sdr', '105', '17%', '0.2%', '0.4%'], ['other', '1161', '230%', '0.2%', '38%']],
      [['Process type', 'n', 'total', 'avg', 'max'],
        ['GPU process', '2', '80%', '40%', '52%'], ['renderer', '24', '52%', '2.2%', '6.3%'],
        ['browser', '2', '13%', '6.3%', '9.0%']],
      [['Thread kind', 'n', 'total', 'avg', 'max'],
        ['viz compositor', '2', '39%', '19%', '24%'], ['GPU main', '2', '30%', '15%', '20%'],
        ['IO', '2', '6.2%', '3.1%', '3.5%'], ['other', '40', '5.6%', '0.1%', '1.1%']]]);
    // A server that names only the busiest thread, and no browser figures: one table with that thread.
    await page.evaluate(() => { const {thread_kinds: all} = state.host, chromium = state.chromium;
      state.host.thread_kinds = undefined; state.chromium = null; renderHost();
      window.oneThread = [...document.querySelectorAll('#thread-tip table')].map((t) => t.textContent);
      state.host.thread_kinds = all; state.chromium = chromium; renderHost(); });
    assert.deepEqual(await page.evaluate(() => window.oneThread), ['Thread kindntotalavgmaxmixer_comp_a140%40%40%']);
    // LAT's tip: two short tables, delivery and the browser sources; values are the second column.
    const lat = () => page.locator('#lat-tip td:nth-child(2)').allTextContents();
    const latLabels = () => page.locator('#lat-tip td:first-child').allTextContents();
    const latTitle = () => page.locator('#lat-tip strong').textContent();
    assert.deepEqual(await lat(), ['106.0 ms', '3', '1']);
    // No player is open: the bar shows the median of the mixer's own last three cuts.
    assert.equal(await page.locator('#lat b').textContent(), '90ms');
    await page.evaluate(() => { observeCuts(); observeCuts(); renderLatency(); });
    assert.deepEqual((await latLabels())[0], 'Cut p95 · 3', 'repeated samples are not new cuts');
    // Without a server window the totals since the start; with one, its figures and its span.
    assert.match(await latTitle(), /· total$/);
    await page.evaluate(() => { state.recent = {window_s: 600, covered_s: 754, missed_deadlines: 0, output_drops: 2}; renderLatency(); });
    assert.deepEqual((await lat()).slice(1), ['0', '2']);
    assert.match(await latTitle(), /· 10 min$/);
    await page.evaluate(() => { state.recent.covered_s = 40; renderLatency(); });
    assert.match(await latTitle(), /· 40 s$/);
    // The other inputs in delivery; the browser sources in their own table with the worst and the drops.
    await page.evaluate(() => { state.recent.inputs = {
      browser: {sources: 40, repeats: 120, dropped: 45, worst: {source: 'web_12', repeats: 40, dropped: 2}},
      other: {sources: 152, repeats: 5, dropped: 1}};
      state.browser_drops = {window_s: 600, covered_s: 600, windows: 40, dropped: 12,
        reasons: [['retained_pool_full', 9], ['no_consumer', 3]]}; renderLatency(); });
    assert.deepEqual(await page.locator('#lat-tip th:first-child').allTextContents(), ['Delivery', 'Browser sources · 40']);
    assert.deepEqual(await latLabels(), ['Cut p95 · 3', 'Deadline misses', 'AUX drops', 'PGM video repeats / skips',
      'PGM repeats / skips', 'Worst']);
    assert.deepEqual((await lat()).slice(3), ['5 / 1', '120 / 45', 'web_12 40 / 2']);
    await page.evaluate(() => { state.recent.inputs.browser = {sources: 40, repeats: 0, dropped: 0, worst: null};
      state.browser_drops = null; renderLatency(); });
    assert.deepEqual((await lat()).slice(4), ['0 / 0', '—']);
    await page.evaluate(() => { state.recent = null; renderLatency(); });
    assert.deepEqual(await lat(), ['106.0 ms', '3', '1']);
    sample.recent = {window_s: 600, covered_s: 40, inputs: {browser: {sources: 40,
      repeats: 13, dropped: 13, worst: {source: 'browser_009', repeats: 2, dropped: 2}}}};
    await page.evaluate(recent => { state.recent = recent; renderLatency(); }, sample.recent);
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
        if (id === "thread" && process.env.METER_SCREENSHOTS) await page.screenshot({path: path.join(process.env.METER_SCREENSHOTS, `thread-hover-${width}.png`)});
        if (id === "lat") {
          assert(await page.locator("#lat-tip").evaluate(t => t.scrollWidth <= t.clientWidth), "delivery text must stay inside tooltip");
          if (process.env.METER_SCREENSHOTS) await page.screenshot({path: path.join(process.env.METER_SCREENSHOTS, `lat-hover-${width}.png`)});
        }
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
    // Two GPUs: the meters with a tip list both in it, still without a title.
    assert.deepEqual(await titledTips(), []);
    assert.deepEqual((await values('gpu-enc')).slice(2), ['75%', '75%']);
    assert.deepEqual(await values('cpu'), ['—', '—']);
    await page.evaluate(() => {
      state.setup_revision = 2;
      state.status.cut_latency = {direct: {id: 0, state: 'waiting', recent: []}};
      observeCuts(); renderLatency();
    });
    assert.equal((await page.locator('#lat-tip td:nth-child(2)').allTextContents())[0], '—', 'restart clears prior cut history');
    assert.equal(await page.locator('#lat b').textContent(), '—', 'and the bar has no cut to show');
    // On-air shortcuts are checked only against this mocked mixer.
    await page.evaluate(() => { selected = 'a'; setDirect(true); armedSlot = 0; restFocus(); });
    await Promise.all([page.waitForResponse(r => r.url().endsWith('/api/command')), page.keyboard.press('ArrowRight')]);
    assert.deepEqual(commands.splice(0), [{command: 'preview', scene: 'b'}], 'arrows bypass Direct and slot assignment');
    await page.keyboard.press('ArrowRight');
    await page.waitForFunction(() => state.status.pvw_scene === 'a');
    assert.deepEqual(commands.splice(0), [{command: 'preview', scene: 'a'}], 'arrows wrap visible scenes');
    await page.evaluate(() => { selected = 'b'; currentTransition = 'fade'; tookTransition = true; armedSlot = null; });
    await Promise.all([page.waitForResponse(r => r.url().endsWith('/api/command')), page.keyboard.press('Space')]);
    assert.equal(commands.length, 1);
    assert.equal(commands[0].command, 'fade');
    assert.equal(commands[0].scene, 'a', 'Space takes Preview even when selection differs');
    assert.equal(commands[0].duration_sec, 0.5);
    commands.length = 0;
    // Choosing a transition only prepares Auto (a switcher's transition type); Cut and Auto take
    // the Preview the mixer shows, not the last pick.
    await page.evaluate(() => { currentTransition = 'cut'; selected = 'b'; restFocus(); });
    await page.keyboard.press('f');
    assert.equal(await page.evaluate(() => currentTransition), 'fade', 'F selects fade');
    assert.deepEqual(commands, [], 'F does not take');
    assert.equal(await page.locator('#auto').textContent(), 'Auto · fade 15f', 'Auto shows what it will run, at its rate');
    // Clicking the selector must not leave Space to it: Space after a click is Auto, not a second selection.
    await page.locator('#fade').click();
    await Promise.all([page.waitForResponse(r => r.url().endsWith('/api/command')), page.keyboard.press('Space')]);
    assert.deepEqual(commands.splice(0).map(c => c.command), ['fade'], 'Space after clicking Fade runs Auto');
    await Promise.all([page.waitForResponse(r => r.url().endsWith('/api/command')), page.keyboard.press('c')]);
    assert.deepEqual(commands.splice(0).map(c => [c.command, c.scene]), [['cut', 'a']], 'C cuts the Preview');
    // The lever sends its position with the mix's duration; Program shows the mix's progress.
    await Promise.all([page.waitForResponse(r => r.url().endsWith('/api/command')),
      page.evaluate(() => { $('lever').value = 400; $('lever').dispatchEvent(new Event('input')); })]);
    assert.deepEqual(commands.splice(0).map(c => [c.command, c.position, c.duration_sec]), [['lever', 0.4, 0.5]]);
    await page.evaluate(() => {
      state.status = {...state.status, transition: 'crossfade', transition_on_air: true, transition_manual: true,
                      transition_progress: 0.4, transition_duration_ms: 500};
      progressSample = {at: performance.now(), progress: 0.4}; renderBuses();
    });
    assert.equal(await page.locator('#pgm-state').textContent(), 'on air · crossfade 40% · lever');
    await page.evaluate(() => { state.status = {...state.status, transition: 'idle'}; renderBuses(); });
    await page.evaluate(() => { setDirect(true); restFocus(); });
    assert.equal(await page.locator('#direct').textContent(), 'Direct cut: on');
    await Promise.all([page.waitForResponse(r => r.url().endsWith('/api/command')), page.keyboard.press('2')]);
    assert.equal(commands.splice(0)[0].command, 'cut', 'a Direct pick cuts whatever Auto is set to');
    await page.evaluate(() => { setDirect(false); restFocus(); });
    sample.dsk = [{id: 'logo', on: true}, {id: 'title', on: false}];
    await page.evaluate(async () => { await refresh(); restFocus(); });
    await page.waitForFunction(() => state.dsk?.length === 2);
    await page.keyboard.press('k');
    await page.waitForFunction(() => !keysPending && state.dsk.every(key => !key.on));
    assert.deepEqual(commands.map(c => [c.key, c.on]), [['logo', false]], 'mixed keys all go off');
    commands.length = 0;
    await page.keyboard.press('k');
    await page.waitForFunction(() => !keysPending && state.dsk.every(key => key.on));
    assert.deepEqual(commands.map(c => [c.key, c.on]).sort(), [['logo', true], ['title', true]], 'off keys all go on');
    assert(commands.every(c => c.fade_seconds === 0 && c.curve), 'key shortcuts use the key transition');
    commands.length = 0;
    // Next Transition: an armed key goes with the background take, on the engine's clock; with BKGD
    // off, Cut and Auto take the armed keys alone.
    const nextButton = n => page.locator(`#next-keys button:nth-child(${n})`);
    await nextButton(3).click();
    await page.waitForFunction(() => state.dsk[1].next_take === true);
    assert.deepEqual(commands.splice(0), [{command: 'dsk', keys: ['title'], prepare: true}], 'KEY 2 arms the second key');
    assert(await page.locator('#keys .key:nth-child(2)').evaluate(n => n.classList.contains('armed')), 'the Keys card shows it armed');
    assert(await nextButton(3).evaluate(n => n.classList.contains('air')), 'an on-air key shows its tally');
    await page.evaluate(() => restFocus());
    await Promise.all([page.waitForResponse(r => r.url().endsWith('/api/command')), page.keyboard.press('c')]);
    assert.deepEqual(commands.splice(0).map(c => c.command), ['cut'], 'with BKGD a take is the background take');
    await nextButton(1).click();
    assert.equal(await page.locator('#pvw-state').textContent(), 'Auto · keys');
    assert(await page.locator('#lever').isDisabled(), 'the lever mixes the background only');
    await Promise.all([page.waitForResponse(r => r.url().endsWith('/api/command')), page.keyboard.press('Space')]);
    assert.deepEqual(commands.splice(0).map(c => [c.command, c.keys, c.toggle, c.fade_seconds]), [['dsk', ['title'], true, 0.5]],
      'Space after BKGD is Auto, which fades the armed keys alone at the Fade rate');
    await page.evaluate(() => restFocus());
    await Promise.all([page.waitForResponse(r => r.url().endsWith('/api/command')), page.keyboard.press('c')]);
    assert.deepEqual(commands.splice(0).map(c => [c.command, c.fade_seconds]), [['dsk', 0]], 'Cut cuts them');
    await nextButton(3).click();
    await page.waitForFunction(() => !state.dsk[1].next_take);
    commands.length = 0;
    await page.locator('#cut').click();
    assert.match(await page.locator('#error').textContent(), /Arm BKGD or a key/);
    assert.deepEqual(commands, [], 'nothing armed takes nothing');
    await nextButton(1).click();
    assert(!(await page.locator('#lever').isDisabled()), 'BKGD gives the lever back');
    // FTB: hidden without the keyer; it fades the keyed program at the Fade rate and back, flashing while black.
    assert(await page.locator('#ftb').isHidden(), 'no keyer, no FTB');
    sample.ftb = {on: false, phase: 'off'};
    await page.evaluate(() => refresh());
    await page.waitForFunction(() => !$('ftb').hidden);
    await Promise.all([page.waitForResponse(r => r.url().endsWith('/api/command')), page.locator('#ftb').click()]);
    assert.deepEqual(commands.splice(0).map(c => [c.command, c.toggle, c.fade_seconds]), [['ftb', true, 0.5]]);
    await page.waitForFunction(() => $('ftb').classList.contains('black'));
    assert.equal(await page.locator('#ftb').getAttribute('aria-pressed'), 'true');
    assert.equal(await page.locator('#pgm-state').textContent(), 'on air · black');
    await Promise.all([page.waitForResponse(r => r.url().endsWith('/api/command')), page.keyboard.press('Space')]);
    assert.deepEqual(commands.splice(0).map(c => c.command), ['fade'], 'Space after FTB is Auto, not FTB again');
    await Promise.all([page.waitForResponse(r => r.url().endsWith('/api/command')), page.locator('#ftb').click()]);
    await page.waitForFunction(() => !$('ftb').classList.contains('black') && $('ftb').getAttribute('aria-pressed') === 'false');
    commands.length = 0;
    await page.evaluate(() => {
      for (const key of [' ', 'k']) window.dispatchEvent(new KeyboardEvent('keydown', {key, repeat: true}));
    });
    await page.locator('#fade-s').focus();
    await page.keyboard.press('k');
    await page.keyboard.press('Space');
    await page.keyboard.press('ArrowRight');
    await page.evaluate(() => refresh());
    assert.deepEqual(commands, [], 'typing and held take/key shortcuts never send commands');
    // Rates as on a switcher panel: frames below a second, seconds:frames from one; commands take seconds.
    const rate = async (text) => { await page.locator('#fade-s').fill(text); return page.locator('#auto').textContent(); };
    assert.equal(await rate('45'), 'Auto · fade 1:15', 'from a second the rate shows as seconds:frames');
    await page.locator('#fade-s').press('Enter');
    assert.equal(await page.locator('#fade-s').inputValue(), '1:15', 'an entry shows as the panel would');
    await Promise.all([page.waitForResponse(r => r.url().endsWith('/api/command')), page.keyboard.press('Space')]);
    assert.deepEqual(commands.splice(0).map(c => [c.command, c.duration_sec]), [['fade', 1.5]], 'the command takes seconds');
    await page.locator('#fade-s').press('ArrowDown');
    assert.equal(await page.locator('#fade-s').inputValue(), '1:14', 'arrow keys step a frame');
    await page.evaluate(() => restFocus());
    await page.locator('#fade-ctl [aria-label="One frame longer"]').click();
    assert.equal(await page.locator('#fade-s').inputValue(), '1:15', 'the small arrows step a frame');
    assert(await page.evaluate(() => document.activeElement === $('scenes')), 'and leave the focus where it was');
    await page.locator('#fade-s').hover();
    await page.mouse.wheel(0, 100);
    await page.waitForFunction(() => $('fade-s').value === '1:14');
    for (const _ of [1, 2]) await page.mouse.wheel(0, -100);
    await page.waitForFunction(() => $('fade-s').value === '1:16');
    await page.mouse.wheel(0, 10);
    assert.equal(await page.locator('#fade-s').inputValue(), '1:16', 'the wheel steps a frame per notch, not per small delta');
    await rate('1f');
    await page.locator('#fade-s').press('ArrowDown');
    assert.equal(await page.locator('#fade-s').inputValue(), '1f', 'a fade runs at least a frame');
    for (const bad of ['1:30', '0.5', '1:']) {
      await page.evaluate(() => showError(''));
      assert.equal(await rate(bad), 'Auto · fade', `no rate shown for ${bad}`);
      await page.locator('#fade-s').press('Enter');
      await page.keyboard.press('Space');
      await page.waitForFunction(() => $('error').textContent.includes('seconds:frames'));
      assert.deepEqual(commands, [], `${bad} sends nothing`);
    }
    assert.equal(await rate('15'), 'Auto · fade 15f');
    assert.equal(await page.evaluate(() => { $('key-fade-s').value = '1:00'; return keyTransition().fade_seconds; }), 1,
      'the key fade reads seconds:frames too');
    await page.locator('#key-fade-s').fill('1f');
    await page.locator('#key-fade-s').press('ArrowDown');
    assert.equal(await page.locator('#key-fade-s').inputValue(), '', 'a key fade steps down to a cut');
    await page.locator('#key-fade-s').press('Escape');
    await page.locator('#hint').hover();
    assert.match(await page.locator('#hint-tip').innerText(), /Browse Preview only/);
    assert.match(await page.locator('#hint-tip').innerText(), /Auto: fade to Preview/);
    const shortcutTip = await page.locator('#hint-tip').boundingBox();
    assert(shortcutTip.y >= 0 && shortcutTip.y + shortcutTip.height <= 1080, 'shortcut hover fits above the bottom bar');
    await page.setViewportSize({width: 390, height: 844});
    await page.locator('#power').focus();
    const tip = await page.locator('#power-tip').boundingBox();
    assert(tip.x >= 0 && tip.x + tip.width <= 390);
    assert.deepEqual(errors, []);
    console.log('Compact meters, efficiency values, latency counters and 30-AUX selector passed');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exit(1); });
