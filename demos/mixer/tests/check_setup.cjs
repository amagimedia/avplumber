// Run with Playwright installed (or NODE_PATH pointing to its node_modules) and python3 (or PYTHON).
// All HTTP requests are intercepted; this never applies settings to a live mixer.
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({args: ['--no-sandbox']});
  try {
    const page = await browser.newPage();
    const errors = [], submissions = [];
    let initialSettings = null;
    // The live show's own aux buses (Program preview and Multiviewer) and a Janus API for extra ones.
    const ownAux = [{id: 'mv', label: 'Program preview', full_rate: false, encode: {preset: 'p3', bitrate_kbps: 3000}},
      {id: 'mv2', label: 'Multiviewer', full_rate: false, encode: {preset: 'p3', bitrate_kbps: 3000}}];
    let auxStatus = {aux_buses: ownAux, janus_api: true};
    page.on('pageerror', error => errors.push(error.message));
    const html = fs.readFileSync(path.join(__dirname, '../setup.html'), 'utf8');
    // The status carries the instance's profile as webui.py serves it: the limits asserted below are tesla_t4's.
    const profile = JSON.parse(execFileSync(process.env.PYTHON || 'python3', ['-c',
      'import json; from instance_profiles import INSTANCE_PROFILES as p, InstanceType as t; print(json.dumps(p[t.TESLA_T4]))'],
      {cwd: path.join(__dirname, '..')}));
    await page.route('http://mixer.test/**', route => {
      if (route.request().url().endsWith('/api/setup')) {
        if (route.request().method() === 'POST') submissions.push(route.request().postDataJSON());
        return route.fulfill({json: {phase: 'idle', message: 'Ready', settings: initialSettings,
          instance_type: 'tesla_t4', profile, ...auxStatus}});
      }
      return route.fulfill({contentType: 'text/html', body: html});
    });
    const reset = async (settings = null) => {
      initialSettings = settings;
      await page.goto('http://mixer.test/setup/');
      await page.waitForFunction(() => !document.getElementById('apply').disabled);
    };
    const apply = async () => {
      const count = submissions.length;
      const reply = page.waitForResponse(response => response.url().endsWith('/api/setup') &&
        response.request().method() === 'POST');
      await page.locator('#apply').click();
      await reply;
      assert.equal(submissions.length, count + 1);
      return submissions.at(-1);
    };
    await reset();
    assert.match(await page.locator('#source-limit').textContent(), /Maximum 55 .* measured on tesla_t4\.$/);
    // Every encoded output has its row, at the profile's preset and setup_runtime.DEFAULT_KBPS or,
    // for an own aux bus, the status's encode; the clean feed counts even while off.
    const rows = () => page.locator('#encodes tr').evaluateAll(rows => rows.map(row =>
      [...row.querySelectorAll('b,small'), row.lastElementChild].map(cell => cell.textContent).join(' | ')));
    assert.deepEqual(await rows(), ['Program · SDR | H.264 · 60 fps | 13.1%', 'Program · HLG | HEVC · 60 fps | 26.3%',
      'Clean feed · SDR | H.264 · 60 fps · off, counted | 13.1%', 'Program preview | H.264 · 30 fps | 6.6%',
      'Multiviewer | H.264 · 30 fps | 6.6%', 'Extra aux · each | H.264 · 30 fps | 6.6%']);
    const preset = output => page.getByLabel(`${output} preset`, {exact: true});
    const bitrate = output => page.getByLabel(`${output} bitrate, Mbit/s`, {exact: true});
    assert.equal(await bitrate('Program · HLG').inputValue(), '8');
    await preset('Program · SDR').selectOption('p1');
    await bitrate('Program · SDR').fill('8.5');
    const p3 = kbps => ({preset: 'p3', bitrate_kbps: kbps});
    const defaults = {sdr: p3(6000), hdr: p3(8000), sdr_clean: p3(6000), mv: p3(3000), mv2: p3(3000), extra: p3(3000)};
    const edited = await apply();
    assert.deepEqual(edited.encodes, {...defaults, sdr: {preset: 'p1', bitrate_kbps: 8500}}, 'an encode edit alone must reach the backend');
    assert.equal('bitrate_kbps' in edited, false);
    for (const value of ['25', '1.5', '']) {
      await bitrate('Multiviewer').fill(value);
      assert.equal(await page.locator('#error').textContent(), 'Multiviewer: the bitrate must be from 2 to 20 Mbit/s.');
      assert.equal(await page.locator('#apply').isDisabled(), true, `${value || 'no'} Mbit/s cannot be applied`);
    }
    await bitrate('Multiviewer').fill('20');
    assert.equal((await apply()).encodes.mv2.bitrate_kbps, 20000);
    await reset(edited);
    assert.equal(await preset('Program · SDR').inputValue(), 'p1', 'saved encodes load');
    assert.equal(await bitrate('Program · SDR').inputValue(), '8.5');
    const {encodes: _encodes, ...withoutEncodes} = edited;
    await reset(withoutEncodes);
    assert.deepEqual((await apply()).encodes, defaults, 'settings saved before per-output encodes load the defaults');

    // Exercise actual keystrokes: re-rendering must not reinsert zero while editing.
    for (const id of ['sdr420', 'hlg420', 'hlg422', 'sdr422', 'browser', 'sdr420_raw', 'hlg420_raw']) {
      await reset();
      const input = page.locator(`#count-${id}`);
      await input.fill('0');
      await input.press('Backspace');
      assert.equal(await input.inputValue(), '', `${id}: clearing zero must leave an editable blank`);
      await input.press('2');
      assert.equal(await input.inputValue(), '2');
      await input.press('ArrowUp');
      assert.equal(await input.inputValue(), '3');
      await input.fill('');
      await input.press('Tab');
      assert.equal(await input.inputValue(), '0', `${id}: an empty count becomes zero on blur`);
      const setup = await apply();
      assert.equal(setup.weights[['sdr420', 'hlg420', 'hlg422', 'sdr422', 'browser', 'sdr420_raw', 'hlg420_raw'].indexOf(id)], 0);
    }

    await reset();
    await page.locator('#mode').selectOption('8:420');
    await page.locator('#count-browser').fill('0');
    await page.locator('#count-sdr420_raw').fill('0');   // the default 4:2:2 upload became NV12
    const onlySource = page.locator('#count-sdr420');
    await onlySource.fill('');
    assert.equal(await onlySource.inputValue(), '');
    assert.equal(await page.locator('#apply').isDisabled(), true, 'zero total cannot be applied');
    await onlySource.press('8');
    assert.equal((await apply()).source_count, 8, 'typing a replacement restores a valid setup');

    const sdr = {...submissions.at(-1), fps: 30, source_count: 82, scene_count: 192,
      browser_ring_size: 6, weights: [33, 0, 0, 0, 26, 23, 0]};
    await reset(sdr);
    await page.locator('#mode').selectOption('10:420');
    const hdr = await apply();
    // Half of each 4:2:0 path turns HLG; P010 costs two upload units, so one upload moves to a browser.
    assert.deepEqual(hdr.weights, [16, 17, 0, 0, 27, 11, 11], 'HDR carries HLG decodes and uploads');
    assert.equal(hdr.source_count, 82);
    await page.locator('#mode').selectOption('8:420');
    assert.deepEqual((await apply()).weights, [33, 0, 0, 0, 27, 22, 0], 'SDR round-trip keeps the total and engines');
    await reset({...sdr, source_count: 8, weights: [0, 0, 0, 0, 8, 0, 0]});
    await page.locator('#mode').selectOption('10:420');
    assert.deepEqual((await apply()).weights, [0, 0, 0, 0, 8, 0, 0], 'browser-only setup must not gain decoders');

    // Typing a new total redistributes as the Balanced mix, not the old proportions.
    await reset({...sdr, bit_depth: 8, chroma: '420', source_count: 5, weights: [4, 0, 0, 0, 1, 0, 0]});
    await page.locator('#sources').fill('106');
    assert.deepEqual((await apply()).weights, [36, 0, 0, 0, 36, 34, 0], 'a new total takes the Balanced mix');

    // Balanced is the measured 30 fps baseline on SDR; HDR scales it to the 10-bit capacity.
    await reset({...sdr, bit_depth: 8, chroma: '420', dsk: ['lower_third', 'ticker', 'bug_left', 'bug_right']});
    await page.locator('[data-preset=balanced]').click();
    await page.locator('#sources').fill('106');
    assert.deepEqual((await apply()).weights, [36, 0, 0, 0, 36, 34, 0], 'Balanced on SDR is the 30 fps baseline');
    await page.locator('#mode').selectOption('10:420');
    assert.deepEqual((await apply()).weights, [16, 16, 0, 0, 31, 12, 11], 'HDR 4:2:0 scales the baseline to its capacity, half HLG');
    await page.locator('#mode').selectOption('10:422');
    assert.deepEqual((await apply()).weights, [14, 14, 3, 0, 27, 10, 9], 'HDR 4:2:2 scales to its capacity and adds HLG v210 sources');
    await page.locator('#mode').selectOption('8:420');
    const back = await apply();
    assert.deepEqual(back.weights, [36, 0, 0, 0, 36, 34, 0], 'back on SDR the baseline returns');
    assert.equal(back.source_count, 106);

    // Leaving HDR 4:2:2 keeps each source on its engine where the mode allows, trims
    // what exceeds a budget and refills the headroom: 100 mixed sources with four keys,
    // scaled down to the 4:2:2 capacity on load, come back as 40 NVDEC + 30 browser + 30 NV12.
    await reset({...sdr, fps: 25, source_count: 100, bit_depth: 10, chroma: '422',
      weights: [20, 19, 4, 19, 19, 10, 9], dsk: ['lower_third', 'ticker', 'bug_left', 'bug_right']});
    await page.locator('#mode').selectOption('8:420');
    const light = await apply();
    assert.deepEqual(light.weights, [40, 0, 0, 0, 30, 30, 0], 'SDR keeps a full, in-budget mix');
    assert.equal(light.source_count, 100, 'the total scaled down for 4:2:2 comes back');
    await reset({...sdr, fps: 25, source_count: 30, bit_depth: 10, chroma: '422', weights: [10, 5, 4, 3, 5, 2, 1]});
    await page.locator('#mode').selectOption('10:420');
    assert.deepEqual((await apply()).weights, [10, 5, 0, 0, 5, 5, 5], '4:2:2 uploads become 4:2:0 uploads of the same colour');

    // The default canvas is 10-bit 4:2:2: 0.74 of 110 at 25/30 fps, of 90 at 50 (scaled) and of 75 at 60.
    for (const [fps, maximum] of [[25, 81], [30, 81], [50, 66], [60, 55]]) {
      await reset();
      await page.locator('#fps').selectOption(String(fps));
      await page.locator('[data-preset=equal]').click();
      await page.locator('#sources').fill(String(maximum));
      assert.equal(await page.locator('#error').textContent(), '');
      const setup = await apply();
      assert.equal(setup.source_count, maximum);
      assert.equal(setup.weights.reduce((a,b)=>a+b,0), maximum);
      assert(setup.weights[0]+setup.weights[1] <= Math.min(40, Math.floor(1100 / fps)));
      await page.locator('#sources').fill(String(maximum + 1));
      assert.equal((await apply()).source_count, maximum, `${fps} fps stops at ${maximum}`);
      await page.locator('#fps').selectOption('60');
      assert.equal((await apply()).source_count, 55);
    }

    await reset();
    await page.locator('#fps').selectOption('30');
    await page.locator('[data-preset=equal]').click();
    await page.locator('#sources').fill('81');   // the 4:2:2 maximum at 30 fps
    const mixed = await apply();
    assert.equal(mixed.weights.reduce((sum, n) => sum + n, 0), 81);
    assert(mixed.weights[2] <= 4 && mixed.weights[4] <= 40);
    await page.locator('#scenes').fill('193');
    assert.equal((await apply()).scene_count, 192);
    for (const [fps, units] of [[25, 30], [30, 34], [50, 20], [60, 17]]) {
      await reset();
      await page.locator('#fps').selectOption(String(fps));
      await page.locator('#count-sdr420_raw').fill('40');
      const raw = (await apply()).weights;
      assert.equal(raw[5] + 2 * raw[6], units, `NV12 fills what P010 leaves of the ${fps} fps upload budget`);
    }
    await reset();
    await page.locator('#fps').selectOption('25');
    await page.locator('#count-browser').fill('41');
    assert.equal((await apply()).weights[4], 40);
    for (const mode of ['8:420', '10:420', '10:422']) {
      await reset();
      await page.locator('#mode').selectOption(mode);
      assert.equal(await page.locator('#count-sdr420_raw').isEnabled(), true);
      await page.locator('#count-sdr420_raw').fill('3');
      const raw = await apply();
      assert.equal(raw.weights[5], 3);
      assert.equal(raw.weights.reduce((sum, n) => sum + n, 0), raw.source_count);
      assert.match(await page.locator('#engine-split').textContent(), /3 raw NV12 upload/);
    }
    for (const mode of ['10:420', '10:422']) {
      await reset();
      await page.locator('#mode').selectOption(mode);
      const raw = page.locator('#count-hlg420_raw');
      assert.equal(await raw.isEnabled(), true);
      await raw.fill('3');
      const hdr = await apply();
      assert.equal(hdr.weights[6], 3);
      assert.match(await page.locator('#engine-split').textContent(), /3 raw P010 upload/);
      await page.locator('#mode').selectOption('8:420');
      assert.equal(await raw.isDisabled(), true);
      const sdr = await apply();
      assert.equal(sdr.weights[6], 0);
      assert.equal(sdr.weights[5], hdr.weights[5] + 3 + hdr.weights[2] + hdr.weights[3], 'P010 and 4:2:2 uploads stay uploads, as NV12');
    }
    await reset();
    // The browser ring size is not a setup control: the show takes the frame rate's default.
    assert.equal(await page.locator('#browser-ring-size').count(), 0);
    assert.equal('browser_ring_size' in await apply(), false);

    // Extra aux outputs fill NVENC to the profile's budget beside the program, its HLG copy on a
    // 10-bit canvas, the clean feed and the two own buses, at p3 (setup_runtime.extra_aux_limit).
    const extraAux = page.locator('#extra-aux');
    for (const [mode, limits] of [['8:420', [10, 8, 8, 6]], ['10:420', [8, 6, 4, 2]]]) {
      await reset();
      await page.locator('#mode').selectOption(mode);
      await page.locator('#dsk-pages input[value=lower_third]').check();
      await page.locator('#clean-feed').check();
      for (const [i, fps] of [25, 30, 50, 60].entries()) {
        await page.locator('#fps').selectOption(String(fps));
        assert.equal(await extraAux.getAttribute('max'), String(limits[i]), `${mode} at ${fps} fps`);
        if (!limits[i]) {
          assert.equal(await extraAux.isDisabled(), true, 'no room, no input');
          continue;
        }
        await extraAux.fill('99');
        const setup = await apply();
        assert.equal(setup.extra_aux, limits[i], `${mode} at ${fps} fps stops at ${limits[i]}`);
        assert.equal(setup.clean_feed, true);
      }
    }
    // A change that lowers the limit lowers the count; one that raises it leaves the count.
    await reset();
    await page.locator('#mode').selectOption('8:420');
    await page.locator('#dsk-pages input[value=lower_third]').check();
    await page.locator('#clean-feed').check();
    await page.locator('#fps').selectOption('25');
    await extraAux.fill('9');
    await page.locator('#fps').selectOption('30');
    assert.equal(await extraAux.inputValue(), '8');
    await page.locator('#mode').selectOption('10:420');
    assert.equal(await extraAux.inputValue(), '6');
    // The clean feed counts even while off: the maximum does not move with it.
    await page.locator('#clean-feed').uncheck();
    assert.equal(await extraAux.getAttribute('max'), '6');
    assert.equal((await apply()).extra_aux, 6);
    assert.match(await page.locator('#extra-aux-note').textContent(),
      /^Extra aux outputs: at most 6, each with random layouts of 4–16 sources, at the preset and bitrate under Encodes\.$/);
    assert.equal(await page.locator('#encodes ~ tfoot').innerText().then(t => t.split(/\s+/).join(' ')),
      'Total of 80% · at most 6 extra aux 78.8%', 'six extra outputs beside 39.4% of encodes');
    // Each encode costs its preset: the HLG program at p5 leaves 3 at 30 fps and exceeds the budget
    // at 60; the monitors at p1 leave 9 and 3.
    await preset('Program · HLG').selectOption('p5');
    assert.equal(await extraAux.getAttribute('max'), '3');
    await page.locator('#fps').selectOption('60');
    assert.equal(await page.locator('#error').textContent(),
      'The encodes need 95.3% of NVENC, above its 80% budget: choose faster presets.');
    assert.equal(await page.locator('#apply').isDisabled(), true, 'encodes above the budget cannot be applied');
    assert.equal(await extraAux.isDisabled(), true);
    await preset('Program · HLG').selectOption('p3');
    for (const output of ['Program preview', 'Multiviewer', 'Extra aux · each']) await preset(output).selectOption('p1');
    assert.equal(await extraAux.getAttribute('max'), '3');
    await page.locator('#fps').selectOption('30');
    assert.equal(await extraAux.getAttribute('max'), '9');
    await extraAux.fill('9');
    const monitors = await apply();
    assert.deepEqual([monitors.extra_aux, monitors.encodes.mv.preset, monitors.encodes.extra.preset], [9, 'p1', 'p1']);
    const saved = submissions.at(-1);
    await reset(saved);
    assert.equal(await extraAux.inputValue(), '9', 'the saved count loads');
    const {extra_aux: _, ...older} = saved;
    await reset(older);
    assert.equal((await apply()).extra_aux, 0, 'settings saved before extra aux outputs load with none');
    auxStatus = {aux_buses: [], janus_api: false};
    await reset(saved);
    assert.equal(await extraAux.isDisabled(), true);
    assert.match(await page.locator('#extra-aux-note').textContent(), /--janus-api/);
    assert.deepEqual((await rows()).map(row => row.split(' | ')[0]), ['Program · SDR', 'Program · HLG', 'Clean feed · SDR'], 'no own buses, no extra row');
    assert.equal((await apply()).extra_aux, 0, 'without a Janus API there are none');
    assert.deepEqual(errors, []);
    console.log('PASS: normal source-count editing, per-output encodes, allocation, limits, raw SDR/HDR upload and extra aux outputs');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
