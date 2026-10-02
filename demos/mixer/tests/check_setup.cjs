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
    const ownAux = [{id: 'mv', label: 'Program preview', full_rate: false}, {id: 'mv2', label: 'Multiviewer', full_rate: false}];
    let auxStatus = {aux_buses: ownAux, janus_api: true};
    page.on('pageerror', error => errors.push(error.message));
    const html = fs.readFileSync(path.join(__dirname, '../setup.html'), 'utf8');
    // The status carries the instance's profile as webui.py serves it: the limits asserted below are tesla_t4's.
    const profiles = JSON.parse(execFileSync(process.env.PYTHON || 'python3', ['-c',
      'import json; from instance_profiles import INSTANCE_PROFILES as p; print(json.dumps({t.value: v for t, v in p.items()}))'],
      {cwd: path.join(__dirname, '..')}));
    let instanceType = 'tesla_t4';
    await page.route('http://mixer.test/**', route => {
      if (route.request().url().endsWith('/api/setup')) {
        if (route.request().method() === 'POST') submissions.push(route.request().postDataJSON());
        return route.fulfill({json: {phase: 'idle', message: 'Ready', settings: initialSettings,
          instance_type: instanceType, profile: profiles[instanceType], ...auxStatus}});
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
    const text = id => page.locator(`#${id}`).textContent();
    const sources = page.locator('#sources'), extraAux = page.locator('#extra-aux');
    await reset();
    // Unique sources start at the maximum of the default canvas, 10-bit 4:2:2 at 60 fps.
    assert.match(await text('source-limit'), /^Maximum 55 .* measured on tesla_t4\.$/);
    assert.equal(await sources.inputValue(), '55');

    // Every encoded output has its row at the profile's default (nvenc.defaults); the clean feed's
    // only while it is on, though it counts even while off.
    const rows = () => page.locator('#encodes > .output').evaluateAll(rows => rows.map(row =>
      [...row.querySelectorAll('b,small')].map(cell => cell.textContent).join(' | ')));
    assert.deepEqual(await rows(), ['Program | H.264 · 60 fps · 13% NVENC', 'Program HLG | HEVC 10-bit · 60 fps · 18% NVENC',
      'Program preview | H.264 · 30 fps · 6.1% NVENC', 'Multiviewer | H.264 · 30 fps · 6.1% NVENC',
      'Extra aux | H.264 · 30 fps · 6.1% NVENC each · max 3']);
    assert.equal(await text('nvenc-note'), 'NVENC 57% of 80% (clean feed counted while off) · 3 more extra aux fit');
    assert.match(await text('engine-split'), /\. NVENC 57% of 80%\.$/, 'the summary shows the NVENC total');
    const preset = output => page.getByLabel(`${output} preset`, {exact: true});
    const bitrate = output => page.getByLabel(`${output} bitrate, Mbit/s`, {exact: true});
    const defaultsButton = page.locator('#output-defaults');
    assert.equal(await defaultsButton.isDisabled(), true, 'nothing to reset');
    assert.equal(await bitrate('Program HLG').inputValue(), '8');
    await preset('Program').selectOption('p1');
    await bitrate('Program').fill('8.5');
    const encode = (preset, mbps) => ({preset, bitrate_kbps: mbps * 1000});
    const defaults = {sdr: encode('p3', 6), hdr: encode('p3', 8), sdr_clean: encode('p3', 6), mv: encode('p1', 4),
      mv2: encode('p1', 4), extra: encode('p1', 4)};
    const edited = await apply();
    assert.deepEqual(edited.encodes, {...defaults, sdr: encode('p1', 8.5)}, 'an encode edit alone must reach the backend');
    assert.equal('bitrate_kbps' in edited, false);
    for (const value of ['25', '1.5', '']) {
      await bitrate('Multiviewer').fill(value);
      assert.equal(await text('error'), 'Multiviewer: the bitrate must be from 2 to 20 Mbit/s.');
      assert.equal(await page.locator('#apply').isDisabled(), true, `${value || 'no'} Mbit/s cannot be applied`);
    }
    await bitrate('Multiviewer').fill('20');
    assert.equal((await apply()).encodes.mv2.bitrate_kbps, 20000);
    await defaultsButton.click();
    assert.deepEqual((await apply()).encodes, defaults, 'Defaults resets every preset and bitrate');
    assert.equal(await bitrate('Program').inputValue(), '6');
    assert.equal(await defaultsButton.isDisabled(), true);
    await reset(edited);
    assert.equal(await preset('Program').inputValue(), 'p1', 'saved encodes load');
    assert.equal(await bitrate('Program').inputValue(), '8.5');
    const {encodes: _encodes, ...withoutEncodes} = edited;
    await reset(withoutEncodes);
    assert.deepEqual((await apply()).encodes, defaults, 'settings saved before per-output encodes load the defaults');

    // The total follows the maximum unless a lower one is typed; that one stays, within the maximum.
    await reset();
    await page.locator('#fps').selectOption('30');
    assert.equal(await sources.inputValue(), '81');
    await page.locator('#mode').selectOption('8:420');
    assert.equal(await sources.inputValue(), '110');
    await page.locator('#dsk-pages input[value=lower_third]').check();
    assert.equal(await sources.inputValue(), '109', 'a key page takes a source');
    await sources.fill('70');
    await page.locator('#fps').selectOption('60');
    assert.equal((await apply()).source_count, 70, 'a lower total stays');
    await page.locator('#mode').selectOption('10:422');
    assert.equal((await apply()).source_count, 54, 'capped at the maximum');
    await page.locator('#fps').selectOption('25');
    assert.equal(await sources.inputValue(), '70', 'and back below it');
    await sources.fill('120');
    assert.equal(await sources.inputValue(), '80', 'above the maximum is the maximum');
    await page.locator('#fps').selectOption('60');
    assert.equal(await sources.inputValue(), '54', 'and follows it again');
    await reset({...edited, fps: 30, bit_depth: 8, chroma: '420', source_count: 40, dsk: []});
    assert.equal(await sources.inputValue(), '40', 'a saved lower total loads');
    await reset({...edited, fps: 30, bit_depth: 8, chroma: '420', source_count: 110, dsk: []});
    await page.locator('#fps').selectOption('60');
    assert.equal(await sources.inputValue(), '75', 'a saved maximum follows the maximum');
    await sources.fill('');
    assert.equal(await sources.inputValue(), '', 'clearing leaves an editable blank');
    assert.equal(await page.locator('#apply').isDisabled(), true, 'no total cannot be applied');
    await sources.press('8');
    const eight = await apply();
    assert.deepEqual([eight.source_count, eight.weights.reduce((a, b) => a + b, 0)], [8, 8], 'typing a replacement restores a valid setup');

    // The source mix is always Balanced: read-only counts, the docs' table with four key pages
    // (docs/cookbook/source-limits.html: SDR NVDEC, HLG NVDEC, HLG v210, browser, NV12, P010).
    assert.equal(await page.locator('#source-controls input, [data-preset]').count(), 0);
    const balanced = {'8:420': [[40, 0, 0, 36, 30, 0], [36, 0, 0, 36, 34, 0], [22, 0, 0, 36, 20, 0], [18, 0, 0, 36, 17, 0]],
      '10:420': [[16, 19, 0, 31, 10, 10], [15, 18, 0, 30, 12, 11], [10, 12, 0, 34, 7, 6], [8, 10, 0, 28, 6, 5]],
      '10:422': [[14, 13, 3, 27, 10, 10], [13, 13, 3, 25, 12, 11], [11, 11, 3, 24, 7, 6], [9, 9, 2, 20, 6, 5]]};
    await reset({...eight, source_count: 110, dsk: ['lower_third', 'ticker', 'bug_left', 'bug_right']});
    for (const [mode, mixes] of Object.entries(balanced)) {
      await page.locator('#mode').selectOption(mode);
      for (const [i, fps] of [25, 30, 50, 60].entries()) {
        await page.locator('#fps').selectOption(String(fps));
        const {weights: [sdr, hlg, v210, sdr422, browser, nv12, p010], source_count} = await apply();
        assert.deepEqual([sdr, hlg, v210, browser, nv12, p010], mixes[i], `${mode} at ${fps} fps`);
        assert.equal(sdr422 + sdr + hlg + v210 + browser + nv12 + p010, source_count);
        assert.equal(await page.locator('#count-browser').textContent(), String(browser));
      }
    }
    assert.equal(await page.locator('#count-sdr422').isHidden(), true, 'a kind the mix leaves out has no row');
    await page.locator('#scenes').fill('193');
    assert.equal((await apply()).scene_count, 192);
    // The browser ring size is not a setup control: the show takes the frame rate's default.
    assert.equal(await page.locator('#browser-ring-size').count(), 0);
    assert.equal('browser_ring_size' in await apply(), false);

    // Extra aux outputs fill NVENC to the profile's budget beside the program, its HLG copy on a
    // 10-bit canvas, the clean feed and the two own buses, at the defaults (setup_runtime.extra_aux_limit).
    for (const [mode, limits] of [['8:420', [11, 8, 9, 6]], ['10:420', [10, 7, 6, 3]]]) {
      await reset();
      await page.locator('#mode').selectOption(mode);
      await page.locator('#dsk-pages input[value=lower_third]').check();
      await page.locator('#clean-feed').check();
      for (const [i, fps] of [25, 30, 50, 60].entries()) {
        await page.locator('#fps').selectOption(String(fps));
        assert.equal(await extraAux.getAttribute('max'), String(limits[i]), `${mode} at ${fps} fps`);
        await extraAux.fill('99');
        const setup = await apply();
        assert.equal(setup.extra_aux, limits[i], `${mode} at ${fps} fps stops at ${limits[i]}`);
        assert.equal(setup.clean_feed, true);
      }
    }
    assert.deepEqual((await rows()).map(row => row.split(' | ')[0]),
      ['Program', 'Program HLG', 'Clean feed', 'Program preview', 'Multiviewer', 'Extra aux'], 'the clean feed on has its row');
    // A change that lowers the limit lowers the count; one that raises it leaves the count.
    await reset();
    await page.locator('#mode').selectOption('8:420');
    await page.locator('#dsk-pages input[value=lower_third]').check();
    await page.locator('#clean-feed').check();
    await page.locator('#fps').selectOption('25');
    await extraAux.fill('10');
    await page.locator('#fps').selectOption('30');
    assert.equal(await extraAux.inputValue(), '8');
    await page.locator('#mode').selectOption('10:420');
    assert.equal(await extraAux.inputValue(), '7');
    await page.locator('#mode').selectOption('8:420');
    assert.equal(await extraAux.inputValue(), '7');
    await page.locator('#mode').selectOption('10:420');
    // The clean feed counts even while off: the maximum does not move with it.
    await page.locator('#clean-feed').uncheck();
    assert.equal(await extraAux.getAttribute('max'), '7');
    assert.equal((await apply()).extra_aux, 7);
    assert.equal((await rows()).some(row => row.startsWith('Clean feed')), false);
    assert.equal(await text('nvenc-note'), 'NVENC 77% of 80% (clean feed counted while off) · no more extra aux fit');
    await extraAux.fill('2');
    // Over the budget the page names the output to lower and refuses as the server does: the
    // programs and the clean feed at p5 at 60 fps (as tests/test_setup_runtime.py's OVER_BUDGET).
    await page.locator('#clean-feed').check();
    await page.locator('#fps').selectOption('60');
    for (const output of ['Program', 'Clean feed', 'Program HLG']) await preset(output).selectOption('p5');
    assert.equal(await text('nvenc-note'), 'Over budget: NVENC 95% of 80%. Lower Program to p3.');
    assert.equal(await text('error'), 'The encodes need 94.7% of NVENC, above its 80% budget: choose faster presets.');
    assert.equal(await page.locator('#apply').isDisabled(), true, 'encodes above the budget cannot be applied');
    assert.equal(await page.locator('#nvenc-bar.over').count(), 1);
    assert.deepEqual([await extraAux.inputValue(), await extraAux.isDisabled()], ['0', true], 'no room, no extra aux');
    await preset('Program').selectOption('p3');
    assert.equal(await text('error'), '');
    assert.equal(await extraAux.getAttribute('max'), '0');
    await preset('Program HLG').selectOption('p1');
    assert.equal(await extraAux.getAttribute('max'), '2');
    await extraAux.fill('2');
    const fitted = await apply();
    assert.deepEqual([fitted.extra_aux, fitted.encodes.sdr_clean.preset, fitted.encodes.hdr.preset], [2, 'p5', 'p1']);
    await reset(fitted);
    assert.equal(await extraAux.inputValue(), '2', 'the saved count loads');
    const {extra_aux: _, ...older} = fitted;
    await reset(older);
    assert.equal((await apply()).extra_aux, 0, 'settings saved before extra aux outputs load with none');
    auxStatus = {aux_buses: [], janus_api: false};
    await reset(fitted);
    assert.equal(await extraAux.count(), 0);
    assert.match(await text('nvenc-note'), / · extra aux need webui\.py --janus-api$/);
    assert.deepEqual((await rows()).map(row => row.split(' | ')[0]), ['Program', 'Program HLG', 'Clean feed'], 'no own buses, no extra row');
    assert.equal((await apply()).extra_aux, 0, 'without a Janus API there are none');
    // The L4's 10-bit canvases have their own limits (mode_limits): the largest show is the measured
    // mix, NVDEC and the browser windows full, v210 the only upload on 4:2:2 and none on 4:2:0.
    instanceType = 'nvidia_l4';
    await reset();
    assert.match(await text('source-limit'), /^Maximum 88 .* measured on nvidia_l4\.$/);
    assert.deepEqual((await apply()).weights, [22, 22, 4, 0, 40, 0, 0]);
    await page.locator('#mode').selectOption('10:420');
    assert.match(await text('source-limit'), /^Maximum 88 /);
    assert.deepEqual((await apply()).weights, [22, 26, 0, 0, 40, 0, 0]);   // the v210 share folds into the HLG decodes
    await page.locator('#fps').selectOption('30');
    await page.locator('#mode').selectOption('8:420');
    assert.match(await text('source-limit'), /^Maximum 170 /);
    assert.deepEqual((await apply()).weights, [83, 0, 0, 0, 40, 47, 0]);
    assert.deepEqual(errors, []);
    console.log('PASS: Balanced source counts, the maximum total, per-output encodes, NVENC budget and extra aux outputs');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
