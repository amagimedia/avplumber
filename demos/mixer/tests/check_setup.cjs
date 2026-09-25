// Run with Playwright installed (or NODE_PATH pointing to its node_modules).
// All HTTP requests are intercepted; this never applies settings to a live mixer.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({args: ['--no-sandbox']});
  try {
    const page = await browser.newPage();
    const errors = [], submissions = [];
    let initialSettings = null;
    page.on('pageerror', error => errors.push(error.message));
    const html = fs.readFileSync(path.join(__dirname, '../setup.html'), 'utf8');
    await page.route('http://mixer.test/**', route => {
      if (route.request().url().endsWith('/api/setup')) {
        if (route.request().method() === 'POST') submissions.push(route.request().postDataJSON());
        return route.fulfill({json: {phase: 'idle', message: 'Ready', settings: initialSettings}});
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
    await page.locator('#bitrate').fill('8500');
    assert.equal((await apply()).bitrate_kbps, 8500, 'bitrate edit alone must reach the backend');

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
    assert.deepEqual(hdr.weights, [16, 17, 0, 0, 26, 23, 0], 'HDR must preserve decoder, browser and upload counts');
    assert.equal(hdr.source_count, 82);
    await page.locator('#mode').selectOption('8:420');
    assert.deepEqual((await apply()).weights, sdr.weights, 'SDR round-trip restores the source mix');
    await reset({...sdr, source_count: 8, weights: [0, 0, 0, 0, 8, 0, 0]});
    await page.locator('#mode').selectOption('10:420');
    assert.deepEqual((await apply()).weights, [0, 0, 0, 0, 8, 0, 0], 'browser-only setup must not gain decoders');

    for (const [fps, maximum] of [[25, 110], [30, 83], [50, 50], [60, 41]]) {
      await reset();
      await page.locator('#fps').selectOption(String(fps));
      await page.locator('[data-preset=equal]').click();
      await page.locator('#sources').fill(String(maximum));
      assert.equal(await page.locator('#error').textContent(), '');
      const setup = await apply();
      assert.equal(setup.source_count, maximum);
      assert.equal(setup.weights.reduce((a,b)=>a+b,0), maximum);
      assert(setup.weights[0]+setup.weights[1] <= (fps<=30?40:20));
      await page.locator('#sources').fill(String(maximum + 1));
      assert.equal((await apply()).source_count, maximum, `${fps} fps stops at ${maximum}`);
      await page.locator('#fps').selectOption('60');
      assert.equal((await apply()).source_count, 41);
    }

    await reset();
    await page.locator('#fps').selectOption('30');
    await page.locator('[data-preset=equal]').click();
    await page.locator('#sources').fill('83');
    const mixed = await apply();
    assert.equal(mixed.weights.reduce((sum, n) => sum + n, 0), 83);
    assert(mixed.weights[2] <= 4 && mixed.weights[4] <= 32);
    await page.locator('#scenes').fill('193');
    assert.equal((await apply()).scene_count, 192);
    for (const [fps, maximum] of [[25, 28], [30, 23], [50, 14], [60, 11]]) {
      await reset();
      await page.locator('#fps').selectOption(String(fps));
      await page.locator('#count-sdr420_raw').fill('29');
      assert.equal((await apply()).weights[5], maximum);
    }
    await reset();
    await page.locator('#fps').selectOption('25');
    await page.locator('#count-browser').fill('33');
    assert.equal((await apply()).weights[4], 32);
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
      assert.equal((await apply()).weights[6], 3);
      assert.match(await page.locator('#engine-split').textContent(), /3 raw P010 upload/);
      await page.locator('#mode').selectOption('8:420');
      assert.equal(await raw.isDisabled(), true);
      const sdr = await apply();
      assert.equal(sdr.weights[6], 0);
      assert.equal(sdr.weights[5], 3);
    }
    await reset();
    for (const [fps, ring] of [[25, 6], [50, 9], [30, 6], [60, 9]]) {
      await page.locator('#fps').selectOption(String(fps));
      assert.equal((await apply()).browser_ring_size, ring);
    }
    for (const ring of [6, 9, 11]) {
      await page.locator('#browser-ring-size').fill(String(ring));
      assert.equal((await apply()).browser_ring_size, ring);
    }
    await page.locator('#browser-ring-size').fill('0');
    assert.equal(await page.locator('#apply').isDisabled(), true);
    assert.deepEqual(errors, []);
    console.log('PASS: normal source-count editing, setup bitrate, allocation, limits and raw SDR/HDR upload');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
