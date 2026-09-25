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
    page.on('pageerror', error => errors.push(error.message));
    const html = fs.readFileSync(path.join(__dirname, '../setup.html'), 'utf8');
    await page.route('http://mixer.test/**', route => {
      if (route.request().url().endsWith('/api/setup')) {
        if (route.request().method() === 'POST') submissions.push(route.request().postDataJSON());
        return route.fulfill({json: {phase: 'idle', message: 'Ready', settings: null}});
      }
      return route.fulfill({contentType: 'text/html', body: html});
    });
    const reset = async () => {
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

    for (const fps of [25, 30]) {
      await reset();
      await page.locator('#fps').selectOption(String(fps));
      await page.locator('#mode').selectOption('8:420');
      await page.locator('#count-browser').fill('0');
      await page.locator('#sources').fill('96');
      assert.equal(await page.locator('#error').textContent(), '');
      const setup = await apply();
      assert.equal(setup.source_count, 96);
      assert.deepEqual(setup.weights, [96, 0, 0, 0, 0]);
      await page.locator('#fps').selectOption('60');
      assert.equal((await apply()).source_count, 48);
    }

    await reset();
    await page.locator('#fps').selectOption('30');
    await page.locator('#sources').fill('96');
    const mixed = await apply();
    assert.equal(mixed.weights.reduce((sum, n) => sum + n, 0), 96);
    assert(mixed.weights[2] <= 4 && mixed.weights[4] <= 32);
    await page.locator('#scenes').fill('129');
    assert.equal((await apply()).scene_count, 128);
    assert.deepEqual(errors, []);
    console.log('PASS: setup bitrate, 96-source allocation, FPS limits and source caps');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
