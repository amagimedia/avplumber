// Mocked HTTP only: failed iframe navigation must recover without mixer commands.
const assert = require('node:assert/strict');
const {launch, openPage, bigShow} = require('./page_fixture.cjs');

(async () => {
  const browser = await launch();
  try {
    const show = bigShow();
    show.state.settings.program_outputs[0].audio = true;
    const {page, errors, commands} = await openPage(browser, show);
    await page.waitForFunction(() => viewers.length === 2 && viewers.every(v => v.stats));
    // Keep the healthy program unmuted: neither recovery nor a source refresh
    // should reload it or discard the operator's audio choice.
    await page.locator('#wall > .viewer').first().locator('header .mute').click();
    const [program, detection] = await page.evaluate(() => viewers.map(v => v.src));
    let programLoads = 0, detectionLoads = 0;
    page.on('request', request => {
      if (request.url() === program) programLoads++;
      if (request.url() === detection) detectionLoads++;
    });
    await page.route(detection, route => route.fulfill({status: 503, contentType: 'text/html', body: '<html><body>Unavailable</body></html>'}));
    await page.evaluate(() => loadViewer(viewers[1]));
    await page.waitForFunction(() => viewers[1].frame.contentDocument?.body?.textContent === 'Unavailable');
    assert.equal(detectionLoads, 1);
    await page.evaluate(() => { recoverViewer(viewers[1]); });
    assert.equal(detectionLoads, 1, 'a load event alone does not trigger rapid retries');
    await page.unroute(detection);
    await page.evaluate(() => {
      const v = viewers[1];
      v.playerWaitAt = performance.now() - PLAYER_TIMEOUT_MS - 1;
      // A hidden document must not reload frames or retain an expired deadline.
      Object.defineProperty(document, 'hidden', {configurable: true, value: true});
      recoverViewer(v);
      if (performance.now() - v.playerWaitAt > 100) throw Error('hidden grace not reset');
      Object.defineProperty(document, 'hidden', {configurable: true, value: false});
      v.playerWaitAt = performance.now() - PLAYER_TIMEOUT_MS - 1;
      recoverViewer(v);
    });
    await page.waitForFunction(() => viewers[1].stats);
    assert.equal(detectionLoads, 2, 'the failed iframe reloads and resumes its heartbeat');
    assert.equal(programLoads, 0, 'healthy program is not reloaded');
    assert.equal(await page.evaluate(() => viewers[0].muted), false);
    assert.deepEqual(commands, [], 'recovery sends no mixer commands');
    assert.deepEqual(errors, []);

    // Recovery of the audio viewer re-sends its saved mute preference after load.
    await page.evaluate(() => loadViewer(viewers[0]));
    await page.waitForFunction(() => viewers[0].stats);
    assert.equal(await page.evaluate(() => viewers[0].muted), false);
    assert.equal(programLoads, 1);
    assert.deepEqual(commands, []);
    console.log('Failed iframe recovery, heartbeat, hidden grace, isolated reload and audio preference passed');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
