// Playwright; all requests are intercepted, so this never controls a live mixer. The viewers' program audio mute:
// offered only on the output whose settings say it carries the show's audio, muted at first, toggled by its button.
const assert = require('node:assert/strict');
const {launch, openPage, bigShow} = require('./page_fixture.cjs');

(async () => {
  const browser = await launch();
  try {
    for (const audio of [false, true]) {
      const show = bigShow();
      if (audio) show.state.settings.program_outputs[0].audio = true;
      const {page, errors} = await openPage(browser, {...show, viewport: {width: 1512, height: 860}});
      await page.waitForFunction(() => state && $('scenes').children.length);
      const mute = (n) => page.locator('#wall > .viewer').nth(n).locator('header .mute');
      assert.equal(await mute(0).isVisible(), audio, `the program viewer offers the mute only with audio (${audio})`);
      assert(!(await mute(1).isVisible()), 'a viewer of another output never does');
      if (audio) {
        assert.equal(await mute(0).getAttribute('aria-pressed'), 'true', 'program audio starts muted');
        await mute(0).click();
        assert.equal(await mute(0).getAttribute('aria-pressed'), 'false');
        assert.equal(await page.evaluate(() => viewers[0].muted), false);
      }
      assert.deepEqual(errors, []);
      await page.close();
    }
    console.log('Program audio mute offered with audio only, muted at first, toggled passed');
  } finally {
    await browser.close();
  }
})().catch((error) => { console.error(error); process.exit(1); });
