// Playwright; all requests are intercepted, so this never controls a live mixer. The deck and the outputs column on a
// show like the demo's (page_fixture.cjs bigShow): shortcut hints, scene pages, the transition panel, the dividers,
// the output and layout menus, stream stats, and a page that fits the window.
const assert = require('node:assert/strict');
const {launch, openPage, bigShow} = require('./page_fixture.cjs');

(async () => {
  const browser = await launch();
  try {
    const show = bigShow();
    let {page, errors, commands, sent} = await openPage(browser, {...show, viewport: {width: 1512, height: 860}});
    const settle = () => page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
    const reload = async () => { await page.reload(); await page.waitForFunction(() => state && $('scenes').children.length); await settle(); };
    await page.waitForFunction(() => state && $('scenes').children.length);
    await settle();
    const text = (selector) => page.locator(selector).textContent();
    const texts = (selector) => page.locator(selector).allTextContents();
    const pressed = (selector) => page.locator(selector).evaluateAll((all) => all.map((b) => b.getAttribute('aria-pressed')));
    const viewer = (n) => page.locator('#wall > .viewer').nth(n);

    // A faint shortcut where it acts.
    const hints = () => page.evaluate(() => [...document.querySelectorAll('kbd')]
      .filter((k) => !k.closest('.tip') && k.checkVisibility()).map((k) => k.textContent));
    const shown = await hints();
    for (const k of ['← →', '⇧D', 'D', 'K', '⇧1–8', 'C', 'Space', 'F', 'W', '1–9', '< >'])
      assert(shown.includes(k), `a ${k} hint shows where it acts: ${JSON.stringify(shown)}`);
    assert.equal(await page.locator('#hint').evaluate((h) => h.firstChild.textContent), '⌨', 'the ⌨ button in the strip holds the shortcuts');
    assert.equal(await text('#hint-tip h4'), 'Keyboard shortcuts');
    assert.equal(await page.locator('#top a[href="/wall"]').textContent(), 'Outputs ↗');
    assert(await page.locator('#setup-link').isVisible(), 'Setup shows when the mixer allows it');
    assert.equal(await page.locator('#graph-link:visible').getAttribute('href'), '/graph/', 'Graph shows where the server says it is');

    // Program and Preview: FTB beside Program, Direct beside Preview, Next under it.
    assert.equal(await text('#pgm'), 'grid_100');
    assert.equal(await text('#pvw'), 'alpha_overlay_230');
    assert.equal(await text('#pgm-state'), 'on air');
    assert.equal(await text('#pvw-state'), 'Auto · fade 15f');
    assert(await page.locator('.bus.pgm #ftb').isVisible() && await page.locator('.bus.pvw #direct').isVisible());
    assert.deepEqual(await texts('#next-keys button'), ['BKGD', '1', '2', '3', '4']);
    assert.deepEqual(await pressed('#next-keys button'), ['true', 'true', 'false', 'false', 'false'], 'BKGD and the armed key');

    // Keys: their own rates bright, the ones following All faint at the All rate; the armed key marked.
    assert.deepEqual(await page.locator('#keys .rt').evaluateAll((all) => all.map((t) => [t.textContent, t.classList.contains('own')])),
      [['15f', true], ['12f', false], ['CUT', true], ['12f', false]]);
    assert.deepEqual(await texts('#keys .key .take'), ['1Lower third', '2Ticker', '3Bug left', '4Bug right']);
    assert.equal(await text('#all-rate'), '12f');
    assert(await page.locator('#keys .key:nth-child(1)').evaluate((k) => k.classList.contains('armed')));
    // Slots: the Program preview's eight, each with its ×.
    assert.equal(await text('#slots-title'), 'Slots');
    assert.equal(await text('#slot-note'), 'Program preview · click a slot, then a scene');
    assert.equal(await page.locator('#slots .slotc .slot').count(), 8);
    assert.equal(await page.locator('#slots .slotc .clear').count(), 8);
    assert.equal(await text('#slots .slotc:nth-child(3) .slot'), 'M3grid_060');

    // Stream stats under each screen, from its player, with what each figure means in its tip.
    const stats = (n) => viewer(n).locator('.vstats > span > b').allTextContents();
    await page.waitForFunction(() => document.querySelector('#wall > .viewer .vstats b').textContent === '38 ms');
    await page.waitForFunction(() => document.querySelectorAll('#wall > .viewer')[1].querySelector('.vstats b').textContent === '41 ms');
    assert.deepEqual(await stats(0), ['38 ms', '30', '12 ms']);
    assert.deepEqual(await stats(1), ['41 ms', '30', '14 ms']);
    assert.equal(await viewer(0).locator('.vstats > span:first-child > b').getAttribute('class'), 'ok', 'RTT in green up to 200 ms');
    const meanings = [];
    for (const n of [0, 1, 2]) {
      const stat = viewer(0).locator(`.vstats > span:nth-child(${n + 1})`);
      await stat.hover();
      const tip = stat.locator('[role="tooltip"]');
      assert(await tip.isVisible(), 'a stat shows its tip');
      meanings.push([await tip.locator('h4').textContent(), await tip.locator('p').count()]);
    }
    assert.deepEqual(meanings, [['RTT', 1], ['FPS', 1], ['Buffer', 1]]);
    assert.match(await viewer(0).locator('.vstats > span:first-child .tip p').textContent(), /^WebRTC network round trip/);
    assert.equal(await viewer(0).evaluate((v) => [...v.querySelectorAll('[title]')].filter((e) => e.querySelector('.tip')).length), 0);

    // The output menu: the outputs by group, each over its details; Escape closes it.
    const menuOf = (n, name) => viewer(n).locator(`.picker[data-menu="${name}"]`);
    await menuOf(0, 'outputs').click();
    const listbox = viewer(0).locator('.pick:has(.picker[data-menu="outputs"]) [role="listbox"]');
    assert(await listbox.isVisible());
    const options = await listbox.locator('li').evaluateAll((all) => all.map((li) => li.getAttribute('role') === 'option'
      ? [li.querySelector('b').textContent, li.querySelector('.meta').textContent, li.getAttribute('aria-selected')] : li.textContent));
    assert.deepEqual(options.slice(0, 7), ['Program', ['Program dirty', 'with keys · SDR · H.264 · 1920×1080p30', 'true'],
      ['Program clean', 'no keys · H.264 SDR · 30 fps', 'false'], 'Monitoring', ['Program preview', 'PGM · PVW · 8 scene slots · 30 fps', 'false'],
      ['Multiviewer', '192 sources · 12 pages · 30 fps', 'false'], ['Aux 0', '4 sources · 30 fps', 'false']]);
    assert.equal(options.length, 19, 'two groups, Program dirty, the clean feed and 15 buses');
    await page.keyboard.press('Escape');
    assert(await listbox.isHidden(), 'Escape closes the menu');
    // A bus with two or more layouts gets a layout menu; a pick sends it.
    assert(await menuOf(0, 'layouts').isHidden(), 'Program has no layouts');
    assert.equal(await menuOf(1, 'layouts').textContent(), 'PVW · PGM grid');
    await menuOf(1, 'layouts').click();
    const layouts = viewer(1).locator('.pick:has(.picker[data-menu="layouts"]) [role="option"]');
    assert.deepEqual(await layouts.evaluateAll((all) => all.map((li) => [li.textContent, li.getAttribute('aria-selected')])),
      [['PVW · PGM gridpreset', 'true'], ['Source pagespreset', 'false']]);
    await Promise.all([sent(), layouts.nth(1).click()]);
    assert.deepEqual(commands.splice(0), [{command: 'aux_layout', bus: 'pvw', layout: {preset: 'source_pages'}}]);
    // The Multiviewer's pages: its pager in the viewer, [ ] and the arrows.
    await menuOf(1, 'outputs').click();
    await viewer(1).locator('[role="option"]', {hasText: 'Multiviewer'}).click();
    const pager = viewer(1).locator('.pager');
    await pager.waitFor();
    assert.equal(await pager.locator('.mono').textContent(), '1 / 12');
    assert.equal(await pager.locator('.mono').getAttribute('title'), 'Sources 1–16 of 192');
    assert((await hints()).includes('[ ]'), 'the [ ] hint shows beside the pager');
    assert(await menuOf(1, 'layouts').isHidden(), 'one layout, no layout menu');
    await page.evaluate(() => restFocus());
    const ends = () => pager.locator('.pg').evaluateAll((all) => all.map((b) => b.disabled));
    assert.deepEqual(await ends(), [true, false], 'the first page has no previous page');
    await page.keyboard.press('[');
    await Promise.all([sent(), page.keyboard.press(']')]);
    await Promise.all([sent(), pager.locator('button[aria-label="Next multiview page"]').click()]);
    assert.deepEqual(commands.splice(0), [{command: 'aux_page', bus: 'mv', step: 1}, {command: 'aux_page', bus: 'mv', step: 1}],
      '[ on the first page sends nothing');
    const mv = show.state.aux_buses.find((a) => a.id === 'mv');
    mv.page = 11;
    await page.evaluate(() => refresh());
    await page.waitForFunction((count) => count.textContent === '12 / 12', await pager.locator('.mono').elementHandle());
    assert.deepEqual(await ends(), [false, true], 'the last page has no next page');
    await page.keyboard.press(']');
    await Promise.all([sent(), page.keyboard.press('[')]);
    assert.deepEqual(commands.splice(0), [{command: 'aux_page', bus: 'mv', step: -1}], '] on the last page sends nothing');
    mv.page = 0;
    await page.evaluate(() => refresh());
    await page.waitForFunction((count) => count.textContent === '1 / 12', await pager.locator('.mono').elementHandle());
    // A closed and reopened viewer gets stat tips of its own: no ids left behind.
    await viewer(1).locator('.close').click();
    assert.equal(await page.locator('#wall > .viewer').count(), 1);
    assert(await page.locator('#vsplit').isHidden(), 'one viewer, no divider');
    assert.equal(await viewer(0).evaluate((v) => v.style.flex), '', 'and no share');
    await page.locator('#viewer-count button[data-n="2"]').click();
    const ids = await page.locator('.vstats > span').evaluateAll((all) => all.map((s) => s.id));
    assert.equal(new Set(ids).size, 6);
    assert.equal(await page.locator('.vstats .tip').count(), 6);
    await menuOf(1, 'outputs').click();
    await viewer(1).locator('[role="option"]', {hasText: 'Program preview'}).click();

    // Scene pages: as many tiles as the grid holds, rows stretched to fill it, no scrolling.
    const scenesPage = () => page.evaluate(() => {
      const grid = $('scenes'), tiles = [...grid.children], box = grid.getBoundingClientRect(), pad = parseFloat(getComputedStyle(grid).paddingBottom);
      const header = [...document.querySelectorAll('#scenes-card > header > *')].filter((e) => e.checkVisibility()).map((e) => e.getBoundingClientRect());
      return {tiles: tiles.length, pageSize, count: $('count').textContent, pages: $('pages').textContent, buttons: $('pages').querySelectorAll('button').length,
              scroll: grid.scrollHeight - grid.clientHeight, gap: box.bottom - pad - (tiles.at(-1)?.getBoundingClientRect().bottom ?? 0),
              oneLine: Math.max(...header.map((r) => r.top)) - Math.min(...header.map((r) => r.top)) < 12,
              first: tiles[0]?.dataset.scene, numbers: tiles.map((t) => t.querySelector('.n').textContent).join('')};
    });
    let s = await scenesPage();
    const n = s.tiles, pages = Math.ceil(256 / n);
    assert(n > 9 && n === s.pageSize, `the page holds the grid's tiles: ${JSON.stringify(s)}`);
    assert.equal(s.count, `1–${n} of 256`);
    assert.equal(s.scroll, 0, 'the grid does not scroll');
    assert(s.gap >= 0 && s.gap < 12, `the rows fill the grid: ${s.gap}`);
    assert(s.oneLine, 'the scenes header keeps one line');
    assert.equal(s.numbers, '123456789', 'the first nine tiles carry their digits');
    assert.equal(s.first, 'random_001', 'scenes in order: by family, then by number');
    assert.deepEqual([s.buttons, s.pages], [0, `1 / ${pages}`], 'nine page buttons do not fit beside the filters here');
    assert(await page.locator('#pg-prev').isDisabled());
    await page.keyboard.press('>');
    s = await scenesPage();
    assert.equal(s.count, `${n + 1}–${2 * n} of 256`, '> turns the page');
    assert(!(await page.locator('#pg-prev').isDisabled()));
    // A digit picks on the page shown.
    await Promise.all([sent(), page.keyboard.press('1')]);
    assert.deepEqual(commands.splice(0), [{command: 'preview', scene: s.first}]);
    await page.keyboard.press('<');
    assert.equal((await scenesPage()).count, `1–${n} of 256`, '< turns it back');
    // → from the page's last tile turns the page with the Preview; ← from the first scene wraps to the last.
    await page.evaluate(() => { selected = visible.at(-1); });
    await Promise.all([sent(), page.keyboard.press('ArrowRight')]);
    s = await scenesPage();
    assert.equal(s.count, `${n + 1}–${2 * n} of 256`);
    assert.deepEqual(commands.splice(0), [{command: 'preview', scene: s.first}]);
    await page.evaluate(() => { selected = shownIds[0]; });
    await Promise.all([sent(), page.keyboard.press('ArrowLeft')]);
    assert.deepEqual(commands.splice(0), [{command: 'preview', scene: 'alpha_overlay_256'}]);
    assert.equal((await scenesPage()).count, `${(pages - 1) * n + 1}–256 of 256`);
    // A filter starts again at its first page; with few pages they get buttons.
    await page.locator('#chips button', {hasText: 'PiP'}).click();
    assert.deepEqual(await page.locator('#chips button').evaluateAll((all) => all.map((b) => [b.textContent, b.getAttribute('aria-pressed')])),
      [['All 256', 'false'], ['Random 48', 'false'], ['Grid 96', 'false'], ['Fullscreen 32', 'false'], ['PiP 48', 'true'], ['Browser 32', 'false']]);
    s = await scenesPage();
    assert.equal(s.count, `1–${Math.min(n, 48)} of 48`, 'a filter resets to its first page');
    assert.equal(s.buttons, Math.ceil(48 / n), 'two pages fit as buttons');
    assert.deepEqual(await pressed('#pages button'), ['true', 'false']);
    await page.locator('#pages button:nth-child(2)').click();
    assert.equal((await scenesPage()).count, `${n + 1}–48 of 48`);
    assert.equal(await page.evaluate(() => store.get('filter')), 'pip', 'the filter is kept');
    await page.locator('#chips button', {hasText: 'All'}).click();
    // At smaller sizes more pages than nine show as "p / N"; the page size follows the window and holds still.
    for (const width of [1332, 1512]) {
      for (let height = 700; height <= 1080; height += 20) {
        await page.setViewportSize({width, height});
        await settle();
        const a = await scenesPage();
        await settle(); await settle();
        const b = await scenesPage();
        assert.deepEqual(b, a, `the scenes hold still at ${width}x${height}`);
        assert(a.oneLine && a.scroll === 0 && a.tiles === a.pageSize, `${width}x${height}: ${JSON.stringify(a)}`);
        if (Math.ceil(256 / a.pageSize) > 9) assert.equal(a.pages, `1 / ${Math.ceil(256 / a.pageSize)}`);
      }
    }
    // The page is the window's height: the scenes and the outputs end level, nothing scrolls.
    for (const [width, height] of [[1332, 760], [1512, 860], [1920, 1050]]) {
      await page.setViewportSize({width, height});
      await settle();
      const fit = await page.evaluate(() => {
        const viewers = [...document.querySelectorAll('#wall > .viewer')].filter((v) => !v.hidden);
        return {scrollX: document.documentElement.scrollWidth - innerWidth, scrollY: document.documentElement.scrollHeight - innerHeight,
                level: Math.abs($('scenes-card').getBoundingClientRect().bottom - viewers.at(-1).getBoundingClientRect().bottom), viewers: viewers.length};
      });
      assert(fit.scrollX <= 0 && fit.scrollY <= 0 && fit.level <= 1 && fit.viewers === 2, `${width}x${height} fits: ${JSON.stringify(fit)}`);
    }
    // Stacked (1150 px and narrower) the page scrolls and the grid keeps eight rows.
    await page.setViewportSize({width: 1100, height: 800});
    await settle();
    const stacked = await page.evaluate(() => ({rows: pageSize / Number(getComputedStyle($('scenes')).getPropertyValue('--cols')),
                                                 scrolls: document.documentElement.scrollHeight > innerHeight}));
    assert.deepEqual(stacked, {rows: 8, scrolls: true});
    await page.setViewportSize({width: 1512, height: 860});
    await settle();

    // The transition panel. Fade and Dip: a swatch picks Dip's colour and selects it; Fade mixes again, the ring stays.
    await page.locator('#tp .dips [data-dip="#000000"]').click();
    assert.deepEqual([await page.locator('#fade').getAttribute('aria-pressed'), await page.locator('#dip').getAttribute('aria-pressed')], ['false', 'true']);
    assert.equal(await page.evaluate(() => [store.get('fadeColor'), store.get('fadeDip')].join()), 'black,black');
    await Promise.all([sent(), page.keyboard.press('Space')]);
    assert.equal(commands.splice(0)[0].color, '#000000', 'Auto dips through black');
    await page.locator('#fade').click();
    assert.deepEqual(await pressed('#tp .dips button'), ['true', 'false', 'false'], 'the dip colour keeps its ring');
    await Promise.all([sent(), page.keyboard.press('Space')]);
    assert(!('color' in commands.splice(0)[0]), 'Fade mixes');
    await page.locator('#tp .dips [data-dip="#ffffff"]').click();
    await page.keyboard.press('f');
    assert.equal(await page.locator('#fade').getAttribute('aria-pressed'), 'true', 'F selects Fade, a mix');
    await page.locator('#dip').click();
    await Promise.all([sent(), page.keyboard.press('Space')]);
    assert.equal(commands.splice(0)[0].color, '#ffffff', 'Dip goes through the colour picked last');
    await page.locator('#fade').click();
    // Curves: one pressed, the pick goes with the fade.
    assert.deepEqual(await pressed('#tp .seg.icons button'), ['true', 'false', 'false', 'false']);
    await page.locator('#tp .seg.icons button[data-value="ease-in"]').click();
    await Promise.all([sent(), page.keyboard.press('Space')]);
    assert.equal(commands.splice(0)[0].curve, 'ease-in');
    await page.locator('#tp .seg.icons button[data-value="linear"]').click();
    // Wipes: the group fills the row; the library's clips are buttons.
    const wipeList = () => page.evaluate(() => ({names: [...$('wipes').children].map((b) => b.textContent),
      fill: Math.abs([...$('wipes').children].reduce((w, b) => w + b.offsetWidth + 2, -2) - $('wipes').clientWidth) <= 1,
      scroll: $('wipes').scrollWidth - $('wipes').clientWidth, custom: !$('wipe-custom').hidden, pageScroll: document.documentElement.scrollWidth - innerWidth}));
    assert.deepEqual(await wipeList(), {names: ['Diagonal sweep', 'Sliding panels'], fill: true, scroll: 0, custom: false, pageScroll: 0});
    await page.keyboard.press('w');
    assert.deepEqual(await pressed('#wipes button'), ['true', 'false'], 'W selects the wipe');
    await page.keyboard.press('w');
    assert.deepEqual(await pressed('#wipes button'), ['false', 'true'], 'W again steps to the next wipe');
    await page.keyboard.press('w');
    assert.deepEqual(await pressed('#wipes button'), ['true', 'false'], 'W wraps round the list');
    await page.locator('#wipes button:nth-child(2)').click();
    await Promise.all([sent(), page.keyboard.press('Space')]);
    assert.deepEqual(commands.splice(0).map((c) => [c.command, c.wipe_file]), [['wipe', '/wipes/sliding_panels.mov']]);
    // Many clips scroll sideways inside the group; the page itself never scrolls.
    show.state.settings.wipes = Array.from({length: 12}, (_, i) => ({id: `w${i}`, name: `Wipe number ${i + 1}`, path: `/wipes/w${i}.mov`}));
    await page.evaluate(() => refresh());
    await page.waitForFunction(() => $('wipes').children.length === 12);
    let w = await wipeList();
    assert(w.scroll > 0 && w.pageScroll <= 0, `twelve clips scroll inside the group: ${JSON.stringify(w)}`);
    await page.locator('#wipes button:nth-child(12)').click();
    assert.equal(await page.locator('#wipes button:nth-child(12)').getAttribute('aria-pressed'), 'true');
    await page.evaluate(() => { $('wipes').scrollLeft = 0; });
    await page.keyboard.press('f');
    await page.keyboard.press('w');
    await page.waitForFunction(() => { const b = $('wipes').lastElementChild.getBoundingClientRect(), l = $('wipes').getBoundingClientRect();
      return b.right <= l.right + 1; });
    await page.keyboard.press('w');
    assert.equal(await page.locator('#wipes button:nth-child(1)').getAttribute('aria-pressed'), 'true', 'W after the last wipe: the first');
    await page.waitForFunction(() => $('wipes').scrollLeft < 1);
    // One clip shows its name; none gives a path field, which a poll keeps.
    show.state.settings.wipes = show.state.settings.wipes.slice(0, 1);
    await page.evaluate(() => refresh());
    await page.waitForFunction(() => $('wipes').children.length === 1);
    assert.deepEqual((await wipeList()).names, ['Wipe number 1']);
    show.state.settings.wipes = [];
    await page.evaluate(() => refresh());
    await page.waitForFunction(() => !$('wipe-custom').hidden);
    assert.deepEqual((await wipeList()).names, ['Media wipe']);
    await page.locator('#wipe-file').fill('/wipes/mine.mov');
    await Promise.all([page.waitForResponse((r) => r.url().endsWith('/api/state')), page.evaluate(() => refresh())]);
    await settle();
    assert.equal(await page.locator('#wipe-file').inputValue(), '/wipes/mine.mov');
    await page.locator('#wipe-file').press('Enter');
    await page.keyboard.press('w');
    await Promise.all([sent(), page.keyboard.press('Space')]);
    assert.deepEqual(commands.splice(0).map((c) => [c.command, c.wipe_file]), [['wipe', '/wipes/mine.mov']]);
    await page.keyboard.press('f');
    // The T-bar: LEDs and the trail fill with the mix's progress from the end the lever rests at, the grip with them.
    const tbar = () => page.evaluate(() => {
      const leds = [...document.querySelectorAll('#tp .leds i')], on = leds.flatMap((led, i) => (led.classList.contains('on') ? [i] : []));
      const trail = getComputedStyle(document.querySelector('#tp .trail'));
      return {leds: leds.length, on: on.length, from: on[0] ?? null, pct: document.querySelector('#tp .pct').textContent,
              live: document.querySelector('#tp .pct').classList.contains('live'), trail: [trail.getPropertyValue('--a'), trail.getPropertyValue('--b')].map(Number),
              grip: Number(document.querySelector('#tp .track').style.getPropertyValue('--v')), run: $('auto').classList.contains('run'), held: $('auto').classList.contains('held'), heldOn: $('tp').classList.contains('held-on')};
    });
    // A refresh while a poll is on its way only queues another, so wait for the page to show the change.
    const status = async (change) => {
      Object.assign(show.state.status, change);
      await page.evaluate(() => refresh());
      await page.waitForFunction((change) => Object.entries(change).every(([k, v]) => JSON.stringify(state.status[k]) === JSON.stringify(v)), change);
    };
    const rest = Number(await page.locator('#lever').inputValue()) / 1000;
    assert.deepEqual(await tbar(), {leds: 40, on: 0, from: null, pct: '0%', live: false, trail: [0, 0], grip: rest, run: false, held: false, heldOn: false});
    await status({transition: 'crossfade', transition_on_air: true, transition_progress: 0.4, transition_duration_ms: 1e9});
    await page.waitForFunction(() => document.querySelector('#tp .pct').textContent === '40%');   // a mix first seen midway eases in
    let t = await tbar();
    assert.deepEqual({...t, trail: t.trail.map((v) => Math.round(v * 100) / 100), grip: Math.round(t.grip * 100) / 100},
      {leds: 40, on: 16, from: 0, pct: '40%', live: true, trail: [0, 0.4], grip: 0.4, run: true, held: false, heldOn: false}, 'Auto running at 40%');
    await status({transition_paused: true});
    t = await tbar();
    assert.deepEqual([t.on, t.run, t.held, t.heldOn], [16, false, true, true], 'held: Auto and the LEDs in amber');
    // A mix from the lever's far end, where a completed take left it: the grip runs from there with them.
    await status({transition: 'idle', transition_on_air: false, transition_paused: false});
    await status({transition: 'crossfade', transition_on_air: true, lever: {origin: 1, position: 1}});
    await page.waitForFunction(() => document.querySelector('#tp .pct').textContent === '40%');
    t = await tbar();
    assert.deepEqual([t.on, t.from, t.trail.map((v) => Math.round(v * 100) / 100), Math.round(t.grip * 100) / 100], [16, 24, [0.6, 1], 0.6],
      'from the far end they fill from the right');
    await status({transition: 'wipe', transition_progress: 0, lever: {origin: 0, position: 0}});
    t = await tbar();
    assert.deepEqual([t.on, t.run], [0, true], 'a wipe runs Auto without a progress of its own');
    await status({transition: 'idle', transition_on_air: false});
    await page.waitForTimeout(300);
    assert.equal(Math.round((await tbar()).grip * 1000) / 1000, rest, 'once the mix ends the grip is back where the lever rests');
    // Cut flashes as pressed.
    await Promise.all([sent(), page.locator('#cut').click()]);
    commands.length = 0;
    assert(await page.locator('#cut').evaluate((c) => c.classList.contains('flash')));
    await page.waitForFunction(() => !$('cut').classList.contains('flash'));
    // The lever's grip follows it; its two ends are 0 and 1000.
    const track = await page.locator('#tp .track').boundingBox();
    for (const [x, want] of [[track.x + 2, 0], [track.x + track.width - 2, 1000]]) {
      await page.mouse.click(x, track.y + track.height / 2);
      assert.equal(await page.locator('#lever').inputValue(), String(want));
      assert.equal(await page.locator('#tp .track').evaluate((e) => e.style.getPropertyValue('--v')), String(want / 1000));
    }
    await page.waitForFunction(() => document.activeElement === $('scenes'));
    commands.length = 0;
    // Auto again: Halt or Complete, a switch whose choice this browser keeps.
    await page.locator('#tp .seg.again [data-value="complete"]').click();
    await reload();
    assert.deepEqual(await pressed('#tp .seg.again button'), ['false', 'true'], 'Complete is kept after a reload');
    await page.locator('#tp .seg.again [data-value="halt"]').click();

    // The dividers: deck | outputs and output | output, kept in this browser.
    const deckWidth = () => page.evaluate(() => $('deck').getBoundingClientRect().width);
    const wide = await deckWidth(), split = await page.locator('#split').boundingBox();
    await page.mouse.move(split.x + split.width / 2, split.y + 200);
    await page.mouse.down();
    await page.mouse.move(split.x + split.width / 2 - 100, split.y + 200, {steps: 4});
    await page.mouse.up();
    const dragged = await page.evaluate(() => store.get('deck'));
    assert(Math.abs(dragged - (wide - 100)) <= 4 && dragged === await deckWidth(), `a drag sets the deck width: ${[wide, dragged]}`);
    await page.locator('#split').focus();
    await page.keyboard.press('ArrowLeft');
    assert.equal(await page.evaluate(() => store.get('deck')), Math.round(await deckWidth()), 'the arrow keys step it');
    assert(Math.abs(await deckWidth() - (dragged - 24)) <= 1);
    const kept = await deckWidth();
    await reload();
    assert.equal(await deckWidth(), kept, 'the deck width is kept after a reload');
    await page.locator('#split').dblclick();
    assert.equal(await page.evaluate(() => store.get('deck')), null, 'a double-click restores the default');
    assert(Math.abs(await deckWidth() - wide) <= 1);
    const heights = () => page.evaluate(() => [...document.querySelectorAll('#wall > .viewer')].map((v) => v.getBoundingClientRect().height));
    const [a0, b0] = await heights(), vsplit = await page.locator('#vsplit').boundingBox();
    await page.mouse.move(vsplit.x + vsplit.width / 2, vsplit.y + vsplit.height / 2);
    await page.mouse.down();
    await page.mouse.move(vsplit.x + vsplit.width / 2, vsplit.y + vsplit.height / 2 + 80, {steps: 4});
    await page.mouse.up();
    const [a1, b1] = await heights();
    assert(a1 > a0 + 60 && b1 < b0 - 60, `a drag moves the output divider: ${[a0, b0, a1, b1]}`);
    const share = await page.evaluate(() => store.get('share'));
    assert(share > 0.55 && share < 0.8, `the first output's share is kept: ${share}`);
    await page.locator('#vsplit').focus();
    await page.keyboard.press('ArrowUp');
    assert(Math.abs(await page.evaluate(() => store.get('share')) - (share - 0.04)) < 0.01, 'the arrow keys step it');
    for (const _ of Array(12)) await page.keyboard.press('ArrowDown');
    assert.equal(await page.evaluate(() => store.get('share')), 0.8, 'down to a fifth for the other');
    const held = await heights();
    await reload();
    assert.deepEqual((await heights()).map(Math.round), held.map(Math.round), 'the shares are kept after a reload');
    assert.deepEqual(errors, []);

    // macOS's natural scrolling: a swipe up arrives as a positive delta and still means longer; a wheel that the
    // browser says is not inverted follows the classic rule.
    await page.context().close();
    ({page, errors, commands, sent} = await openPage(browser, {...bigShow(), viewport: {width: 1512, height: 860},
      init: () => Object.defineProperty(Navigator.prototype, 'platform', {get: () => 'MacIntel'})}));
    await page.waitForFunction(() => $('fade-s').value === '15f');
    await page.locator('#fade-s').hover();
    await page.mouse.wheel(0, 100);
    await page.waitForFunction(() => $('fade-s').value === '16f');
    await page.evaluate(() => {
      const e = new WheelEvent('wheel', {deltaY: 100, bubbles: true, cancelable: true});
      Object.defineProperty(e, 'webkitDirectionInvertedFromDevice', {value: false});
      $('fade-ctl').dispatchEvent(e);
    });
    assert.equal(await page.locator('#fade-s').inputValue(), '15f');
    await page.locator('#all-rate').hover();
    await Promise.all([sent(), page.mouse.wheel(0, 100)]);
    assert.deepEqual(commands.splice(0), [{command: 'dsk', all_rate: 13 / 30}], 'the key rates follow the same rule');
    assert.deepEqual(errors, []);
    console.log('Deck hints, keys, slots, stats, output and layout menus, pagers, scene pages, page fit, transition panel and dividers passed');
  } finally { await browser.close(); }
})().catch((e) => { console.error(e); process.exit(1); });
