// Playwright; all requests are intercepted, so this never controls a live mixer.
const assert = require('node:assert/strict');
const path = require('node:path');
const {launch, openPage, bigShow} = require('./page_fixture.cjs');

(async () => {
  const browser = await launch();
  try {
    const gpu = {index: 0, gpu: 49, decoder: 66, encoder: 75, memory_used_mib: 18432, memory_total_mib: 24576,
      power_draw_w: 50, power_limit_w: 72, encoder_sessions: 30, encoder_fps: 750, encoder_mpix_s: 1555.2};
    // What the GPU tip says about the board, from the backend's slower nvidia-smi queries.
    const facts = {name: 'NVIDIA L4', driver: '615.71.09', cuda: '13.4', compute_capability: '8.9', clocks_sm_mhz: 2040,
      clocks_mem_mhz: 6251, temperature_c: 76, pstate: 'P0', pcie_gen: 3, pcie_width: 16, persistence: true, compute_timeslice: 'Long'};
    const cuts = [74, 90, 106].map((ms, i) => ({id: i + 1, ms, state: 'measured'}));
    const aux = Array.from({length: 30}, (_, i) => ({id: `aux${i}`, label: `Aux ${i}`, layout: {preset: 'cells'},
      layouts: [], cells: [], canvas: {w: 1920, h: 1080}, fps: 25, running: true,
      playout: {frames: 100, missed_deadlines: i === 29 ? 2 : 0}, output_drops: i === 29 ? 1 : 0}));
    const clip = (name, frames, complete) => ({path: `/wipes/${name}.mov`, frames, bytes: frames * 1920 * 1080 * 1.5, complete});
    const sample = {setup_revision: 1, gpus: [{...gpu, ...facts}], host: {load1: 12, vcpus: 16, cpu_pct: 55,
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
        playout: {A: {missed_deadlines: 1}, B: {missed_deadlines: 0}},
        wipe_cache: {bytes: 200000000, budget_bytes: 600000000, clips: [clip('diagonal_sweep', 32, true), clip('sliding_panels', 8, false)]}},
      scenes: ['a', 'b'], aux_buses: aux,
      settings: {source_count: 192, source_counts: {video: 120, browser: 40, nv12: 32},
        canvas: {width: 1920, height: 1080, fps: 30, working_format: 'nv12'}, preview_codecs: ['h264'],
        decode: {count: 120, fps: 3600, mpix_s: 7464.96, by_codec: [{codec: 'hevc', sources: 100, fps: 3000}, {codec: 'h264', sources: 20, fps: 600}]},
        encode_by_codec: [{codec: 'h264', sessions: 30, fps: 750}],
        aux_buses: aux.map(a => a.id), preview_outputs: aux.map((a, i) => ({bus: a.id, label: a.label,
          codec: 'h264', mountpoint: 5000 + 4 * i, fps: 25}))}};
    const {page, errors, commands, requests, sent} = await openPage(browser, {state: sample, config: {compute_price: {hourly_usd: 1.2}}});
    await page.waitForFunction(() => document.querySelector('#power b').textContent === '50/72 W');
    // A tip's figures: the second column of its first table (ENC and DEC have a codec table under it).
    const values = id => page.locator(`#${id}-tip table:first-of-type td:nth-child(2)`).allTextContents();
    const rows = (selector) => page.locator(selector).evaluateAll((all) => all.map((t) =>
      [...t.rows].map((r) => [...r.cells].map((c) => c.textContent))));
    const tipTitle = id => page.locator(`#${id}-tip h4`).textContent();
    // One hover per anchor: an anchor with its tip has no title of its own or of a group around it, which the
    // browser would show stacked over the tip.
    const titledTips = () => page.evaluate(() => [...document.querySelectorAll('.tip')].map((t) => t.parentElement)
      .filter((anchor) => anchor.closest('[title]')).map((anchor) => anchor.id));
    assert.deepEqual(await titledTips(), []);
    const TIPPED = ['counts', 'lat', 'gpu-gpu', 'gpu-enc', 'gpu-dec', 'gpu-vram', 'power', 'wipe-cache', 'cpu', 'thread', 'hint'];
    assert.deepEqual(await page.evaluate((ids) => ids.filter((id) => !document.querySelector(`#${id} > .tip[role="tooltip"]`)), TIPPED), []);
    assert.equal(await page.locator('#load > .tip').count(), 0, 'LOAD has no tip');
    // Every meter in the strip: its label and its value.
    assert.deepEqual(await page.locator('#stats .m:not([hidden])').evaluateAll((all) => all.map((m) =>
      [m.id, m.querySelector('.lbl').textContent, m.querySelector(':scope > b').textContent])), [
      ['lat', 'Lat', '90 ms'], ['gpu-gpu', 'GPU', '49%'], ['gpu-enc', 'Enc', '75%'], ['gpu-dec', 'Dec', '66%'],
      ['gpu-vram', 'VRAM', '18.0/24.0 GB'], ['power', 'Power', '50/72 W'], ['wipe-cache', 'Wipes', '0.2/0.6 GB'],
      ['load', 'Load', '12.0/16'], ['cpu', 'CPU', '55%'], ['thread', 'Thread', '40%']]);
    // The counts are short in the strip and whole in their tip.
    assert.equal(await page.locator('#counts .txt').textContent(),
      '192 sources · 2 scenes · 120 NVDEC · 40 browser · 32 raw 4:2:0 · 1080p30 SDR');
    assert.equal(await tipTitle('counts'), 'Mixer setup');
    assert.deepEqual(await rows('#counts-tip table'), [[['', ''], ['Sources', '192'], ['Scenes', '2'], ['NVDEC', '120'],
      ['browser', '40'], ['raw 4:2:0', '32'], ['Canvas', '1920×1080 · 30p · SDR 4:2:0']]]);
    assert.equal(await tipTitle('power'), 'Power efficiency');
    assert.deepEqual(await values('power'), ['3.84', '1555.2', '31.10', '$0.00625']);
    assert.equal(await tipTitle('gpu-vram'), 'VRAM');
    assert.deepEqual(await rows('#gpu-vram-tip table'), [[['', ''], ['Sources / GiB used', '10.67'], ['Free', '6.00 GiB']]]);
    // ENC and DEC: totals, then the load by codec.
    assert.equal(await tipTitle('gpu-enc'), 'NVENC');
    assert.deepEqual(await rows('#gpu-enc-tip table'), [[['', ''], ['Encoders', '30'], ['Total FPS', '750'], ['MPix / s', '1555']],
      [['By codec', 'sessions', 'fps'], ['H.264', '30', '750']]]);
    assert.equal(await tipTitle('gpu-dec'), 'NVDEC');
    assert.deepEqual(await rows('#gpu-dec-tip table'), [[['', ''], ['Decoders', '120'], ['Total FPS', '3600'], ['MPix / s', '7465']],
      [['By codec', 'sources', 'fps'], ['HEVC', '100', '3000'], ['H.264', '20', '600']]]);
    // GPU: what the board is and how it runs.
    assert.equal(await tipTitle('gpu-gpu'), 'GPU 0 · NVIDIA L4');
    assert.match(await page.locator('#gpu-gpu-tip .about').textContent(), /^Share of the last second in which the GPU was running work/);
    const gpuFacts = [['', ''], ['Driver · CUDA', '615.71.09 · 13.4'], ['Compute capability', '8.9'], ['Memory', '24.0 GiB'],
      ['Power limit', '72 W'], ['Clocks · SM / memory', '2040 / 6251 MHz'], ['Temperature · state', '76 °C · P0'], ['PCIe', 'Gen3 ×16'],
      ['Persistence · time-slice', 'on · Long']];
    assert.deepEqual(await rows('#gpu-gpu-tip table'), [gpuFacts]);
    // A backend without the facts: the meters as before, the facts unknown.
    await page.evaluate(() => { window.older = {gpus: state.gpus}; state.gpus = state.gpus.map(({index, gpu, decoder, encoder, memory_used_mib, memory_total_mib,
      power_draw_w, power_limit_w, encoder_sessions, encoder_fps, encoder_mpix_s}) => ({index, gpu, decoder, encoder, memory_used_mib,
      memory_total_mib, power_draw_w, power_limit_w, encoder_sessions, encoder_fps, encoder_mpix_s}));
      const {decode, encode_by_codec} = state.settings; delete state.settings.decode; delete state.settings.encode_by_codec; renderMeters();
      Object.assign(window.older, {decode, encode_by_codec}); });
    assert.equal(await tipTitle('gpu-gpu'), 'GPU 0');
    assert.deepEqual(await rows('#gpu-gpu-tip table'), [[['', ''], ['Driver · CUDA', '— · —'], ['Compute capability', '—'], ['Memory', '24.0 GiB'],
      ['Power limit', '72 W'], ['Clocks · SM / memory', '— / — MHz'], ['Temperature · state', '— · —'], ['PCIe', '—'],
      ['Persistence · time-slice', '— · —']]]);
    assert.deepEqual(await rows('#gpu-enc-tip table'), [[['', ''], ['Encoders', '30'], ['Total FPS', '750'], ['MPix / s', '1555']]]);
    assert.deepEqual(await rows('#gpu-dec-tip table'), [[['', ''], ['Decoders', '—'], ['Total FPS', '—'], ['MPix / s', '—']]]);
    assert.equal(await page.locator('#gpu-dec > b').textContent(), '66%', 'DEC keeps its meter');
    await page.evaluate(() => { const {gpus, ...settings} = window.older; state.gpus = gpus; Object.assign(state.settings, settings); renderMeters(); });
    // WIPES: each clip decoded in GPU memory, by file name; one still loading; none yet without clips.
    assert.equal(await tipTitle('wipe-cache'), 'Wipe clips decoded in GPU memory');
    assert.deepEqual(await rows('#wipe-cache-tip table'), [[['', ''], ['diagonal_sweep.mov', '32 frames · 95 MB'],
      ['sliding_panels.mov', '8 frames · 24 MB · loading']]]);
    await page.evaluate(() => { const clips = state.status.wipe_cache.clips; state.status.wipe_cache.clips = []; renderWipeCache();
      window.noClips = document.querySelector('#wipe-cache-tip td').textContent; state.status.wipe_cache.clips = clips; renderWipeCache(); });
    assert.equal(await page.evaluate(() => window.noClips), 'none yet');
    assert.equal(await tipTitle('cpu'), 'CPU efficiency');
    assert.deepEqual(await values('cpu'), ['6.00', '32.0']);
    // THREAD shows the busiest thread. Its tip is one table of the mixer's threads by kind, the
    // largest total first and "other" last: count, total, mean and busiest thread. Under it
    // Chromium's cores' worth in its header, its processes and its GPU process's threads.
    assert.equal(await page.locator('#thread > b').textContent(), '40%');
    assert.equal(await tipTitle('thread'), 'CPU by thread and process · % of one core');
    const headings = (selector) => page.locator(selector).evaluateAll((all) => all.map((p) =>
      p.children.length ? [...p.children].map((c) => c.textContent) : p.textContent));
    assert.deepEqual(await headings('#thread-tip .grp, #thread-tip .sub'),
      [['Mixer', '6.00/16 cores'], ['Chromium', '1.48/16 cores'], 'GPU-process threads']);
    const tables = () => rows('#thread-tip table');
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
    const latTitle = () => tipTitle('lat');
    assert.deepEqual(await lat(), ['106.0 ms', '3', '1']);
    // No player is open: the bar shows the median of the mixer's own last three cuts.
    assert.equal(await page.locator('#lat > b').textContent(), '90 ms');
    // The number takes the bar's colour past 200 ms and past 400 ms.
    const latLevel = (ms) => page.evaluate((ms) => { const keep = cutLatency;
      cutLatency = {at: performance.now(), data: {latency: {ms}}}; renderLatency();
      const b = $('lat').querySelector(':scope > b'), out = [b.className, getComputedStyle(b).color];
      cutLatency = keep; renderLatency(); return out; }, ms);
    assert.equal((await latLevel(90))[0], 'ok');
    const [warn, bad, quick] = [await latLevel(300), await latLevel(600), await latLevel(90)];
    assert.equal(warn[0], 'warn'); assert.equal(bad[0], 'bad');
    assert.notEqual(warn[1], quick[1]); assert.notEqual(bad[1], warn[1]);
    await page.evaluate(() => { observeCuts(); observeCuts(); renderLatency(); });
    assert.deepEqual((await latLabels())[0], 'Cut p95 · 3', 'repeated samples are not new cuts');
    // Without a server window the totals since the start; with one, its figures and its span.
    assert.equal(await latTitle(), 'Latency & delivery · total');
    await page.evaluate(() => { state.recent = {window_s: 600, covered_s: 754, missed_deadlines: 0, output_drops: 2}; renderLatency(); });
    assert.deepEqual((await lat()).slice(1), ['0', '2']);
    assert.equal(await latTitle(), 'Latency & delivery · last 10 min');
    await page.evaluate(() => { state.recent.covered_s = 40; renderLatency(); });
    assert.equal(await latTitle(), 'Latency & delivery · last 40 s');
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
    // One line from 1280 px up: the counts give way first, every meter and tool keeps its room.
    await page.mouse.move(0, 500);
    const strip = () => page.evaluate(() => {
      const top = $('top').getBoundingClientRect(), inside = (e) => { const r = e.getBoundingClientRect(); return r.left >= top.left && r.right <= top.right + 0.5; };
      return {height: $('top').offsetHeight, scroll: $('top').scrollWidth - $('top').clientWidth,
              clipped: [...document.querySelectorAll('#top .m:not([hidden]), #top .tools > *')].filter((e) => !inside(e)).map((e) => e.id || e.className),
              cut: [...document.querySelectorAll('#top .m:not([hidden]) b, #top .lbl')].filter((e) => e.scrollWidth > e.clientWidth + 0.5).length};
    });
    const oneLine = (await strip()).height;
    assert(oneLine < 50, `the strip is one line at 1920 px: ${oneLine}`);
    for (const width of [1920, 1600, 1512, 1440, 1332, 1280]) {
      await page.setViewportSize({width, height: 1080});
      const box = await strip();
      assert.deepEqual(box, {height: oneLine, scroll: 0, clipped: [], cut: 0}, `the strip keeps one line at ${width} px`);
      if (process.env.METER_SCREENSHOTS) await page.screenshot({path: path.join(process.env.METER_SCREENSHOTS, `meters-${width}.png`)});
      for (const id of [...TIPPED, 'gpu-vram']) {
        await page.locator(`#${id}`).hover();
        const tip = await page.locator(`#${id}-tip`).boundingBox();
        assert(tip && tip.x >= 0 && tip.x + tip.width <= width, `${id} tooltip must fit at ${width} px`);
        if (id === "thread" && process.env.METER_SCREENSHOTS) await page.screenshot({path: path.join(process.env.METER_SCREENSHOTS, `thread-hover-${width}.png`)});
        if (id === "lat") {
          assert(await page.locator("#lat-tip").evaluate(t => t.scrollWidth <= t.clientWidth), "delivery text must stay inside tooltip");
          if (process.env.METER_SCREENSHOTS) await page.screenshot({path: path.join(process.env.METER_SCREENSHOTS, `lat-hover-${width}.png`)});
        }
      }
    }
    // Narrower, while the deck has its full width, the counts truncate rather than wrap.
    await page.setViewportSize({width: 1280, height: 1080});
    assert(await page.locator('#counts .txt').evaluate(t => t.scrollWidth > t.clientWidth), 'the counts give way first');
    await page.setViewportSize({width: 1920, height: 1080});
    await page.keyboard.press('Escape');
    const selector = page.locator('.viewer > header > .pick').first();
    await selector.locator('.picker').click();
    const menu = selector.locator('.menu[role="listbox"]');
    assert.equal(await menu.locator('[role="option"]').count(), 31, 'program plus every AUX');
    assert.deepEqual(await menu.locator('.grp').allTextContents(), ['Program', 'Monitoring']);
    assert(await menu.evaluate(e => e.scrollHeight > e.clientHeight), 'long output list scrolls');
    assert.equal(await menu.locator('[aria-selected="true"]').textContent(), 'ProgramH.264 · 1920×1080p30 SDR 4:2:0',
      'each output over its details; the one shown is selected');
    await page.keyboard.press('End');
    assert.match(await page.locator(':focus').textContent(), /Aux 29/);
    assert(await menu.evaluate(e => e.scrollTop > 0));
    await page.keyboard.press('Enter');
    assert.match(await selector.locator('.picker').textContent(), /Aux 29/);
    assert(await menu.isHidden(), 'a choice closes the menu');
    await selector.locator('.picker').click();
    assert.match(await menu.locator('[aria-selected="true"]').textContent(), /^Aux 29/, 'the menu marks the output shown');
    await page.keyboard.press('Home');
    assert.match(await page.locator(':focus').textContent(), /Program/);
    await page.keyboard.press('Escape');
    assert(await menu.isHidden(), 'Escape closes the menu');
    assert(await selector.locator('.picker').evaluate(b => document.activeElement === b), 'and gives the focus back to its button');
    // Missing readings remain unknown; all-board ratios must not silently use one board.
    await page.evaluate(() => {
      state.gpus.push({...state.gpus[0], index: 1, power_draw_w: null, encoder_mpix_s: null});
      state.host.mixer_cpu_pct = null;
      renderMeters(); renderHost();
    });
    assert.deepEqual(await values('power'), ['—', '—', '—', '$0.00625']);
    // Two GPUs: the meters with a tip list both in it, still without a title.
    assert.deepEqual(await titledTips(), []);
    assert.deepEqual((await values('gpu-enc')).slice(3), ['75%', '75%']);
    assert.equal(await tipTitle('gpu-gpu'), '2 GPUs · the meter shows the busiest');
    assert.deepEqual(await headings('#gpu-gpu-tip .grp'), [['GPU 0 · NVIDIA L4', '49%'], ['GPU 1 · NVIDIA L4', '49%']]);
    assert.equal(await page.locator('#gpu-gpu .lbl').textContent(), 'GPU ×2');
    assert.deepEqual(await values('cpu'), ['—', '—']);
    await page.evaluate(() => {
      state.setup_revision = 2;
      state.status.cut_latency = {direct: {id: 0, state: 'waiting', recent: []}};
      observeCuts(); renderLatency();
    });
    assert.equal((await page.locator('#lat-tip td:nth-child(2)').allTextContents())[0], '—', 'restart clears prior cut history');
    assert.equal(await page.locator('#lat > b').textContent(), '—', 'and the bar has no cut to show');
    // On-air shortcuts are checked only against this mocked mixer.
    await page.evaluate(() => { selected = 'a'; setDirect(true); armedSlot = 0; restFocus(); });
    await Promise.all([sent(), page.keyboard.press('ArrowRight')]);
    assert.deepEqual(commands.splice(0), [{command: 'preview', scene: 'b'}], 'arrows bypass Direct and slot assignment');
    await page.keyboard.press('ArrowRight');
    await page.waitForFunction(() => state.status.pvw_scene === 'a');
    assert.deepEqual(commands.splice(0), [{command: 'preview', scene: 'a'}], 'arrows wrap visible scenes');
    await page.evaluate(() => { selected = 'b'; currentTransition = 'fade'; tookTransition = true; armedSlot = null; });
    await Promise.all([sent(), page.keyboard.press('Space')]);
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
    assert.equal(await page.locator('#fade').getAttribute('aria-pressed'), 'true', 'Fade lights');
    assert.equal(await page.locator('#auto').getAttribute('title'), 'Auto: run the selected transition to Preview (Space)', 'Auto keeps its title');
    assert.equal(await page.locator('#direct').getAttribute('title'), 'Direct cut: picks cut straight to Program (Shift+D)', 'Direct keeps its title when on');
    assert.deepEqual(await page.locator('#fade-ctl .step').evaluateAll(bs => bs.map(b => [b.title, b.ariaLabel])),
      [['', 'One frame shorter'], ['', 'One frame longer']], 'the steppers show the rate box\'s title');
    assert.equal(await page.locator('#pvw-state').textContent(), 'picks cut to program', 'Preview says Direct is on');
    // Clicking the selector must not leave Space to it: Space after a click is Auto, not a second selection.
    await page.locator('#fade').click();
    await Promise.all([sent(), page.keyboard.press('Space')]);
    assert.deepEqual(commands.splice(0).map(c => c.command), ['fade'], 'Space after clicking Fade runs Auto');
    await Promise.all([sent(), page.keyboard.press('c')]);
    assert.deepEqual(commands.splice(0).map(c => [c.command, c.scene]), [['cut', 'a']], 'C cuts the Preview');
    // The lever sends its position with the mix's duration; Program shows the mix's progress.
    await page.waitForFunction(() => gripBack === null);   // the lever takes no press while the grip goes back to it
    await Promise.all([sent(),
      page.evaluate(() => { $('lever').value = 400; $('lever').dispatchEvent(new Event('input')); })]);
    assert.deepEqual(commands.splice(0).map(c => [c.command, c.position, c.duration_sec]), [['lever', 0.4, 0.5]]);
    // Program shows the lever's mix at once, and a mix from the polls.
    const pgmState = () => $('pgm-state').textContent;
    const poll = async (change) => {
      Object.assign(sample.status, change);
      await page.evaluate(() => refresh());
      await page.waitForFunction((c) => Object.entries(c).every(([k, v]) => JSON.stringify(state.status[k]) === JSON.stringify(v)), change);
    };
    assert.equal(await page.evaluate(pgmState), 'on air · fade 40% · lever');
    await poll({transition: 'crossfade', transition_on_air: true, transition_manual: true, transition_progress: 0.4,
                transition_duration_ms: 500, lever: {position: 0.4, origin: 0, engaged: true}});
    assert.equal(await page.evaluate(pgmState), 'on air · fade 40% · lever', 'and so does the poll');
    // This page's own take names the mix: a dip by its colour, a wipe by its clip; until that mix ends.
    assert.equal(await page.evaluate(() => { tookAs = {transition: 'crossfade', name: 'dip'}; renderBuses(); return $('pgm-state').textContent; }),
      'on air · dip 40% · lever');
    // A lever back at an end reads idle at once, before the poll says the mix is over.
    assert.equal(await page.evaluate(() => { $('lever').value = 0; $('lever').dispatchEvent(new Event('input')); return $('pgm-state').textContent; }),
      'on air', 'a lever back at its origin is idle');
    // A wipe first seen halfway eases in to its progress.
    await poll({transition: 'wipe', transition_on_air: true, transition_manual: false, transition_progress: 0.5,
                transition_duration_ms: 1e9, lever: {position: 0, origin: 0, engaged: true}});
    await page.evaluate(() => { tookAs = {transition: 'wipe', name: 'diagonal sweep'}; });
    await page.waitForFunction(() => document.querySelector('#tp .pct').textContent === '50%');
    assert.equal(await page.evaluate(pgmState), 'on air · diagonal sweep 50%', 'a wipe with a progress shows it, and so does the T-bar');
    // The mixer only records the lever under another's mix, so the lever takes no press and its arrows keep it put.
    commands.length = 0;
    assert.deepEqual(await page.evaluate(() => {
      const events = getComputedStyle($('lever')).pointerEvents;
      $('lever').value = 900; $('lever').dispatchEvent(new Event('input'));
      return [events, $('lever').value];
    }), ['none', '0'], 'the lever stays put under a wipe');
    assert.deepEqual(commands, [], 'and sends nothing');
    await poll({transition: 'idle', transition_progress: 0});
    await poll({transition: 'wipe', transition_progress: 0});
    assert.equal(await page.evaluate(pgmState), 'on air · wipe 0%', 'another controller\'s wipe has no name here');
    await poll({transition: 'idle', transition_on_air: false});
    await page.waitForFunction(() => gripBack === null);   // on its way back the grip takes a press only where the lever rests
    // Released, the lever gives the focus back: no focus ring stays on it and the shortcuts stay live.
    await Promise.all([sent(), page.locator('#lever').click()]);
    assert(await page.evaluate(() => document.activeElement === $('scenes')), 'a released lever gives the focus back');
    // A press off the grip puts it under the pointer at once, with one position; a press on the grip only picks it up.
    await page.waitForTimeout(150);
    assert.deepEqual(commands.splice(0).map(c => [c.command, c.position]), [['lever', 0.5]], 'a press off the grip sends one position');
    const track = await page.locator('#tp .track').boundingBox(), onGrip = track.x + track.width / 2 + 10;
    await page.mouse.move(onGrip, track.y + track.height / 2);
    await page.mouse.down();
    await page.waitForTimeout(150);
    assert.deepEqual(commands.splice(0), [], 'a press on the grip puts no mix on air');
    await page.mouse.up();
    commands.length = 0;
    await page.evaluate(() => { setDirect(true); restFocus(); });
    assert.equal(await page.locator('#direct').getAttribute('aria-pressed'), 'true', 'Direct lights in the Preview box');
    await Promise.all([sent(), page.keyboard.press('2')]);
    assert.equal(commands.splice(0)[0].command, 'cut', 'a Direct pick cuts whatever Auto is set to');
    await page.evaluate(() => restFocus());
    assert.equal(await page.locator('#fade').getAttribute('aria-pressed'), 'true');
    await page.keyboard.press('d');
    assert.deepEqual(await page.evaluate(() => [$('dip').ariaPressed, $('fade').ariaPressed, direct]), ['true', 'false', true], 'D selects Dip, Direct stays');
    await page.keyboard.press('Shift+D');
    assert.deepEqual(await page.evaluate(() => [$('dip').ariaPressed, direct]), ['true', false], 'Shift+D turns Direct off, Dip stays');
    await page.keyboard.press('f');
    await page.evaluate(() => restFocus());
    // Keys: one following the All rate (null: no rate of its own), one cutting at its own rate.
    sample.dsk = [{id: 'logo', on: true, rate: null, all_rate: 0.4}, {id: 'title', on: false, rate: 0, all_rate: 0.4}];
    await page.evaluate(async () => { await refresh(); restFocus(); });
    await page.waitForFunction(() => state.dsk?.length === 2);
    assert.deepEqual(commands, [], 'the keys send nothing when they show up');
    await page.keyboard.press('k');
    await page.waitForFunction(() => !keysPending && state.dsk.every(key => !key.on));
    assert.deepEqual(commands.map(c => [c.key, c.on]), [['logo', false]], 'mixed keys all go off');
    commands.length = 0;
    await page.keyboard.press('k');
    await page.waitForFunction(() => !keysPending && state.dsk.every(key => key.on));
    assert.deepEqual(commands.map(c => [c.key, c.on]).sort(), [['logo', true], ['title', true]], 'off keys all go on');
    assert(commands.every(c => !('fade_seconds' in c) && c.curve), 'K takes each key at its own rate, with the Fade curve');
    commands.length = 0;
    // All is K's button.
    await page.locator('#all-take').click();
    await page.waitForFunction(() => !keysPending && state.dsk.every(key => !key.on));
    assert.deepEqual(commands.map(c => [c.key, c.on]).sort(), [['logo', false], ['title', false]], 'All takes every key off');
    commands.length = 0;
    await page.locator('#all-take').click();
    await page.waitForFunction(() => !keysPending && state.dsk.every(key => key.on));
    assert.deepEqual(commands.map(c => [c.key, c.on]).sort(), [['logo', true], ['title', true]], 'and on again');
    assert(commands.every(c => !('fade_seconds' in c) && c.curve));
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
    await Promise.all([sent(), page.keyboard.press('c')]);
    assert.deepEqual(commands.splice(0).map(c => c.command), ['cut'], 'with BKGD a take is the background take');
    await nextButton(1).click();
    assert.equal(await page.locator('#pvw-state').textContent(), 'Auto · keys');
    assert(await page.locator('#lever').isDisabled(), 'the lever mixes the background only');
    await Promise.all([sent(), page.keyboard.press('Space')]);
    assert.deepEqual(commands.splice(0).map(c => [c.command, c.keys, c.toggle, c.fade_seconds]), [['dsk', ['title'], true, 0.5]],
      'Space after BKGD is Auto, which fades the armed keys alone at the Fade rate');
    await page.evaluate(() => restFocus());
    await Promise.all([sent(), page.keyboard.press('c')]);
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
    await Promise.all([sent(), page.locator('#ftb').click()]);
    assert.deepEqual(commands.splice(0).map(c => [c.command, c.toggle, c.fade_seconds]), [['ftb', true, 0.5]]);
    await page.waitForFunction(() => $('ftb').classList.contains('black'));
    assert.equal(await page.locator('#ftb').getAttribute('aria-pressed'), 'true');
    assert.equal(await page.locator('#pgm-state').textContent(), 'on air · black');
    await Promise.all([sent(), page.keyboard.press('Space')]);
    assert.deepEqual(commands.splice(0).map(c => c.command), ['fade'], 'Space after FTB is Auto, not FTB again');
    await Promise.all([sent(), page.locator('#ftb').click()]);
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
    const rate = async (text) => { await page.locator('#fade-s').fill(text); return page.locator('#pvw-state').textContent(); };
    assert.equal(await rate('45'), 'Auto · fade 1:15', 'from a second the rate shows as seconds:frames');
    await page.locator('#fade-s').press('Enter');
    assert.equal(await page.locator('#fade-s').inputValue(), '1:15', 'an entry shows as the panel would');
    await Promise.all([sent(), page.keyboard.press('Space')]);
    assert.deepEqual(commands.splice(0).map(c => [c.command, c.duration_sec]), [['fade', 1.5]], 'the command takes seconds');
    await page.locator('#fade-s').press('ArrowDown');
    assert.equal(await page.locator('#fade-s').inputValue(), '1:14', 'arrow keys step a frame');
    await page.evaluate(() => restFocus());
    await page.locator('#fade-ctl [aria-label="One frame longer"]').click();
    assert.equal(await page.locator('#fade-s').inputValue(), '1:15', 'the small arrows step a frame');
    assert(await page.evaluate(() => document.activeElement === $('scenes')), 'and leave the focus where it was');
    await page.locator('#fade-ctl [aria-label="One frame longer"]').click({modifiers: ['Shift']});
    assert.equal(await page.locator('#fade-s').inputValue(), '2:15', 'with Shift they step a second');
    await page.locator('#fade-ctl [aria-label="One frame shorter"]').click({modifiers: ['Shift']});
    assert.equal(await page.locator('#fade-s').inputValue(), '1:15');
    // The wheel: a frame per notch, up longer. Off macOS a wheel turned up arrives as a negative delta.
    await page.locator('#fade-s').hover();
    await page.mouse.wheel(0, 100);
    await page.waitForFunction(() => $('fade-s').value === '1:14');
    for (const _ of [1, 2]) await page.mouse.wheel(0, -100);
    await page.waitForFunction(() => $('fade-s').value === '1:16');
    await page.mouse.wheel(0, 10);
    assert.equal(await page.locator('#fade-s').inputValue(), '1:16', 'the wheel steps a frame per notch, not per small delta');
    await page.keyboard.down('Shift');
    await page.mouse.wheel(0, -100);
    await page.keyboard.up('Shift');
    await page.waitForFunction(() => $('fade-s').value === '2:16');
    await page.locator('#fade-ctl .step[data-step="-1"]').hover();
    await page.mouse.wheel(0, 100);
    await page.waitForFunction(() => $('fade-s').value === '2:15');   // anywhere over the rate control, its steps too
    // Stepping stops at 30 s; a typed rate has no cap.
    await rate('29:20');
    await page.keyboard.down('Shift');
    await page.locator('#fade-ctl .step[data-step="1"]').click();
    await page.keyboard.up('Shift');
    assert.equal(await page.locator('#fade-s').inputValue(), '30:00', 'steps stop at 30 s');
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
    await page.evaluate(() => { showError(''); restFocus(); });
    // Each key takes at its own rate, or the All rate while it has none: a following key's tag is faint, its own bright.
    const tag = n => page.locator(`#keys .key:nth-child(${n}) .rt`), keyRate = page.locator('#keys .rt.edit input');
    const tags = () => page.locator('#keys .rt').evaluateAll(all => all.map(t => [t.textContent, t.classList.contains('own')]));
    const showsTags = (want) => page.waitForFunction((want) => JSON.stringify([...document.querySelectorAll('#keys .rt')]
      .map(t => [t.textContent, t.classList.contains('own')])) === JSON.stringify(want), want);
    commands.length = 0;
    assert.deepEqual(await tags(), [['12f', false], ['CUT', true]], 'each key shows its rate: the All rate faint, its own bright');
    assert.equal(await page.locator('#all-rate').textContent(), '12f', 'the All rate shows beside All');
    assert.match(await tag(1).getAttribute('title'), /^Logo: follows All · click: cut, ¼ s, ½ s, 1 s, then back to All/);
    assert.match(await tag(2).getAttribute('title'), /^Title: its own rate/);
    // F2 types the rate in place.
    await tag(1).focus();
    await page.keyboard.press('F2');
    assert.equal(await keyRate.inputValue(), '12f', 'a following key starts from the All rate');
    await Promise.all([page.waitForResponse(r => r.url().endsWith('/api/state')), page.evaluate(() => refresh())]);
    await page.evaluate(() => new Promise(r => requestAnimationFrame(r)));
    assert.equal(await keyRate.count(), 1, 'a poll keeps the field being edited');
    await keyRate.press('ArrowUp');
    assert.equal(await keyRate.inputValue(), '13f', 'the arrow keys step a frame');
    await keyRate.fill('20');
    await Promise.all([sent(), keyRate.press('Enter')]);
    assert.deepEqual(commands.splice(0), [{command: 'dsk', key: 'logo', rate: 20 / 30}], 'Enter sets the rate, in seconds');
    await showsTags([['20f', true], ['CUT', true]]);
    await tag(1).focus();
    await page.keyboard.press('F2');
    await keyRate.fill('5');
    await keyRate.press('Escape');
    assert.deepEqual(commands, [], 'Escape keeps the rate');
    assert.equal(await tag(1).textContent(), '20f');
    await tag(2).focus();
    await page.keyboard.press('F2');
    await keyRate.fill('1:40');
    await keyRate.press('Enter');
    assert.match(await page.locator('#error').textContent(), /key's rate/);
    assert.deepEqual(commands, [], 'a bad rate sends nothing');
    await tag(2).focus();
    await page.keyboard.press('F2');
    await keyRate.fill('15');
    await Promise.all([sent(), page.evaluate(() => restFocus())]);
    assert.deepEqual(commands.splice(0), [{command: 'dsk', key: 'title', rate: 0.5}], 'leaving the field sets the rate too');
    await tag(1).focus();
    await page.keyboard.press('F2');
    await keyRate.fill('c');
    await Promise.all([sent(), keyRate.press('Enter')]);
    assert.deepEqual(commands.splice(0), [{command: 'dsk', key: 'logo', rate: 0}], 'c makes the key cut');
    await showsTags([['CUT', true], ['15f', true]]);
    await tag(1).focus();
    await page.keyboard.press('F2');
    await keyRate.fill('all');
    await Promise.all([sent(), keyRate.press('Enter')]);
    assert.deepEqual(commands.splice(0), [{command: 'dsk', key: 'logo', rate: null}], '"all" makes the key follow All again');
    await showsTags([['12f', false], ['15f', true]]);
    await tag(1).focus();
    await page.keyboard.press('F2');
    await page.evaluate(() => restFocus());
    await page.evaluate(() => refresh());
    assert.deepEqual(commands, [], 'a following key left unchanged keeps following');
    // Even when the shown rate is the All rate rounded to frames.
    sample.dsk.forEach((k) => { k.all_rate = 0.3667; });
    await page.evaluate(() => refresh());
    await showsTags([['11f', false], ['15f', true]]);
    await tag(1).focus();
    await page.keyboard.press('F2');
    await keyRate.press('Enter');
    assert.deepEqual(commands, [], 'an untouched rounded rate sends nothing');
    sample.dsk.forEach((k) => { k.all_rate = 0.4; });
    await page.evaluate(() => refresh());
    await showsTags([['12f', false], ['15f', true]]);
    // A click steps the key's own presets at the show's rate (cut, ¼ s, ½ s, 1 s), then back to All. The tag shows each
    // step at once; the mixer gets the last of a burst of clicks.
    let reply = sent();
    await tag(1).click();
    assert.deepEqual((await tags())[0], ['CUT', true], 'the first click gives a following key its own cut');
    await reply;
    assert.deepEqual(commands.splice(0), [{command: 'dsk', key: 'logo', rate: 0}]);
    const burst = [];
    reply = sent();
    for (const _ of [1, 2, 3, 4]) { await tag(1).click(); burst.push((await tags())[0]); }
    assert.deepEqual(burst, [['8f', true], ['15f', true], ['1:00', true], ['12f', false]], 'then a quarter, a half, a whole second and back to All');
    await reply;
    assert.deepEqual(commands.splice(0), [{command: 'dsk', key: 'logo', rate: null}], 'one command for the burst');
    await page.evaluate(() => refresh());
    await showsTags([['12f', false], ['15f', true]]);
    // The All rate: a click steps its presets, the wheel a frame; only the keys that follow it change.
    await Promise.all([sent(), page.locator('#all-rate').click()]);
    assert.deepEqual(commands.splice(0), [{command: 'dsk', all_rate: 0.5}], 'a click on the All rate steps its presets');
    await showsTags([['15f', false], ['15f', true]]);
    await page.locator('#all-rate').hover();
    await Promise.all([sent(), page.mouse.wheel(0, -100)]);
    assert.deepEqual(commands.splice(0), [{command: 'dsk', all_rate: 16 / 30}], 'the wheel turned up adds a frame to the All rate');
    await showsTags([['16f', false], ['15f', true]]);
    // F2 types the All rate in place too, and the polls keep the page live meanwhile.
    await page.locator('#all-rate').focus();
    await page.keyboard.press('F2');
    const allField = page.locator('#keys-card .allkeys .rt.edit input');
    assert.equal(await allField.inputValue(), '16f');
    for (const _ of [1, 2]) await page.evaluate(() => refresh());
    assert(await page.locator('#conn').isHidden(), 'a poll with the All rate being typed renders');
    await allField.press('Escape');
    assert.equal(await page.locator('#all-rate').textContent(), '16f', 'Escape brings the All tag back');
    commands.length = 0;
    assert.equal(await page.locator('#all-rate').textContent(), '16f');
    // The wheel over a key's tag gives it its own rate, from the one it shows; with Shift a second.
    await tag(1).hover();
    await Promise.all([sent(), page.mouse.wheel(0, 100)]);
    assert.deepEqual(commands.splice(0), [{command: 'dsk', key: 'logo', rate: 0.5}], 'the wheel turned down takes a frame off');
    await showsTags([['15f', true], ['15f', true]]);
    await tag(2).hover();
    await page.keyboard.down('Shift');
    await Promise.all([sent(), page.mouse.wheel(0, -100)]);
    await page.keyboard.up('Shift');
    assert.deepEqual(commands.splice(0), [{command: 'dsk', key: 'title', rate: 1.5}], 'with Shift the wheel steps a second');
    await showsTags([['15f', true], ['1:15', true]]);
    await page.evaluate(() => restFocus());
    const take = n => page.locator(`#keys .key:nth-child(${n}) .take`);
    assert.match(await take(2).getAttribute('title'), /^Title · take off air at 1:15 · Shift: cut/);
    await Promise.all([sent(), take(2).click()]);
    const click = commands.splice(0)[0];
    assert(click.key === 'title' && !('fade_seconds' in click) && click.curve, 'a click takes the key at its own rate');
    await Promise.all([sent(), page.keyboard.press('Space')]);
    assert.notEqual(commands.splice(0)[0].command, 'dsk', 'Space with the key focused is Auto, not the key again');
    await Promise.all([sent(), take(2).click({modifiers: ['Shift']})]);
    assert.equal(commands.splice(0)[0].fade_seconds, 0, 'Shift+click cuts');
    // A mixer without the All rate: no All tag, and a key's presets go round without following anything.
    sample.dsk = [{id: 'logo', on: true, rate: 1}, {id: 'title', on: true, rate: 0}];
    await page.evaluate(() => refresh());
    await showsTags([['1:00', true], ['CUT', true]]);
    assert(await page.locator('#all-rate').isHidden(), 'no All rate, no All tag');
    assert.deepEqual(commands, [], 'nothing is sent when the keys change');
    await Promise.all([sent(), tag(1).click()]);
    assert.deepEqual(commands.splice(0), [{command: 'dsk', key: 'logo', rate: 0}], 'after a second the presets start again at cut');
    sample.dsk = [{id: 'logo', on: true, rate: null, all_rate: 0.4}, {id: 'title', on: true, rate: 0, all_rate: 0.4}];
    await page.evaluate(async () => { await refresh(); restFocus(); });
    await showsTags([['12f', false], ['CUT', true]]);
    // Auto pressed again during its own mix: Halt sends the fade again (the mixer holds it), Complete cuts.
    const again = v => page.locator(`#auto-ctl button[data-value="${v}"]`);
    assert.deepEqual([await again('halt').getAttribute('aria-pressed'), await again('complete').getAttribute('aria-pressed')], ['true', 'false']);
    await page.evaluate(() => { nextBkgd = true; selectTransition('fade'); });
    // A wipe this page drew to its end is over: a wipe waiting for air after it is a new take, and C cuts to Preview.
    const shown = sample.status, buses = {pgm_scene: 'a', pvw_scene: 'b'};
    sample.status = {...shown, ...buses, transition: 'wipe', transition_on_air: true, transition_manual: false, transition_progress: 0.99,
                     transition_duration_ms: 1000};
    await page.evaluate(() => refresh());
    await page.waitForFunction(() => mix?.ended > 0 && mix.drawn >= 1);
    sample.status = {...shown, ...buses, transition: 'idle', transition_on_air: false, transition_progress: 0};
    await page.evaluate(() => refresh());
    sample.status = {...shown, ...buses, transition: 'wipe', transition_on_air: false, transition_progress: 0, transition_duration_ms: 1000};
    await page.evaluate(async () => { await refresh(); restFocus(); });
    await page.waitForFunction(() => state.status.transition === 'wipe' && !state.status.transition_on_air);
    await Promise.all([sent(), page.keyboard.press('c')]);
    assert.deepEqual(commands.splice(0).map(c => [c.command, c.scene]), [['cut', 'b']], 'C after a wipe drawn to its end cuts to Preview');
    sample.status = {...shown, transition: 'crossfade', transition_on_air: true, transition_manual: false};
    await page.evaluate(async () => { await refresh(); restFocus(); });
    await page.waitForFunction(() => state.status.transition === 'crossfade');
    await Promise.all([sent(), page.keyboard.press('Space')]);
    assert.deepEqual(commands.splice(0).map(c => c.command), ['fade'], 'Halt: Auto again is the mixer\'s own second press');
    await again('complete').click();
    assert.deepEqual([await again('halt').getAttribute('aria-pressed'), await again('complete').getAttribute('aria-pressed')], ['false', 'true']);
    await Promise.all([sent(), page.keyboard.press('Space')]);
    assert.deepEqual(commands.splice(0).map(c => c.command), ['cut'], 'Complete: Auto again finishes the mix at once');
    sample.status = {...sample.status, transition: 'wipe'};
    await page.evaluate(async () => { await refresh(); restFocus(); });
    await page.waitForFunction(() => state.status.transition === 'wipe');
    await Promise.all([sent(), page.keyboard.press('Space')]);
    assert.deepEqual(commands.splice(0).map(c => c.command), ['cut'], 'Complete finishes a wipe too');
    sample.status = {...sample.status, transition: 'idle', transition_on_air: false};
    await page.evaluate(async () => { await refresh(); restFocus(); });
    await page.waitForFunction(() => state.status.transition === 'idle');
    await Promise.all([sent(), page.keyboard.press('Space')]);
    assert.deepEqual(commands.splice(0).map(c => c.command), ['fade'], 'with no mix running Auto starts one');
    assert.equal(await page.evaluate(() => store.get('autoAgain')), 'complete', 'the choice is kept');
    await page.hover('#hint');
    assert.match(await page.locator('#hint-tip').innerText(), /Auto to Preview \(again during its mix: Halt or Complete\)/);
    await again('halt').click();
    // The slots card edits the viewer last clicked among those showing a bus with slots.
    for (const aux of sample.aux_buses.slice(0, 2)) Object.assign(aux, {cells: [0, 1].map(slot => ({role: 'slot', slot})), scenes: ['a', 'b']});
    const outs = await page.evaluate(() => viewers.map(v => v.out));
    assert.equal(outs.length, 2, 'two viewers on the wall');
    const show = (a, b) => page.evaluate(async ([a, b]) => { viewers[0].out = a; viewers[1].out = b; await refresh(); }, [a, b]);
    await show('aux0', 'aux1');
    await page.waitForFunction(() => !$('slots-card').hidden && viewers[1].grid === 'aux1');
    assert.equal(await page.locator('#slots-title').textContent(), 'Slots');
    assert.equal(await page.locator('#slot-note').textContent(), 'Aux 0 · click a slot, then a scene', 'the first viewer with slots by default');
    assert.equal(await page.locator('#slot-keys').textContent(), '⇧1–2', 'Shift and a digit for each slot');
    const editing = () => page.evaluate(() => viewers.map(v => v.el.classList.contains('edit')));
    assert.deepEqual(await editing(), [true, false], 'with two on screen, the edited one says so');
    await page.locator('.viewer').nth(1).locator('.screen').click();
    assert.match(await page.locator('#slot-note').textContent(), /^Aux 1 · /, 'a click on a viewer makes the card edit its bus');
    assert.deepEqual(await editing(), [false, true]);
    commands.length = 0;
    await Promise.all([sent(), page.keyboard.press('Space')]);
    assert.deepEqual(commands.splice(0).map(c => c.command), ['fade'], 'Space after a click on a viewer is Auto: the shortcuts are global');
    // A slot armed, a scene picked: the slot takes it; Shift and a digit put the selected scene in a slot; × clears one.
    await page.locator('#slots .slotc:nth-child(2) .slot').click();
    assert.equal(await page.locator('#slot-note').textContent(), 'Aux 1 · pick a scene for M2');
    assert(await page.locator('#scenes').evaluate(g => g.classList.contains('armed')), 'the scenes wait for the pick');
    await Promise.all([sent(), page.locator('#scenes .tile').first().click()]);
    assert.deepEqual(commands.splice(0).map(c => [c.command, c.bus, c.scenes]), [['aux', 'aux1', ['a', 'a']]]);
    await page.evaluate(() => { selected = 'b'; restFocus(); });
    await Promise.all([sent(), page.keyboard.press('Shift+Digit1')]);
    assert.deepEqual(commands.splice(0).map(c => [c.command, c.bus, c.scenes]), [['aux', 'aux1', ['b', 'b']]]);
    await Promise.all([sent(), page.locator('#slots .slotc:nth-child(1) .clear').click()]);
    assert.deepEqual(commands.splice(0).map(c => [c.command, c.bus, c.scenes]), [['aux', 'aux1', [null, 'b']]]);
    await show('aux0', 'program');
    await page.waitForFunction(() => !viewers[1].grid);
    assert.match(await page.locator('#slot-note').textContent(), /^Aux 0 · /, 'the other takes over when it stops showing slots');
    assert.deepEqual(await editing(), [false, false], 'one on screen needs no marker');
    for (const aux of sample.aux_buses.slice(0, 2)) Object.assign(aux, {cells: [], scenes: []});
    await show(...outs);
    await page.locator('#hint').hover();
    assert.match(await page.locator('#hint-tip').innerText(), /Browse Preview only/);
    assert.match(await page.locator('#hint-tip').innerText(), /Auto to Preview/);
    assert.deepEqual(await page.locator('#hint-tip tbody tr').evaluateAll((rows) => rows.map((r) => r.innerText.replace(/\s+/g, ' ')).filter((r) => /^(D|W|⇧D) /.test(r))),
      ['D Select dip for Auto', 'W Select a wipe for Auto; again: the next wipe', '⇧D Direct cut: picks cut straight to Program']);
    const shortcutTip = await page.locator('#hint-tip').boundingBox();
    assert(shortcutTip.y >= 0 && shortcutTip.y + shortcutTip.height <= 1080, 'shortcut hover fits above the bottom bar');
    await page.setViewportSize({width: 390, height: 844});
    // The keyboard reaches a meter's tip too.
    await page.locator('#gpu-vram').focus();
    await page.keyboard.press('Tab');
    assert(await page.locator('#power').evaluate(m => document.activeElement === m));
    const tip = await page.locator('#power-tip').boundingBox();
    assert(tip.x >= 0 && tip.x + tip.width <= 390);
    // A failed poll shows its reason in the strip; the next good one hides it.
    await page.route('**/api/state', (route) => route.fulfill({status: 502, contentType: 'application/json',
      body: JSON.stringify({error: 'mixer not reachable: connection refused'})}), {times: 1});
    await page.evaluate(() => refresh());
    assert.equal(await page.locator('#conn').textContent(), 'mixer not reachable: connection refused');
    assert(await page.locator('#conn').isVisible(), 'the reason shows');
    await page.route('**/api/state', (route) => route.abort('connectionrefused'), {times: 1});
    await page.evaluate(() => refresh());
    assert(await page.locator('#conn').isVisible() && (await page.locator('#conn').textContent()).trim(), 'a refused poll shows its reason too');
    await page.evaluate(() => refresh());
    assert(await page.locator('#conn').isHidden(), 'a good poll hides it');
    // On one line a long reason is cut short in the strip, never widening the page; its title has all of it.
    await page.setViewportSize({width: 1512, height: 860});
    const longReason = 'mixer not reachable: connection refused while asking for the state, retrying every second';
    await page.route('**/api/state', (route) => route.fulfill({status: 502, contentType: 'application/json',
      body: JSON.stringify({error: longReason})}), {times: 1});
    await page.evaluate(() => refresh());
    assert(await page.locator('#conn').isVisible());
    assert.equal(await page.locator('#conn').getAttribute('title'), longReason);
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'a long reason does not widen the page');
    await page.evaluate(() => refresh());
    assert(await page.locator('#conn').isHidden());
    // A click on a heading keeps the focus in the menu: the shortcuts stay off and Escape still closes it.
    await page.mouse.move(960, 700);
    await page.keyboard.press('Escape');   // no tip left over the picker
    await selector.locator('.picker').click();
    await menu.locator('.grp').first().click();
    commands.length = 0;
    await page.keyboard.press('2');
    assert.deepEqual(commands, [], 'no shortcut while a menu is open');
    await page.keyboard.press('Escape');
    assert(await menu.isHidden(), 'Escape closes the menu after a click on its heading');
    // The default deck leaves the outputs column its 300 px minimum below 1266 px too.
    await page.setViewportSize({width: 1280, height: 720});
    assert(await page.evaluate(() => { const m = document.querySelector('main'), s = getComputedStyle(m);
      return $('deck').getBoundingClientRect().width <= m.clientWidth - parseFloat(s.paddingLeft) - parseFloat(s.paddingRight) - 312 + 0.5; }),
      'the default deck keeps the outputs their minimum');
    // The divider keeps the width it shows: dragged past the outputs' minimum, the deck stays put when the window widens.
    const split = await page.locator('#split').boundingBox();
    await page.mouse.move(split.x + split.width / 2, split.y + 100);
    await page.mouse.down();
    await page.mouse.move(1275, split.y + 100, {steps: 4});
    await page.mouse.up();
    const deckNow = () => page.evaluate(() => $('deck').getBoundingClientRect().width), clamped = await deckNow();
    assert.equal(await page.evaluate(() => store.get('deck')), Math.round(clamped), 'the divider saves the width it shows');
    await page.setViewportSize({width: 1920, height: 1050});
    assert(Math.abs(await deckNow() - clamped) <= 1, 'a wider window keeps the dragged deck');
    await page.locator('#split').dblclick();
    // × forgets the closed viewer's output: 2 opens the usual second output. 1 only hides it, and 2 brings it back.
    const viewerOuts = () => page.evaluate(() => viewers.map(v => v.out)), count = (n) => page.locator(`#viewer-count [data-n="${n}"]`).click();
    await page.evaluate(() => { viewers[1].out = viewers[0].out; return refresh(); });
    await page.locator('.viewer').nth(1).locator('.close').click();
    const second = await page.evaluate(() => secondOutput().value), [first] = await viewerOuts();
    assert.notEqual(second, first);
    await count(2);
    assert.deepEqual(await viewerOuts(), [first, second], '× then 2 opens the second output, not the closed one');
    const other = await page.evaluate((second) => {
      viewers[1].out = outputOptions().map(o => o.value).find(v => v !== second && v !== viewers[0].out) ?? viewers[0].out;
      return refresh().then(() => viewers[1].out);
    }, second);
    await count(1);
    await count(2);
    assert.deepEqual(await viewerOuts(), [first, other], '1 then 2 brings back the hidden output');
    // The Signal flow tab and the graph view are gone, and nothing asks for the graph's port.
    assert.equal(await page.locator('#tab-graph, #graph, #graph-link, #graph-open').count(), 0);
    assert.deepEqual(requests.filter(url => url.includes(':22222')), []);
    assert.deepEqual(errors, []);
    await tbarUnderJitter(browser);
    console.log('Meters and their tips, latency counters, 30-AUX selector, shortcuts, keys, rates and the T-bar under jitter passed');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exit(1); });

// The T-bar over a jittery link: a mixer's takes and lever as far as its status shows them (frame by frame on its own
// clock, the bridge's 200 ms cache, now and then the reply before), every reply and lever message late by its own
// delay. The page's animation frames come from a 120 Hz clock, as on a ProMotion display, and the T-bar is recorded on
// every one of them.
async function tbarUnderJitter(browser) {
  const now = () => performance.now(), F = 1000 / 30, sleep = (ms) => new Promise((r) => setTimeout(r, Math.max(0, ms)));
  let seed = 7;
  const random = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
  // src/mixer/primitives/fade_curve.hpp, and its inverse
  const CURVES = {linear: [(x) => x, (p) => p], 'ease-in': [(x) => x * x, Math.sqrt],
    'ease-out': [(x) => x * (2 - x), (p) => 1 - Math.sqrt(1 - p)],
    'ease-in-out': [(x) => x * x * (3 - 2 * x), (p) => 0.5 - Math.sin(Math.asin(1 - 2 * p) / 3)]};
  const WIPE_MS = 32 * F;
  // A fade's progress is its curve's, a dip's and a wipe's their clock's (fade.cpp, wipe.cpp).
  const mixer = {mode: 'idle', start: 0, duration: 0, curve: 'linear', heldAt: null, manual: false, origin: 0, position: 0, cutAt: 0, takes: [],
    x(t) { return Math.min(1, Math.max(0, ((this.heldAt ?? Math.floor(t / F) * F) - this.start) / this.duration)); },
    status(t) {
      if (this.mode === 'cut' && t >= this.cutAt) this.mode = 'idle';
      if (['crossfade', 'wipe'].includes(this.mode) && !this.manual && this.x(t) >= 1) this.mode = 'idle';
      const on = ['crossfade', 'wipe'].includes(this.mode), manual = on && this.manual;
      return {transition: this.mode, transition_on_air: on && t >= this.start, transition_paused: on && this.heldAt !== null,
        transition_manual: manual, transition_progress: manual ? Math.abs(this.position - this.origin) : on ? CURVES[this.curve][0](this.x(t)) : 0,
        transition_duration_ms: Math.round(this.duration), lever: {position: this.position, origin: this.origin, engaged: true}, now_pts_ms: Math.floor(t) + 1e6};
    },
    command(c, t) {
      const next = Math.ceil(t / F) * F;
      this.status(t);
      if (c.command === 'cut') Object.assign(this, {mode: 'cut', cutAt: next + F, manual: false, heldAt: null});
      if (c.command === 'wipe' && this.mode === 'idle') {   // a second wipe is ignored
        Object.assign(this, {mode: 'wipe', start: next + 3 * F, duration: WIPE_MS, curve: 'linear', heldAt: null});
        this.takes.push({start: this.start, duration: this.duration, curve: 'linear'});
      }
      if (c.command === 'fade' && this.mode === 'crossfade' && this.manual)   // Auto takes over the lever's mix
        Object.assign(this, {manual: false, start: next - Math.abs(this.position - this.origin) * this.duration});
      else if (c.command === 'fade' && this.mode === 'crossfade' && this.heldAt === null) this.heldAt = next;
      else if (c.command === 'fade' && this.mode === 'crossfade') Object.assign(this, {start: this.start + next - this.heldAt, heldAt: null});
      else if (c.command === 'fade' && this.mode !== 'wipe') {
        Object.assign(this, {mode: 'crossfade', start: next + F, duration: c.duration_sec * 1000, curve: c.color ? 'linear' : c.curve || 'linear', heldAt: null});
        this.takes.push({start: this.start, duration: this.duration, curve: this.curve});
      }
      if (c.command !== 'lever' || (this.mode !== 'idle' && !this.manual)) return;
      this.position = c.position;
      const progress = Math.abs(c.position - this.origin);
      if (progress >= 0.998) Object.assign(this, {origin: 1 - this.origin, mode: 'idle', manual: false});
      else if (progress > 0.002) Object.assign(this, {mode: 'crossfade', manual: true, duration: c.duration_sec * 1000, start: t});
      else if (this.manual) Object.assign(this, {mode: 'idle', manual: false});
    }};
  const delay = () => 10 + random() * 250;
  let cache = null, served = null, changedAt = 0, lastSeq = -1, wsAt = 0;
  const streamed = [];   // the lever's positions that came over its socket, in the order they came
  const lever = (c, t) => { if (c.seq > lastSeq) { lastSeq = c.seq; mixer.command(c, t); changedAt = t; } };
  // A 120 Hz display: animation frames on its grid, the frame's time to each callback; __onFrame runs after them.
  const frameClock = () => {
    const period = 1000 / 120, queue = [], grid = (t) => Math.max(last + period, Math.round(t / period) * period);
    let timer = null, last = -Infinity;
    window.__asked = 0;
    const tick = () => {
      timer = null;
      const at = performance.now(), ts = last = grid(at);   // a timer a little early or late still hits its frame
      queue.splice(0).forEach((cb) => cb(ts));
      window.__onFrame?.(ts, at);
      schedule();
    };
    const schedule = () => {
      if (timer === null && (queue.length || window.__onFrame)) timer = setTimeout(tick, grid(performance.now() + period / 2) - performance.now());
    };
    window.requestAnimationFrame = (cb) => { queue.push(cb); schedule(); return ++window.__asked; };
    window.__startFrames = schedule;
  };
  const show = bigShow(), {context, page, errors} = await openPage(browser, {...show, init: frameClock});
  await page.route('**/api/state', async (route) => {
    await sleep(delay());
    const t = now(), fresh = cache && t - cache.at < 200 && cache.at > changedAt ? cache : {at: t, status: mixer.status(t)};
    const reply = served && random() < 0.15 ? served : fresh;
    cache = served = fresh;
    await sleep(delay());
    await route.fulfill({json: {...show.state, status: {...show.state.status, ...reply.status}}});
  });
  await page.route('**/api/command', async (route) => {
    const c = route.request().postDataJSON();
    await sleep(delay());
    if (c.command === 'lever') lever(c, now()); else { mixer.command(c, now()); changedAt = now(); }
    await sleep(delay());
    await route.fulfill({json: {ok: true}});
  });
  await page.routeWebSocket(/\/api\/lever$/, (ws) => ws.onMessage((m) => {
    streamed.push(JSON.parse(m));
    wsAt = Math.max(wsAt, now() + 10 + random() * 120);   // in order, as TCP keeps them
    setTimeout(() => lever({...JSON.parse(m), command: 'lever'}, now()), wsAt - now());   // as web.py hands it to the bridge
  }));
  await page.reload();
  await page.waitForFunction(() => state?.status && $('scenes').children.length);
  // The page has been open a while: its clock has seen many polls.
  await page.evaluate(async () => { for (let i = 0; i < 25; i++) await refresh(); });
  // The page's clock in this process's: both count from their own time origin.
  const offset = await page.evaluate(() => performance.timeOrigin) - performance.timeOrigin;
  await page.evaluate(() => {
    touched.fade = true;
    window.frames_ = [];
    const q = (s) => document.querySelector(s);
    window.__onFrame = (t, at) => {
      const trail = q('#tp .trail').style;
      frames_.push({t, at, p: trail.getPropertyValue('--b') - trail.getPropertyValue('--a'),
        v: Number(q('#tp .track').style.getPropertyValue('--v')), lever: $('lever').value / 1000, asked: __asked,
        settled: mix ? mix.fix.at + mix.fix.ms : 0});
    };
    __startFrames();
  });
  const at = () => page.evaluate(() => performance.now());
  const press = async (key) => { const t = await at(); await page.keyboard.press(key); return t; };
  const rate = (text) => page.evaluate((text) => { $('fade-s').value = text; }, text);
  const curve = (name) => page.locator(`#curves [data-value="${name}"]`).click();
  const recorded = (from, to) => page.evaluate(([from, to]) => frames_.filter((f) => f.t >= from && f.t <= to), [from, to]);
  const rising = (frames, what) => frames.forEach((f, i) => i && assert(f.p >= frames[i - 1].p - 0.001,
    `${what}: the T-bar steps back from ${frames[i - 1].p} to ${f.p} at ${Math.round(f.t)} ms`));
  // The frames of a take: up to the one that clears it.
  const take = (frames) => { const end = frames.findIndex((f, i) => i && f.p < frames[i - 1].p - 0.5); return end < 0 ? frames : frames.slice(0, end); };
  // At most the curve's steepest rate plus the catch-up, a frame at a time.
  const steepest = {linear: 1, 'ease-in': 2, 'ease-out': 2, 'ease-in-out': 1.5};
  const smooth = (frames, duration, name, what) => frames.forEach((f, i) => i && assert(f.p - frames[i - 1].p <= (f.t - frames[i - 1].t) * (1.5 * steepest[name] / duration + 0.001) + 0.003,
    `${what}: the T-bar jumps from ${frames[i - 1].p} to ${f.p} at ${Math.round(f.t)} ms`));
  const measured = [];
  // The T-bar against the mix, once the page's last correction of the take has eased in: every frame's lag behind the
  // mixer, the time between the frame and the mixer's moment of that progress (through the mixer's curve), is the same
  // to within 1 ms, so the T-bar has the mix's curve and rate, and it is under 200 ms (the network's least delay and a
  // frame's grain). Measured over the whole take: the largest change of speed from one frame to the next, in the mix's
  // own speed (1 / duration).
  const shaped = (frames, name, duration, what) => {
    const truth = mixer.takes.at(-1), [c, inverse] = CURVES[truth.curve], whole = frames.filter((f) => f.p > 0.02 && f.p < 0.98);
    assert.equal(truth.curve, name, `${what}: the mixer runs ${name}`);
    const mid = whole.filter((f) => f.t >= Math.max(...frames.map((g) => g.settled))), n = mid.length;
    const lags = mid.map((f) => f.t + offset - truth.start - inverse(f.p) * truth.duration), lag = lags.reduce((s, x) => s + x, 0) / n;
    const shape = Math.max(...mid.map((f) => Math.abs(f.p - c((f.t + offset - lag - truth.start) / truth.duration))));
    let jerk = 0;
    for (let i = 2; i < whole.length; i++) {
      const v = (a, b) => (b.p - a.p) / (b.t - a.t) * duration;
      jerk = Math.max(jerk, Math.abs(v(whole[i - 1], whole[i]) - v(whole[i - 2], whole[i - 1])));
    }
    const gaps = frames.slice(1).map((f, i) => f.at - frames[i].at).sort((a, b) => a - b), spread = Math.max(...lags) - Math.min(...lags);
    measured.push({what, frames: whole.length, settled: n, gapMs: gaps[gaps.length >> 1], lag, spread, shape, jerk});
    assert(gaps[gaps.length >> 1] < 9.5, `${what}: drawn at 120 Hz (median frame ${gaps[gaps.length >> 1]} ms)`);
    assert(n >= 10, `${what}: settled for ten frames at least (${n} of ${whole.length})`);
    assert(spread < 1 && lag > -F && lag < 200, `${what}: runs the mix's curve ${Math.round(lag)} ms behind it, to within ${spread} ms`);
  };
  // An Auto: from 0 to 100% along its curve, never back, and the take ends.
  const auto = async (key, name, duration, what) => {
    const t0 = await press(key);
    await sleep(duration + 800);
    const frames = await recorded(t0, await at()), run = take(frames);
    rising(run, what);
    smooth(run, duration, name, what);
    shaped(run, name, duration, what);
    assert(run.at(-1).p === 1 && frames.at(-1).p === 0, `${what} reaches 100% and then the take ends`);
  };
  await rate('1:00');
  await auto('Space', 'linear', 1000, 'linear Auto');
  for (const name of ['ease-in', 'ease-out', 'ease-in-out']) {
    await curve(name);
    await auto('Space', name, 1000, `${name} Auto`);
  }
  // A dip runs its clock, whichever curve its halves have; so does a wipe, its clip's.
  await page.locator('#dip').click();
  await auto('Space', 'linear', 1000, 'ease-in-out dip');
  await page.keyboard.press('w');
  await auto('Space', 'linear', WIPE_MS, 'wipe');
  await page.keyboard.press('f');
  // Held, an eased Auto and a dip stand still; Auto again runs them on to the end.
  await rate('2:00');
  for (const what of ['held and resumed Auto', 'held and resumed dip']) {
    if (what.endsWith('dip')) await page.locator('#dip').click();
    const t0 = await press('Space');
    await sleep(700);
    await press('Space');
    await sleep(2500);
    const resume = await press('Space');
    await sleep(2000);
    const frames = await recorded(t0, await at()), eased = take(frames);
    rising(eased, what);
    smooth(eased, 2000, what.endsWith('dip') ? 'linear' : 'ease-in-out', what);
    // Held where the mixer held it, once its reply is in (the network's delay after the press, up to four times its
    // one way, and a poll in flight) and eased in.
    const held = eased.filter((f) => f.t > resume - 500 && f.t < resume);
    assert(held.at(-1).p - held[0].p < 0.002, `${what}: held, the T-bar stands still: ${held[0].p} to ${held.at(-1).p}`);
    assert(eased.at(-1).p === 1 && eased.length < frames.length, `${what}: resumed, it runs to the end`);
  }
  await page.locator('#fade').click();
  // Cut during an Auto clears the T-bar at the press, and no late reply brings it back; the next Auto starts from 0.
  await curve('linear');
  await press('Space');
  await sleep(600);
  const cut = await press('c');
  await sleep(1500);
  let frames = await recorded(cut, await at());
  const cleared = frames.findIndex((f) => f.p === 0);
  assert(frames[cleared].t - cut < 100 && frames.slice(cleared).every((f) => f.p === 0), 'a cut clears the T-bar at once and for good');
  let t0 = await press('Space');
  await sleep(2500);
  frames = await recorded(t0, await at());
  assert.equal(frames[0].p, 0, 'the next Auto starts from 0');
  rising(take(frames), 'the next Auto');
  // Dragged, the grip stays under the pointer; released halfway, the mix stays where it was let go through every poll.
  await page.waitForFunction(() => !$('tp').classList.contains('owned'));
  const box = await page.locator('#tp .track').boundingBox(), y = box.y + box.height / 2, x = (v) => box.x + 18 + v * (box.width - 36);
  await page.mouse.move(x(0), y);
  await page.mouse.down();
  t0 = await at();
  for (let i = 1; i <= 30; i++) { await page.mouse.move(x(0.6 * i / 30), y); await sleep(16); }
  const release = await at();
  await page.mouse.up();
  await sleep(2500);
  frames = await recorded(t0, release);
  frames.forEach((f, i) => assert(Math.abs(f.v - f.lever) <= 0.002 && (!i || f.v >= frames[i - 1].v), `dragged, the grip follows the pointer at ${Math.round(f.t)} ms`));
  frames = await recorded(release + 50, await at());
  assert(frames.every((f) => f.lever === frames[0].lever && Math.abs(f.p - f.lever) < 0.002 && Math.abs(f.v - f.lever) < 0.002) && frames[0].lever > 0.5,
    'released, the lever\'s mix and grip stay put');
  // The drag's positions streamed over the lever's socket, numbered in order, up to where it was let go.
  assert(streamed.length >= 20 && streamed.every((m, i) => !i || m.seq > streamed[i - 1].seq) && streamed.at(-1).position === frames[0].lever,
    `the lever streams its positions in order (${streamed.length})`);
  // Standing still, the page asks for no animation frame.
  frames = await recorded(release + 1500, await at());
  assert.equal(frames.at(-1).asked, frames[0].asked, 'a T-bar standing still asks for no frames');
  assert.deepEqual(errors, []);
  for (const m of measured)
    console.log(`  ${m.what}: ${m.frames} frames at ${m.gapMs.toFixed(1)} ms; settled ${m.settled}: ${m.lag.toFixed(1)} ms behind the mix to within ` +
                `${m.spread.toFixed(3)} ms, off its curve by ${m.shape.toExponential(1)}; speed change per frame up to ${m.jerk.toFixed(3)}`);
  await context.close();
}
