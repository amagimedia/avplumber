// The operator page (pyplumber/mixer/gui/assets/index.html) behind a mocked mixer, for the Playwright page checks.
// Every request is intercepted, so a check never controls a live mixer.
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');

const ASSETS = path.join(__dirname, '../../pyplumber/mixer/gui/assets');
const launch = () => chromium.launch(process.env.CHROMIUM ? {executablePath: process.env.CHROMIUM} : {});

// What the mixer does with a command, as far as the next /api/state shows it.
function applyCommand(state, command) {
  const key = (id) => state.dsk.find((k) => k.id === id);
  if (command.command === 'preview') state.status.pvw_scene = command.scene;
  else if (command.command === 'dsk' && command.prepare) state.dsk.forEach((k) => { k.next_take = command.keys.includes(k.id); });
  else if (command.command === 'dsk' && command.keys)
    state.dsk.filter((k) => command.keys.includes(k.id)).forEach((k) => { k.on = command.toggle ? !k.on : command.on; });
  else if (command.command === 'dsk' && 'all_rate' in command) state.dsk.forEach((k) => { k.all_rate = command.all_rate; });
  else if (command.command === 'dsk' && 'rate' in command) key(command.key).rate = command.rate;
  else if (command.command === 'dsk') key(command.key).on = command.on;
  else if (command.command === 'ftb') state.ftb = {on: !state.ftb.on, phase: state.ftb.on ? 'off' : 'on'};
}

// A stand-in for the player at /preview/: it posts its stream stats about twice a second, as the player does,
// [rtt, fps, buffer] by mountpoint.
const playerPage = (stats) => `<!doctype html><body style="margin:0;background:#050607"><script>
const STATS = ${JSON.stringify(stats)}, mp = new URLSearchParams(location.search).get('mountpoint');
const [rtt, fps, buf] = STATS[mp] || STATS.default;
const post = () => parent.postMessage({type: 'avp-preview-stats', search: location.search, rtt_ms: rtt, fps, buffer_ms: buf, status: ''}, '*');
post(); setInterval(post, 400);
</script>`;

// A page on the mocked mixer. `state` is served as /api/state and changed by the commands the page sends, which
// `commands` records; `config` fills the page's config placeholder; `storage` seeds the page's own localStorage
// keys (avplumber.mixer.<key>) where they are not set yet, so a reload keeps what the page stored since.
async function openPage(browser, {state, config = {}, viewport = {width: 1920, height: 1080}, storage = {}, init,
                                  stats = {default: [38, 30, 12]}}) {
  const context = await browser.newContext({viewport});
  const page = await context.newPage(), errors = [], commands = [], requests = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('request', (r) => requests.push(r.url()));
  await page.addInitScript((seed) => {
    for (const [key, value] of Object.entries(seed))
      if (localStorage.getItem(`avplumber.mixer.${key}`) === null) localStorage.setItem(`avplumber.mixer.${key}`, JSON.stringify(value));
  }, storage);
  if (init) await page.addInitScript(init);
  await page.route('**/*', (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === '/api/state') return route.fulfill({json: state});
    if (url.pathname === '/api/command') {
      const command = route.request().postDataJSON();
      commands.push(command);
      applyCommand(state, command);
      return route.fulfill({json: {ok: true}});
    }
    if (url.pathname === '/outputs.js')
      return route.fulfill({contentType: 'text/javascript', body: fs.readFileSync(path.join(ASSETS, 'outputs.js'), 'utf8')});
    if (url.pathname === '/') return route.fulfill({contentType: 'text/html', body: fs.readFileSync(path.join(ASSETS, 'index.html'), 'utf8')
      .replace('<script id="config" type="application/json">{}</script>',
        `<script id="config" type="application/json">${JSON.stringify({preview_base: '/preview/', ...config})}</script>`)});
    if (url.pathname === '/preview/') return route.fulfill({contentType: 'text/html', body: playerPage(stats)});
    return route.fulfill({contentType: 'text/html', body: ''});
  });
  await page.goto('http://mixer.test/');
  // The next answer to a command the page sends.
  const sent = () => page.waitForResponse((r) => r.url().endsWith('/api/command'));
  return {context, page, errors, commands, requests, sent};
}

// A show like the demo's: 256 scenes of five families, four keys, eight slots on the Program preview, a
// Multiviewer of 12 pages and 13 aux buses; meters with the figures the redesign's mock-up prints.
function bigShow() {
  const fps = 30, families = [['random', 48], ['grid', 96], ['fullscreen', 32], ['pip', 48], ['alpha_overlay', 32]];
  let n = 0;
  const scenes = families.flatMap(([family, count]) => Array.from({length: count}, () => `${family}_${String(++n).padStart(3, '0')}`));
  const slots = ['grid_100', 'alpha_overlay_247', 'grid_060', 'random_007', 'random_021', 'random_023', 'random_018', 'random_017'];
  const key = (id, rate, armed = false) => ({id, source: `dsk_${id}`, on: true, phase: 'on', next_take: armed, rate, all_rate: 0.4});
  const dsk = [key('lower_third', 0.5, true), key('ticker', null), key('bug_left', 0), key('bug_right', null)];
  const wipes = ['Diagonal sweep', 'Sliding panels'].map((name) => {
    const id = name.toLowerCase().replace(' ', '_');
    return {id, name, path: `/wipes/${id}.mov`, duration_seconds: null};
  });
  let mountpoint = 100;
  const rendition = () => ({codec: 'h264', color: 'sdr', mountpoint: mountpoint++, fps});
  const sourceCells = (first, count, cols) => Array.from({length: count}, (_, i) => ({role: 'source', id: `sdr420_${String(first + i).padStart(3, '0')}`,
    kind: 'nv12', x: (i % cols) * (1920 / cols), y: Math.floor(i / cols) * (1080 / cols), w: 1920 / cols, h: 1080 / cols}));
  const bus = (id, label, layout, cells, extra = {}) => ({id, label, layout, layouts: [], cells, canvas: {w: 1920, h: 1080}, fps, running: true,
    revision: `${id}-r1`, playout: {frames: 18000, missed_deadlines: 0}, output_drops: 0, ...extra});
  const slotCells = [{role: 'pgm', x: 0, y: 0, w: 960, h: 540}, {role: 'pvw', x: 960, y: 0, w: 960, h: 540},
    ...slots.map((_, i) => ({role: 'slot', slot: i, x: (i % 4) * 480, y: 540 + Math.floor(i / 4) * 270, w: 480, h: 270}))];
  const aux = [
    bus('pvw', 'Program preview', {preset: 'pgm_pvw_grid'}, slotCells,
        {scenes: [...slots], pvw_scene: 'alpha_overlay_230', layouts: [{preset: 'pgm_pvw_grid'}, {preset: 'source_pages'}]}),
    bus('mv', 'Multiviewer', {preset: 'source_pages'}, sourceCells(0, 16, 4), {page: 0, pages: 12, first: 1, total: 192}),
    ...Array.from({length: 13}, (_, i) => bus(`aux${i}`, `Aux ${i}`, {cells: []}, sourceCells(4 * i, 4, 2))),
  ];
  const program_outputs = [{rendition: 'program', ...rendition()}];
  const preview_outputs = [{bus: 'clean_program', label: 'Program clean', rendition: 'program_clean', ...rendition()},
    ...aux.map((a) => ({bus: a.id, label: a.label, layout: a.layout.preset || 'cells', rendition: `aux_${a.id}`, ...rendition()}))];
  const clips = wipes.map((w) => ({path: w.path, frames: 32, bytes: 32 * 1920 * 1080 * 1.5, complete: true}));
  const cuts = [74, 90, 106, 62, 58, 71, 66, 49, 44, 36, 38, 41].map((ms, i) => ({id: i + 1, ms, state: 'measured'}));
  const row = ([name, count, pct, max_pct]) => ({name, count, pct, avg_pct: Math.round(pct / count * 10) / 10, max_pct});
  const state = {
    setup_revision: 1, scenes, dsk, ftb: {on: false, phase: 'off'}, aux_buses: aux,
    status: {pgm_scene: 'grid_100', pvw_scene: 'alpha_overlay_230', transition: 'idle', lever: {origin: 0},
      playout: {A: {missed_deadlines: 0}, B: {missed_deadlines: 0}}, cut_latency: {direct: {...cuts.at(-1), recent: cuts}},
      wipe_cache: {bytes: clips.reduce((total, c) => total + c.bytes, 0), budget_bytes: 6e8, clips}},
    settings: {source_count: 192, source_counts: {video: 105, browser: 40, nv12: 47}, preview_codecs: ['h264'], program_outputs,
      canvas: {width: 1920, height: 1080, fps, working_format: 'nv12'}, direct: false, fade_seconds: 0.5, fade_curve: 'linear',
      fade_color: null, transition: 'fade', wipes, dsk_keys: dsk.map((k) => ({id: k.id, source: k.source})),
      aux_buses: aux.map((a) => a.id), preview_outputs,
      decode: {count: 105, fps: 3150, mpix_s: 6531.84, by_codec: [{codec: 'hevc', sources: 105, fps: 3150}]},
      encode_by_codec: [{codec: 'h264', sessions: 17, fps: 503}]},
    gpus: [{index: 0, gpu: 71, decoder: 66, encoder: 38, memory_used_mib: 18432, memory_total_mib: 24576, power_draw_w: 57,
      power_limit_w: 72, encoder_sessions: 17, encoder_fps: 503, encoder_mpix_s: 1043.0, name: 'NVIDIA L4', driver: '615.71.09',
      cuda: '13.4', compute_capability: '8.9', clocks_sm_mhz: 2040, clocks_mem_mhz: 6251, temperature_c: 76, pstate: 'P0',
      pcie_gen: 3, pcie_width: 16, persistence: true, compute_timeslice: 'Long'}],
    host: {load1: 5.5, vcpus: 12, cpu_pct: 47, mixer_cpu_pct: 237, thread_pct: 12, thread_name: 'mixer_comp_a',
      thread_kinds: [['decode_N', 106, 52, 0.8], ['mixer_comp_a', 1, 12, 12], ['other', 498, 53, 11]].map(row)},
    chromium: {cores_used: 2.68, vcpus: 12, processes: [['GPU process', 4, 156, 40]].map(row), gpu_threads: []},
    recent: {window_s: 600, covered_s: 600, missed_deadlines: 0, output_drops: 0,
      inputs: {browser: {sources: 36, repeats: 0, dropped: 0, worst: null}, other: {sources: 152, repeats: 0, dropped: 0}}},
  };
  const config = {setup: true, compute_price: {hourly_usd: 1.2}};
  // Viewer 1 plays Program dirty, viewer 2 the Program preview.
  const stats = {[program_outputs[0].mountpoint]: [38, 30, 12], default: [41, 30, 14]};
  return {state, config, stats, storage: {viewers: ['program', 'pvw']}};
}

module.exports = {launch, openPage, bigShow};
