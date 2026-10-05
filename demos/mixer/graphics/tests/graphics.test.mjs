// The graphics that ship beside the engine, each run as its page runs it: linked as graphic_pages.py
// links it (the engine, the graphic, the shell's mount line) and given the data-* attributes
// prepare_demo.py gives it. The tests in the first loop cover every directory that holds a
// graphic.js, so a new graphic is checked without a line added here.
import assert from 'node:assert/strict';
import { existsSync, readdirSync } from 'node:fs';
import { test } from 'node:test';
import { RATES, elapse, page, read, settle, shellScripts } from './page.mjs';

const ROOT = new URL('..', import.meta.url);
const NAMES = readdirSync(ROOT, { withFileTypes: true })
  .filter((entry) => entry.isDirectory() && existsSync(new URL(`${entry.name}/graphic.js`, ROOT))).map((entry) => entry.name);
const manifest = (name) => JSON.parse(read(`${name}/${name}.ograf.json`));
const HOUR = 3_600_000;   // a whole number of every demo cycle here: a page opened then starts a cycle
// Every element measures like the lower third's plate and holds two copies of a crawl's content.
const LAYOUT = new Proxy({}, { get: () => ({ width: 1016, height: 131, copies: 2, content: 3999.2 }) });

/** The page of a shipped graphic, in a window of the size its manifest asks for; its demo runs. */
async function open(name, { fps = 50, source = `dsk_${name}`, now = HOUR } = {}) {
  const { width, height } = manifest(name).v_avplumber?.window ?? { width: 1920, height: 1080 };
  const fake = page({ graphic: read(`${name}/graphic.js`), boot: shellScripts()[2], dataset: { fps: String(fps), source },
    layout: LAYOUT, now, width, height });
  await settle();
  return { ...fake, element: fake.document.body.children[0], Graphic: fake.registry.get('motion-graphic') };
}
const animations = (element) => Object.values(element.parts).flatMap((part) => part.animations);
const running = (element) => Object.values(element.parts).flatMap((part) => part.running);
// A keyframe of a move is shown for one frame; one equal to the frame before it repaints nothing.
const shown = ({ transform, opacity }) => `${transform} ${opacity}`;
const painted = ({ keyframes }) => keyframes.slice(1).filter((keyframe, k) => shown(keyframe) !== shown(keyframes[k])).length;
const isWhole = (value) => Math.abs(value - Math.round(value)) < 1e-6;
const timers = (clock) => clock.pending.map(Math.round);

test('the demo\'s test source and its four keys are graphics', () => {
  for (const name of ['browser_alpha', 'lower_third', 'ticker', 'bug_left', 'bug_right']) assert.ok(NAMES.includes(name), name);
});

for (const name of NAMES) {
  test(`${name}: declared through the engine alone`, () => {
    const source = read(`${name}/graphic.js`);
    assert.match(source, /^const graphic = Motion\.graphic\(\{$/m, 'the shell mounts the constant named graphic');
    assert.doesNotMatch(source, /requestAnimationFrame|setInterval|setTimeout|\.animate\(|@keyframes|\banimation\s*:|\btransition\s*:/,
      'motion goes through play, stop, actions and loop; timed changes through every');
    assert.doesNotMatch(source, /^\s*(import|export)\b|\b(fetch|WebSocket|XMLHttpRequest)\s*\(|\b(src|href)=|url\((?!["']?data:)/m,
      'a graphic loads nothing at run time');
  });

  test(`${name}: its OGraf manifest names what it declares`, async () => {
    const { main, customActions = [] } = manifest(name);
    assert.equal(main, 'graphic.js');
    const { element, Graphic } = await open(name);
    const ids = customActions.map(({ id }) => id);
    for (const { action: { type, params } } of Graphic.demo?.schedule ?? []) {
      if (type === 'customAction') assert.ok(ids.includes(params.id), `customActions lacks "${params.id}", which the demo runs`);
    }
    for (const id of ids) assert.equal((await element.customAction({ id, skipAnimation: true })).statusCode, 200, `the graphic has no action "${id}"`);
  });

  test(`${name}: every frame is on whole pixels and whole ticks at 25, 30, 50 and 60 fps`, async () => {
    for (const fps of RATES) {
      const opened = await open(name, { fps });
      await elapse(opened, opened.Graphic.demo?.cycle ?? 10_000);
      const tick = opened.Motion.tickUs(fps) / 1000;
      for (const { keyframes, timing } of animations(opened.element)) {
        for (const { transform = '' } of keyframes) assert.doesNotMatch(transform, /translate\([^)]*\./, `${fps} fps: ${transform}`);
        assert.ok(isWhole(timing.duration / tick), `${fps} fps: ${timing.duration} ms is not whole ticks`);
        assert.ok(isWhole(timing.delay / tick + 0.5), `${fps} fps: every sample falls mid-step`);
        if (timing.iterations === Infinity) {
          // One equal step per tick: steps(N) over N ticks, and a crawl's N steps are whole pixels each.
          assert.equal(timing.easing, `steps(${Math.round(timing.duration / tick)})`);
          const [, wrap = 0] = /translate\((-?\d+)px/.exec(keyframes[1].transform) ?? [];
          assert.ok(Number.isInteger(wrap * tick / timing.duration), `${fps} fps: ${wrap} px in ${timing.easing}`);
        } else {
          assert.equal(keyframes.length, Math.round(timing.duration / tick) + 1, 'one keyframe per frame');
        }
      }
    }
  });

  test(`${name}: played and stopped by a controller, it leaves nothing running`, async () => {
    const fake = page({ graphic: read(`${name}/graphic.js`), layout: LAYOUT });
    const element = new fake.graphic();
    await element.load({ renderCharacteristics: { frameRate: 50 } });
    assert.deepEqual([running(element), fake.clock.pending], [[], []], 'loaded');
    element.playAction({});
    await elapse({ ...fake, element }, 10_000);
    element.stopAction({});
    await elapse({ ...fake, element }, 10_000);
    assert.deepEqual([running(element), fake.clock.pending], [[], []], 'no animation and no timer');
  });
}

test('browser_alpha: the marker slides 2 s and rests 4 s, there and back in 12 s', async () => {
  for (const fps of RATES) {
    const opened = await open('browser_alpha', { fps, source: 'browser_000' });
    const { marker, title } = opened.element.parts;
    assert.equal(title.textContent, 'browser_000 · alpha over video');
    await elapse(opened, 1000);
    assert.equal(marker.running.length, 1, 'sliding right');
    await elapse(opened, 1000);
    assert.deepEqual([marker.running, marker.style.transform, timers(opened.clock)], [[], 'translate(1574px, 0px)', [4000]],
      'at rest on the right: one timer to the next cue, nothing else');
    await elapse(opened, 6000);
    assert.deepEqual([marker.running, marker.style.transform, timers(opened.clock)], [[], 'translate(0px, 0px)', [4000]]);
    await elapse(opened, 3999);
    assert.deepEqual(marker.animations.map(painted), [2 * fps, 2 * fps], `${4 * fps} painted frames in a cycle, a third of its ticks`);
    assert.deepEqual(animations(opened.element), marker.animations, 'nothing but the marker moves');
  }
});

test('browser_alpha: a page without a source id still titles itself', async () => {
  const fake = page({ graphic: read('browser_alpha/graphic.js'), boot: shellScripts()[2], dataset: { fps: '60' } });
  await settle();
  assert.equal(fake.document.body.children[0].parts.title.textContent, 'Browser alpha over video');
});

test('browser_alpha: 36 sources spread over the 6 s between slides, so 11 to 13 slide at once', async () => {
  const { Motion, Graphic } = await open('browser_alpha');
  assert.deepEqual([Graphic.demo.cycle, Graphic.demo.stagger], [12000, 6000]);
  const starts = Array.from({ length: 36 }, (_, i) => Motion.phase(`browser_${String(i).padStart(3, '0')}`) * 6);
  const sliding = (t) => starts.filter((start) => ((t - start) % 6 + 6) % 6 < 2).length;
  const counts = Array.from({ length: 600 }, (_, i) => sliding(i / 100));
  assert.deepEqual([Math.min(...counts), Math.max(...counts)], [11, 13]);
  // The wall clock places a source, not its load time: browser_007 starts 0.326 of 6 s into the cycle.
  for (const late of [0, 700, 1900]) {
    const { clock } = await open('browser_alpha', { source: 'browser_007', now: 5 * HOUR + late });
    assert.ok(Math.abs(clock.pending[0] - (Motion.phase('browser_007') * 6000 - late)) < 1e-6);
  }
});

test('lower_third: four names in turn, each on for 5.2 s of a 6 s slot', async () => {
  // Frames that repaint, coming in and going out: the eased plate reaches its place a frame or two
  // before the move ends at 30, 50 and 60 fps.
  const PAINTED = { 25: [15, 10], 30: [17, 12], 50: [29, 20], 60: [34, 24] };
  for (const fps of RATES) {
    const opened = await open('lower_third', { fps });
    const { plate, name, role } = opened.element.parts, people = [];
    for (let slot = 0; slot < 4; slot++) {
      await elapse(opened, 3000);
      people.push([name.textContent, role.textContent]);
      assert.deepEqual([plate.running, plate.style.transform, plate.style.opacity, timers(opened.clock)],
        [[], 'translate(0px, 0px)', '1', [2200]], 'on air and at rest: one timer to the stop cue');
      await elapse(opened, 2999);
      assert.deepEqual([plate.running, plate.style.transform, plate.style.opacity, timers(opened.clock)],
        [[], 'translate(-1057px, 0px)', '0', [1]], 'off and at rest: one timer to the next name');
      if (slot < 3) await elapse(opened, 1);
    }
    assert.deepEqual(people, [['Ada Lovelace', 'Analyst · Engine No. 1'], ['Grace Hopper', 'Compiler desk'],
      ['Alan Turing', 'Codebreaking correspondent'], ['Hedy Lamarr', 'Spread spectrum, live']]);
    assert.deepEqual(plate.animations.map(({ keyframes }) => keyframes.length - 1),
      Array(4).fill([Math.round(0.6 * fps), Math.round(0.4 * fps)]).flat(), '0.6 s in, 0.4 s out');
    assert.deepEqual(plate.animations.map(painted), Array(4).fill(PAINTED[fps]).flat());
    assert.deepEqual(animations(opened.element), plate.animations, 'nothing but the plate moves');
  }
});

test('ticker: the crawl steps the same whole pixels on every frame, played by the compositor alone', async () => {
  const STEP = { 25: 4, 30: 4, 50: 2, 60: 2 };   // 110 px/s
  for (const fps of RATES) {
    const opened = await open('ticker', { fps });
    await elapse(opened, 60_000);
    const { crawl } = opened.element.parts;
    const [loop] = crawl.animations;
    assert.deepEqual(animations(opened.element), [loop], 'one animation, started once');
    assert.deepEqual(loop.keyframes, [{ transform: 'translate(0px, 0px)' }, { transform: 'translate(-4000px, 0px)' }]);
    assert.deepEqual([loop.timing.easing, loop.timing.iterations], [`steps(${4000 / STEP[fps]})`, Infinity], 'every tick paints one step');
    const [first, second] = crawl.children;
    assert.match(first.textContent, /^Downstream keyer on air: clean feed stays unkeyed {3}• {3}Four browser keys, one GPU pass {3}• /);
    assert.deepEqual([second.textContent, first.style.width, second.style.width], [first.textContent, '4000px', '4000px'],
      'two copies, each as wide as the loop');
    assert.deepEqual(opened.clock.pending, [], 'no timer');
  }
});

test('bug_left: the ring turns once in 4 s, one step per frame', async () => {
  for (const fps of RATES) {
    const opened = await open('bug_left', { fps });
    await elapse(opened, 60_000);
    const [loop] = opened.element.parts.ring.animations;
    assert.deepEqual(animations(opened.element), [loop], 'one animation, started once');
    assert.deepEqual(loop.keyframes, [{ transform: 'rotate(0deg)' }, { transform: 'rotate(360deg)' }]);
    assert.deepEqual([loop.timing.easing, loop.timing.iterations], [`steps(${4 * fps})`, Infinity]);
    assert.deepEqual(opened.clock.pending, [], 'no timer');
  }
});

test('bug_right: one timer changes dot and clock together twice a second; nothing animates', async () => {
  const time = (ms) => new Date(ms).toTimeString().slice(0, 8);
  for (const fps of RATES) {
    const opened = await open('bug_right', { fps, now: HOUR + 123 });
    const { dot, clock } = opened.element.parts, states = [];
    for (const wait of [0, 376, 1, 499, 1, 499, 1, 499, 1]) {
      await elapse(opened, wait);
      states.push([dot.style.opacity, clock.textContent]);
      assert.equal(opened.clock.pending.length, 1, 'one timer, aimed at the next half second');
    }
    // Bright on the second, dim on the half; the text changes with the dot, so each is one paint.
    assert.deepEqual(states, [['1', time(HOUR)], ['1', time(HOUR)], ['0.25', time(HOUR)], ['0.25', time(HOUR)],
      ['1', time(HOUR + 1000)], ['1', time(HOUR + 1000)], ['0.25', time(HOUR + 1000)], ['0.25', time(HOUR + 1000)], ['1', time(HOUR + 2000)]]);
    assert.deepEqual(animations(opened.element), []);
  }
});
