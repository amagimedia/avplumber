// The self-running page: a graphic's demo schedule, the wall-clock rule that places every instance
// in its cycle, the per-source stagger, and the shell (host.html) that mounts the graphic.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { ALPHA, LOWER_THIRD, LOWER_THIRD_LAYOUT, TICKER, TICKER_LAYOUT, page, read, settle } from './page.mjs';

const ACTIONS = ['playAction', 'stopAction', 'updateAction', 'customAction'];
/** An element that only records the actions the demo runner calls on it, with the clock time. */
function recorder(clock) {
  const calls = [];
  const element = Object.fromEntries(ACTIONS.map((type) => [type, async (params) => { calls.push([clock.now % 12000, type, params]); }]));
  return { element, calls };
}
const custom = (id, skipAnimation) => ['customAction', { id, payload: undefined, skipAnimation }];
const HOUR = 3_600_000;   // a whole number of 12 s cycles

test('a demo is normalised to an OGraf actions schedule and a cycle length', () => {
  const { Motion } = page();
  assert.deepEqual(Motion.graphic(LOWER_THIRD).demo, {
    schedule: [
      { timestamp: 0, action: { type: 'updateAction', params: { data: { name: 'Ada Lovelace', role: 'Analyst · Engine No. 1' } } } },
      { timestamp: 0, action: { type: 'playAction', params: { goto: 0 } } },
      { timestamp: 5200, action: { type: 'stopAction', params: {} } },
      { timestamp: 6000, action: { type: 'updateAction', params: { data: { name: 'Grace Hopper', role: 'Compiler desk' } } } },
      { timestamp: 6000, action: { type: 'playAction', params: { goto: 0 } } },
      { timestamp: 11200, action: { type: 'stopAction', params: {} } },
    ],
    cycle: 12000,
  });
  assert.deepEqual(Motion.graphic(ALPHA).demo, {
    schedule: [{ timestamp: 0, action: { type: 'customAction', params: { id: 'right', payload: undefined } } },
      { timestamp: 6000, action: { type: 'customAction', params: { id: 'left', payload: undefined } } }],
    cycle: 12000,
  });
  assert.deepEqual(Motion.graphic(TICKER).demo, { schedule: [{ timestamp: 0, action: { type: 'playAction', params: { goto: 0 } } }], cycle: null });
  assert.equal(Motion.graphic({ html: '<p id="a"></p>' }).demo, null);
});

test('a wrong demo names the cue', () => {
  const { Motion } = page();
  const schedule = (cues) => () => Motion.schedule(cues, ['right']);
  assert.throws(schedule('play'), /Motion: demo: must be a list of cues \[seconds, action, data\]/);
  assert.throws(schedule([[0, 'play'], ['1', 'stop']]), /Motion: demo\[1\]: expected \[seconds, action, data\]; got \["1","stop"\]/);
  assert.throws(schedule([[2, 'play'], [1, 'stop']]), /Motion: demo\[1\]: cues must be in time order \(1 s comes after 2 s\)/);
  assert.throws(schedule([[0, 'play'], [0, 'lfet']]), /Motion: demo\[1\]: unknown action "lfet" \(known: play, stop, update, repeat, right\)/);
  assert.throws(schedule([[0, 'play'], [6, 'repeat'], [7, 'stop']]), /Motion: demo\[1\]: repeat must be the last cue/);
  assert.throws(schedule([[0, 'repeat']]), /Motion: demo\[0\]: repeat needs a cycle longer than 0 s/);
});

test('the stagger is the golden-ratio fraction of the trailing number of the source id', () => {
  const { Motion } = page();
  const frac = (index) => index * 0.6180339887 % 1;
  assert.equal(Motion.phase(undefined), 0);
  assert.equal(Motion.phase('dsk_lower_third'), 0);
  assert.equal(Motion.phase('browser_000'), 0);
  assert.equal(Motion.phase('browser_007'), frac(7));
  assert.equal(Motion.phase('browser_036'), frac(36));
  // 36 sources never bunch: each 2 s slide of a 6 s half cycle overlaps about a third of the others.
  const starts = Array.from({ length: 36 }, (_, i) => Motion.phase(`browser_${i + 1}`) * 12 % 6);
  const sliding = (t) => starts.filter((start) => ((t - start) % 6 + 6) % 6 < 2).length;
  const counts = Array.from({ length: 600 }, (_, i) => sliding(i / 100));
  assert.ok(Math.min(...counts) >= 10 && Math.max(...counts) <= 14, `${Math.min(...counts)}..${Math.max(...counts)} of 36 slide at once`);
});

test('cues fire in order on the wall clock, one timer ahead, and the cycle repeats', async () => {
  const { Motion, clock } = page({ now: 5 * HOUR });
  const { element, calls } = recorder(clock);
  Motion.demo(element, Motion.graphic(ALPHA).demo);
  assert.deepEqual([calls, clock.pending], [[], [0]], 'at the start of a cycle nothing is behind');
  await clock.advance(0);
  assert.deepEqual(calls, [[0, ...custom('right', false)]]);
  assert.deepEqual(clock.pending, [6000]);
  await clock.advance(6000);
  assert.deepEqual(clock.pending, [6000]);
  await clock.advance(13000);
  assert.deepEqual(calls, [[0, ...custom('right', false)], [6000, ...custom('left', false)],
    [0, ...custom('right', false)], [6000, ...custom('left', false)]]);
  assert.deepEqual(clock.pending, [5000]);
});

test('a page loaded mid-cycle catches up without animation, whatever its load time', async () => {
  for (const [late, behind, next] of [[1, ['right'], 5999], [5999, ['right'], 1], [6000, ['right'], 0], [6001, ['right', 'left'], 5999], [11999, ['right', 'left'], 1]]) {
    const { Motion, clock } = page({ now: 7 * HOUR + late });
    const { element, calls } = recorder(clock);
    Motion.demo(element, Motion.graphic(ALPHA).demo);
    assert.deepEqual(calls, behind.map((id) => [late, ...custom(id, true)]), `loaded ${late} ms into the cycle`);
    assert.deepEqual(clock.pending, [next]);
  }
});

test('cues due together fire in one task, in declared order', async () => {
  const { Motion, clock } = page({ now: 2 * HOUR + 5900 });
  const { element, calls } = recorder(clock);
  Motion.demo(element, Motion.graphic(LOWER_THIRD).demo);
  assert.deepEqual(calls.map(([, type, params]) => [type, params.skipAnimation]),
    [['updateAction', true], ['playAction', true], ['stopAction', true]]);
  calls.length = 0;
  await clock.advance(100);
  assert.deepEqual(calls, [
    [6000, 'updateAction', { data: { name: 'Grace Hopper', role: 'Compiler desk' }, skipAnimation: false }],
    [6000, 'playAction', { goto: 0, skipAnimation: false }],
  ]);
  assert.deepEqual(clock.pending, [5200]);
});

test('the stagger shifts the cycle by a fraction of itself, the same for every load time', async () => {
  for (const loadedAt of [0, 1234, 9999]) {
    const { Motion, clock } = page({ now: 3 * HOUR + loadedAt });
    const { element, calls } = recorder(clock);
    Motion.demo(element, Motion.graphic(ALPHA).demo, 0.25);
    calls.length = 0;
    await clock.advance(24000);
    assert.deepEqual(calls.map(([at, , { id, skipAnimation }]) => [at, id, skipAnimation]),
      [[3000, 'right', false], [9000, 'left', false], [3000, 'right', false], [9000, 'left', false]], `loaded at ${loadedAt} ms`);
  }
});

test('a demo without repeat runs once from load and leaves no timer', async () => {
  const { Motion, clock } = page({ now: 9 * HOUR + 4321 });
  const { element, calls } = recorder(clock);
  const stop = Motion.demo(element, Motion.schedule([[0, 'play'], [1.5, 'stop']]));
  assert.deepEqual(clock.pending, [0]);
  await clock.advance(1500);
  assert.deepEqual(calls.map(([, type, params]) => [type, params]),
    [['playAction', { goto: 0, skipAnimation: false }], ['stopAction', { skipAnimation: false }]]);
  assert.deepEqual(clock.pending, []);
  assert.equal(typeof stop, 'function');
});

test('the returned function stops the demo', async () => {
  const { Motion, clock } = page({ now: HOUR });
  const { element, calls } = recorder(clock);
  Motion.demo(element, Motion.graphic(ALPHA).demo)();
  assert.deepEqual(clock.pending, []);
  await clock.advance(30000);
  assert.deepEqual(calls, []);
});

test('host mounts the graphic, loads it with the page\'s data-* attributes and runs its demo', async () => {
  const { Motion, clock, document, registry } = page({ now: 4 * HOUR, dataset: { fps: '60', source: 'browser_000' } });
  const Graphic = Motion.graphic(ALPHA);
  const element = await Motion.host(Graphic);
  assert.deepEqual([...registry.values()], [Graphic]);
  assert.deepEqual(document.body.children, [element]);
  assert.equal(element.parts.title.textContent, 'browser_000 · alpha over video', 'fps is the frame rate, the rest is data');
  assert.deepEqual(clock.pending, [0]);
  await clock.advance(0);
  const [animation] = element.parts.marker.animations;
  assert.equal(animation.keyframes.length, 121, '2 s at the 60 fps of data-fps');
  animation.finish();
  await settle();
  assert.deepEqual(element.parts.marker.running, []);
  assert.deepEqual(clock.pending, [6000], 'at rest: one timer to the next cue, nothing else');
});

test('host places each source in the cycle by its id, not by its load time', async () => {
  const { Motion, clock } = page({ now: 4 * HOUR + 100, dataset: { fps: '50', source: 'browser_007' } });
  const element = await Motion.host(Motion.graphic(ALPHA));
  const shift = Math.round(Motion.phase('browser_007') * 12000 * 1000) / 1000;
  assert.ok(shift > 3900 && shift < 4000, 'browser_007 is 0.326 of a cycle in');
  assert.deepEqual(element.parts.marker.animations, [], 'the cycle has not started for this source');
  assert.equal(element.parts.marker.style.transform, 'translate(0px, 0px)', 'left, where the previous cycle ended');
  assert.equal(Math.round(clock.pending[0] * 1000) / 1000, shift - 100);
});

test('host refuses a page without data-fps', async () => {
  const { Motion } = page({ dataset: { source: 'browser_001' } });
  await assert.rejects(Motion.host(Motion.graphic(ALPHA)), /Motion: load\(\) needs renderCharacteristics\.frameRate \(the page shell passes data-fps\)/);
});

test('host.html is the whole page: transparent, self-contained, engine then graphic then mount', async () => {
  const shell = read('host.html');
  assert.equal(shell.split('/*@motion.js*/').length, 2);
  assert.equal(shell.split('/*@graphic.js*/').length, 2);
  assert.equal(shell.split('<html').length, 2, 'graphic_pages.py adds the data-* attributes to the one <html');
  assert.match(shell, /html, body \{[^}]*background: transparent/);
  assert.doesNotMatch(shell, /\b(src|href)=|@import|url\(/, 'nothing is fetched at run time');
  const scripts = Array.from(shell.matchAll(/<script>([\s\S]*?)<\/script>/g), ([, text]) => text);
  assert.deepEqual(scripts.slice(0, 2), ['/*@motion.js*/', '/*@graphic.js*/']);
  assert.ok(shell.indexOf('<body>') < shell.indexOf('<script>'), 'the body exists before the shell mounts into it');

  const graphic = `const graphic = Motion.graphic({
    html: '<div id="plate"></div>',
    play: [{ el: 'plate', opacity: [0, 1], seconds: 0.4 }],
    demo: [[0, 'play']],
  });`;
  const { clock, document, errors } = page({ graphic, boot: scripts[2], dataset: { fps: '25' } });
  await settle();
  const [element] = document.body.children;
  assert.deepEqual(errors, []);
  assert.equal(element.parts.plate.style.opacity, '0');
  await clock.advance(0);
  assert.equal(element.parts.plate.animations[0].keyframes.length, 11);
});

test('the engine is one inlinable script: no module syntax, no frame loop, no network', () => {
  const engine = read('motion.js');
  assert.doesNotMatch(engine, /^\s*(import|export)\b/m, 'a classic inline script cannot hold module syntax');
  assert.doesNotMatch(engine, /\b(requestAnimationFrame|setInterval|fetch|WebSocket|XMLHttpRequest|importScripts)\s*\(/);
  assert.doesNotMatch(engine, /<\/script|<!--/i, 'either would end the inline script early');
});
