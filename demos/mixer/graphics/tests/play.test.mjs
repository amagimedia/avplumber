// What the engine asks of the browser: baked step keyframes and whole-tick timing handed to
// element.animate, the end state committed to inline style, and nothing left running at rest.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { ALPHA, LOWER_THIRD, LOWER_THIRD_LAYOUT, RATES, TICKER, TICKER_LAYOUT, mount, page, settle } from './page.mjs';

const { Motion } = page();
const PLATE = { vw: 1016, vh: 172, width: 1016, height: 131 };
const HIDDEN = { transform: 'translate(-1057px, 0px)', opacity: '0', willChange: 'transform, opacity' };
const SHOWN = { transform: 'translate(0px, 0px)', opacity: '1', willChange: 'transform, opacity' };

test('a move is baked as one step keyframe per frame, played over whole ticks', () => {
  const timing = {
    25: { duration: 2000, delay: -20, fill: 'both' },
    30: { duration: 1999.98, delay: -16.6665, fill: 'both' },
    50: { duration: 2000, delay: -10, fill: 'both' },
    60: { duration: 1999.92, delay: -8.333, fill: 'both' },
  };
  for (const fps of RATES) {
    const track = Motion.resolve({ el: 'marker', x: [0, '82vw'], seconds: 2 }, fps, { vw: 1920, vh: 1080, width: 154, height: 65 });
    const baked = Motion.bake(track, fps);
    assert.deepEqual(baked.timing, timing[fps]);
    assert.equal(baked.keyframes.length, track.frames + 1);
    baked.keyframes.forEach((keyframe, n) => assert.deepEqual(keyframe, {
      transform: `translate(${Motion.stateAt([track], n).marker.x}px, 0px)`, offset: n / track.frames, easing: 'step-end',
    }));
    assert.equal(baked.keyframes[0].transform, 'translate(0px, 0px)');
    assert.equal(baked.keyframes.at(-1).transform, 'translate(1574px, 0px)');
  }
});

test('every baked value is literal: pixels, opacity and degrees as the frame table has them', () => {
  const track = Motion.resolve({ el: 'plate', x: ['-104%', 0], y: [4, 0], opacity: [0, 1], rotate: [-90, 0], seconds: 0.08 }, 25, PLATE);
  assert.deepEqual(Motion.bake(track, 25), {
    keyframes: [
      { transform: 'translate(-1057px, 4px) rotate(-90deg)', opacity: '0', offset: 0, easing: 'step-end' },
      { transform: 'translate(-529px, 2px) rotate(-45deg)', opacity: '0.5', offset: 0.5, easing: 'step-end' },
      { transform: 'translate(0px, 0px) rotate(0deg)', opacity: '1', offset: 1, easing: 'step-end' },
    ],
    timing: { duration: 80, delay: -20, fill: 'both' },
  });
  const fade = Motion.resolve({ el: 'plate', opacity: [1, 0], seconds: 0.04, delay: 0.2 }, 50, PLATE);
  assert.deepEqual(Motion.bake(fade, 50), {
    keyframes: [{ opacity: '1', offset: 0, easing: 'step-end' }, { opacity: '0.5', offset: 0.5, easing: 'step-end' },
      { opacity: '0', offset: 1, easing: 'step-end' }],
    timing: { duration: 40, delay: 190, fill: 'both' },   // 10 frames of delay, less the half tick
  });
});

test('a loop is two keyframes stepped by the browser, one step per tick, for ever', () => {
  const duration = { 25: 40000, 30: 33333, 50: 40000, 60: 33332 };   // 1000 or 2000 steps of one tick
  const delay = { 25: -20, 30: -16.6665, 50: -10, 60: -8.333 };
  for (const fps of RATES) {
    const crawl = Motion.resolve({ el: 'crawl', crawl: { pxPerSecond: 110 } }, fps, { content: 3999.2 });
    assert.deepEqual(Motion.bake(crawl, fps), {
      keyframes: [{ transform: 'translate(0px, 0px)' }, { transform: 'translate(-4000px, 0px)' }],
      timing: { duration: duration[fps], delay: delay[fps], iterations: Infinity, easing: `steps(${crawl.frames})` },
    });
    const spin = Motion.resolve({ el: 'ring', spin: { seconds: 4 } }, fps, {});
    assert.deepEqual(Motion.bake(spin, fps), {
      keyframes: [{ transform: 'rotate(0deg)' }, { transform: 'rotate(360deg)' }],
      timing: { duration: 4 * fps * Motion.tickUs(fps) / 1000, delay: delay[fps], iterations: Infinity, easing: `steps(${4 * fps})` },
    });
  }
});

test('a loaded graphic rests in its start state with nothing running', async () => {
  const { parts, clock } = await mount(LOWER_THIRD, { layout: LOWER_THIRD_LAYOUT });
  assert.deepEqual(parts.plate.style, HIDDEN);
  assert.deepEqual(parts.plate.animations, []);
  assert.deepEqual(clock.pending, []);
  assert.deepEqual([parts.name.textContent, parts.role.textContent], ['Ada Lovelace', 'Analyst · Engine No. 1']);
});

test('nothing of the graphic is visible until its start state is in place', async () => {
  let fontsLoaded;
  const fake = page({ layout: LOWER_THIRD_LAYOUT, fontsReady: new Promise((resolve) => { fontsLoaded = resolve; }) });
  const element = new (fake.Motion.graphic(LOWER_THIRD))();
  const loaded = element.load({ renderCharacteristics: { frameRate: 50 } });
  await settle();
  assert.equal(element.style.visibility, 'hidden');
  assert.equal(element.parts.plate.style.transform, undefined, 'lengths wait for the fonts: text sets widths');
  fontsLoaded();
  await loaded;
  assert.equal(element.style.visibility, '');
  assert.deepEqual(element.parts.plate.style, HIDDEN);
});

test('playing asks the browser for exactly the baked move, at every rate', async () => {
  for (const fps of RATES) {
    const { element, parts } = await mount(LOWER_THIRD, { fps, layout: LOWER_THIRD_LAYOUT, width: 1016, height: 172 });
    element.playAction({});
    const [animation] = parts.plate.animations;
    const expected = Motion.bake(Motion.resolve(LOWER_THIRD.play[0], fps, PLATE), fps);
    assert.equal(parts.plate.animations.length, 1);
    assert.deepEqual(animation.keyframes, expected.keyframes);
    assert.deepEqual(animation.timing, expected.timing);
    assert.equal(animation.keyframes.length, Math.round(0.6 * fps) + 1);
    // Frame 0 is the resting start state, so the first tick of a move repaints nothing.
    assert.deepEqual({ ...animation.keyframes[0], willChange: 'transform, opacity' },
      { ...HIDDEN, offset: 0, easing: 'step-end' });
    animation.finish();
    await settle();
  }
});

test('at the end of a move the end state is committed and the animation cancelled', async () => {
  const { element, parts, clock } = await mount(LOWER_THIRD, { layout: LOWER_THIRD_LAYOUT });
  let done = false;
  const played = element.playAction({}).then((result) => { done = true; return result; });
  const [animation] = parts.plate.animations;
  await settle();
  assert.equal(done, false, 'playAction resolves when the move has ended');
  assert.deepEqual(parts.plate.style, HIDDEN, 'the animation, not inline style, shows the move');
  assert.equal(animation.cancelled, false);
  animation.finish();
  assert.deepEqual(await played, { statusCode: 200, currentStep: 0 });
  assert.deepEqual(parts.plate.style, SHOWN);
  assert.equal(animation.cancelled, true);
  assert.deepEqual(parts.plate.running, []);
  assert.deepEqual(clock.pending, []);
});

test('skipAnimation commits the end state without asking for an animation', async () => {
  const { element, parts } = await mount(LOWER_THIRD, { layout: LOWER_THIRD_LAYOUT });
  assert.deepEqual(await element.playAction({ skipAnimation: true }), { statusCode: 200, currentStep: 0 });
  assert.deepEqual(parts.plate.style, SHOWN);
  await element.stopAction({ skipAnimation: true });
  assert.deepEqual(parts.plate.style, HIDDEN);
  assert.deepEqual(parts.plate.animations, []);
});

test('a new action takes the element over: the move in flight is cancelled, not finished', async () => {
  const { element, parts } = await mount(LOWER_THIRD, { layout: LOWER_THIRD_LAYOUT });
  const played = element.playAction({});
  const stopped = element.stopAction({});
  const [moveIn, moveOut] = parts.plate.animations;
  assert.equal(moveIn.cancelled, true);
  assert.equal(moveOut.cancelled, false);
  assert.equal(moveOut.keyframes[0].transform, 'translate(0px, 0px)', 'the new move starts from its own declared start');
  assert.deepEqual(await played, { statusCode: 200, currentStep: 0 }, 'the interrupted action still resolves');
  moveOut.finish();
  assert.deepEqual(await stopped, { statusCode: 200 });
  assert.deepEqual(parts.plate.style, HIDDEN);
  assert.deepEqual(parts.plate.running, []);
});

test('a custom action plays its own moves', async () => {
  const { element, parts } = await mount(ALPHA, { fps: 60, layout: { marker: { width: 154, height: 65 } } });
  assert.deepEqual(parts.marker.style, { willChange: 'transform, opacity' }, 'no play move, so no start state to commit');
  const moved = element.customAction({ id: 'right' });
  const [animation] = parts.marker.animations;
  assert.equal(animation.keyframes.length, 121);
  assert.deepEqual(animation.timing, { duration: 1999.92, delay: -8.333, fill: 'both' });
  animation.finish();
  assert.deepEqual(await moved, { statusCode: 200 });
  assert.equal(parts.marker.style.transform, 'translate(1574px, 0px)');
});

test('loops run only while the graphic is played, and the crawl fits its copies to the loop', async () => {
  for (const fps of RATES) {
    const { element, parts, clock } = await mount(TICKER, { fps, layout: TICKER_LAYOUT });
    assert.deepEqual(parts.crawl.children.map((copy) => copy.style.width), ['4000px', '4000px']);
    assert.deepEqual(parts.crawl.animations, [], 'nothing runs before play');
    await element.playAction({});
    const [crawl] = parts.crawl.animations;
    assert.deepEqual(crawl.keyframes, [{ transform: 'translate(0px, 0px)' }, { transform: 'translate(-4000px, 0px)' }]);
    assert.equal(crawl.timing.easing, fps > 30 ? 'steps(2000)' : 'steps(1000)');
    assert.equal(crawl.timing.iterations, Infinity);
    await element.playAction({ goto: 0 });
    assert.equal(parts.crawl.animations.length, 1, 'playing again does not restart a running loop');
    await element.stopAction({});
    assert.equal(crawl.cancelled, true);
    assert.deepEqual(clock.pending, []);
  }
});

test('a loop keeps running through the stop move and ends with it', async () => {
  const { element, parts } = await mount({
    html: '<div id="plate"><div id="ring"></div></div>',
    play: [{ el: 'plate', opacity: [0, 1], seconds: 0.2 }],
    stop: [{ el: 'plate', opacity: [1, 0], seconds: 0.2 }],
    loop: [{ el: 'ring', spin: { seconds: 4 } }],
  });
  element.playAction({});
  const [spin] = parts.ring.animations;
  parts.plate.animations[0].finish();
  await settle();
  const stopped = element.stopAction({});
  await settle();
  assert.equal(spin.cancelled, false, 'still turning while the plate fades out');
  parts.plate.animations[1].finish();
  await stopped;
  assert.equal(spin.cancelled, true);
});

test('a play during the stop move keeps the loops', async () => {
  const { element, parts } = await mount({
    html: '<div id="plate"></div><div id="ring"></div>',
    stop: [{ el: 'plate', opacity: [1, 0], seconds: 0.2 }],
    loop: [{ el: 'ring', spin: { seconds: 4 } }],
  });
  await element.playAction({});
  const stopped = element.stopAction({});
  await element.playAction({ goto: 0 });
  parts.plate.animations[0].finish();
  await stopped;
  assert.deepEqual(parts.ring.animations.map((animation) => animation.cancelled), [false]);
});

test('loops end with the last stop move, however many stops were asked', async () => {
  const { element, parts } = await mount({
    html: '<div id="plate"></div><div id="ring"></div>',
    stop: [{ el: 'plate', opacity: [1, 0], seconds: 0.2 }],
    loop: [{ el: 'ring', spin: { seconds: 4 } }],
  });
  await element.playAction({});
  const first = element.stopAction({});
  const second = element.stopAction({});   // takes the plate over: the first stop is done at once
  await first;
  assert.deepEqual(parts.ring.animations.map((animation) => animation.cancelled), [false]);
  parts.plate.animations[1].finish();
  await second;
  assert.deepEqual(parts.ring.animations.map((animation) => animation.cancelled), [true]);
});

test('an update measures again, and restarts only a loop whose length changed', async () => {
  const { element, parts, layout } = await mount({
    ...TICKER,
    html: '<div id="crawl"><span></span><span></span></div><div id="ring"></div>',
    loop: [...TICKER.loop, { el: 'ring', spin: { seconds: 4 } }],
  }, { layout: structuredClone(TICKER_LAYOUT) });
  await element.playAction({});
  layout.crawl.content = 5000.5;
  assert.deepEqual(await element.updateAction({ data: {} }), { statusCode: 200 });
  assert.deepEqual(parts.crawl.children.map((copy) => copy.style.width), ['5002px', '5002px']);
  assert.deepEqual(parts.crawl.animations.map((animation) => animation.cancelled), [true, false]);
  assert.equal(parts.crawl.animations[1].keyframes[1].transform, 'translate(-5002px, 0px)');
  assert.equal(parts.ring.animations.length, 1, 'the spin was not touched');
});

test('timed changes are timers on wall-clock boundaries, alive only while played', async () => {
  const ticks = [];
  const { element, clock } = await mount({
    html: '<div id="dot"></div><div id="clock"></div>',
    every: [{ seconds: 1, run: (el, n) => ticks.push(['second', n, el.clock.id]) },
      { seconds: 0.5, run: (el, n) => ticks.push(['half', n, el.dot.id]) }],
  }, { now: 1_700_000_000_250 });
  assert.deepEqual([ticks, clock.pending], [[], []]);
  await element.playAction({});
  assert.deepEqual(ticks, [['second', 1_700_000_000, 'clock'], ['half', 3_400_000_000, 'dot']]);
  assert.deepEqual(clock.pending, [250, 750], 'one timer each, aimed at the next boundary');
  await clock.advance(750);
  assert.deepEqual(ticks.slice(2), [['half', 3_400_000_001, 'dot'], ['second', 1_700_000_001, 'clock'], ['half', 3_400_000_002, 'dot']]);
  assert.deepEqual(clock.pending, [500, 1000]);
  await element.playAction({ goto: 0 });
  assert.deepEqual(clock.pending, [500, 1000], 'playing again does not add timers');
  await element.stopAction({});
  assert.deepEqual(clock.pending, []);
});

test('dispose leaves nothing running and nothing drawn', async () => {
  const { element, parts, clock } = await mount({
    ...TICKER, play: [{ el: 'tag', opacity: [0, 1], seconds: 1 }], every: [{ seconds: 1, run() {} }],
  }, { layout: TICKER_LAYOUT });
  element.playAction({});
  assert.deepEqual(await element.dispose(), { statusCode: 200 });
  assert.deepEqual([...parts.tag.running, ...parts.crawl.running], []);
  assert.deepEqual(clock.pending, []);
  assert.equal(element.shadowRoot.innerHTML, '');
});
