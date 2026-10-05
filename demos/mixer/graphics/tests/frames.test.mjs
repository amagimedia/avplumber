// The frame table: whole frames and whole pixels, with no browser involved. Expected positions of
// eased moves were computed outside the engine (Newton's method on the CSS cubic-bezier).
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { page, RATES } from './page.mjs';

const { Motion } = page();
const VIEW = { vw: 1920, vh: 1080 };
const PLATE = { ...VIEW, width: 1016, height: 131 };
const table = (track, prop) => Array.from({ length: track.frames + 1 }, (_, n) => Motion.stateAt([track], n)[track.el][prop]);
const isWhole = (values) => values.every(Number.isInteger);

test('a tick is the browser\'s own integer division of a second', () => {
  assert.deepEqual(RATES.map(Motion.tickUs), [40000, 33333, 20000, 16666]);
});

test('durations are whole frames, at least one', () => {
  const frames = (seconds) => RATES.map((fps) => Motion.frames(seconds, fps));
  assert.deepEqual(frames(2), [50, 60, 100, 120]);
  assert.deepEqual(frames(0.6), [15, 18, 30, 36]);
  assert.deepEqual(frames(0.4), [10, 12, 20, 24]);
  assert.deepEqual(frames(4), [100, 120, 200, 240]);
  assert.deepEqual(frames(0.001), [1, 1, 1, 1]);
});

test('lengths become whole pixels: px, % of the element\'s own box, vw and vh', () => {
  const ends = (move, box = PLATE) => {
    const track = Motion.resolve({ el: 'a', seconds: 1, ...move }, 50, box);
    return [track.from, track.to];
  };
  assert.deepEqual(ends({ x: [0, '82vw'] }), [{ x: 0 }, { x: 1574 }]);        // 1574.4
  assert.deepEqual(ends({ x: ['-104%', 0] }), [{ x: -1057 }, { x: 0 }]);      // -1056.64
  assert.deepEqual(ends({ y: ['100%', '-10vh'] }), [{ y: 131 }, { y: -108 }]);
  // Halves round away from zero, so a move and its mirror image travel the same distance.
  assert.deepEqual(ends({ x: [12.4, '7.5px'], y: ['50%', '-50%'] }, { ...PLATE, height: 3 }),
    [{ x: 12, y: 2 }, { x: 8, y: -2 }]);
});

test('a linear move lands on a whole pixel on every frame and ends exactly on its target', () => {
  const first = { 25: [0, 31, 63, 94, 126], 30: [0, 26, 52, 79, 105], 50: [0, 16, 31, 47, 63], 60: [0, 13, 26, 39, 52] };
  for (const fps of RATES) {
    const track = Motion.resolve({ el: 'marker', x: [0, '82vw'], seconds: 2 }, fps, { ...VIEW, width: 154, height: 65 });
    const x = table(track, 'x');
    assert.equal(track.frames, 2 * fps);
    assert.equal(x.length, 2 * fps + 1);
    assert.ok(isWhole(x), `${fps} fps`);
    assert.deepEqual(x.slice(0, 5), first[fps]);
    assert.equal(x[fps], 787);
    assert.equal(x.at(-1), 1574);
    assert.ok(x.every((value, n) => n === 0 || value > x[n - 1]), 'every frame moves forward');
    const steps = new Set(x.slice(1).map((value, n) => value - x[n]));
    assert.ok(steps.size <= 2 && Math.max(...steps) - Math.min(...steps) <= 1, 'steps differ by at most one pixel');
  }
});

test('the table holds the start before frame 0 and the target after the last frame', () => {
  const track = Motion.resolve({ el: 'marker', x: [3, 40], opacity: [0.2, 0.9], seconds: 1 }, 50, PLATE);
  assert.deepEqual(Motion.stateAt([track], -7), { marker: { x: 3, opacity: 0.2 } });
  assert.deepEqual(Motion.stateAt([track], 0), { marker: { x: 3, opacity: 0.2 } });
  assert.deepEqual(Motion.stateAt([track], 50), { marker: { x: 40, opacity: 0.9 } });
  assert.deepEqual(Motion.stateAt([track], 5000), { marker: { x: 40, opacity: 0.9 } });
});

const PLATE_IN = {
  25: [-1057, -773, -518, -336, -222, -150, -102, -69, -47, -30, -19, -11, -6, -2, -1, 0],
  30: [-1057, -820, -596, -417, -292, -208, -150, -109, -79, -57, -41, -28, -19, -12, -7, -4, -2, 0, 0],
  50: [-1057, -915, -773, -638, -518, -417, -336, -273, -222, -182, -150, -124, -102, -84, -69, -57, -47, -38, -30, -24,
    -19, -15, -11, -8, -6, -4, -2, -1, -1, 0, 0],
  60: [-1057, -938, -820, -704, -596, -499, -417, -349, -292, -246, -208, -176, -150, -128, -109, -93, -79, -67, -57,
    -48, -41, -34, -28, -23, -19, -15, -12, -9, -7, -5, -4, -3, -2, -1, 0, 0, 0],
};
const PLATE_OUT = {
  25: [0, -18, -66, -137, -227, -333, -454, -586, -731, -887, -1057],
  30: [0, -13, -47, -99, -165, -244, -333, -433, -541, -657, -782, -914, -1057],
  50: [0, -5, -18, -39, -66, -99, -137, -180, -227, -278, -333, -392, -454, -518, -586, -657, -731, -808, -887, -970, -1057],
  60: [0, -3, -13, -27, -47, -71, -99, -130, -165, -203, -244, -287, -333, -382, -433, -486, -541, -598, -657, -719,
    -782, -847, -914, -984, -1057],
};

test('an eased move lands on a whole pixel on every frame: the lower third in and out', () => {
  for (const fps of RATES) {
    const moveIn = Motion.resolve({ el: 'plate', x: ['-104%', 0], opacity: [0, 1], seconds: 0.6, ease: [0.2, 0.8, 0.2, 1] }, fps, PLATE);
    const moveOut = Motion.resolve({ el: 'plate', x: [0, '-104%'], opacity: [1, 0], seconds: 0.4, ease: 'in' }, fps, PLATE);
    assert.deepEqual(table(moveIn, 'x'), PLATE_IN[fps]);
    assert.deepEqual(table(moveOut, 'x'), PLATE_OUT[fps]);
    assert.deepEqual([table(moveIn, 'opacity')[0], table(moveIn, 'opacity').at(-1)], [0, 1]);
    assert.deepEqual([table(moveOut, 'opacity')[0], table(moveOut, 'opacity').at(-1)], [1, 0]);
  }
  const fade = Motion.resolve({ el: 'plate', opacity: [0, 1], seconds: 0.6, ease: [0.2, 0.8, 0.2, 1] }, 25, PLATE);
  assert.deepEqual(table(fade, 'opacity'),
    [0, 0.269, 0.51, 0.682, 0.79, 0.858, 0.903, 0.934, 0.956, 0.971, 0.982, 0.99, 0.995, 0.998, 1, 1]);
});

test('named eases are the CSS curves', () => {
  // One frame in the middle of a two-frame move over a million pixels shows the curve at 0.5.
  const middle = (ease) => Motion.stateAt([{ el: 'a', start: 0, frames: 2, ease: Motion.resolve(
    { el: 'a', x: [0, 1], seconds: 1, ease }, 50, PLATE).ease, from: { x: 0 }, to: { x: 1000000 } }], 1).a.x;
  assert.equal(middle('linear'), 500000);
  assert.equal(middle('in'), 315357);         // cubic-bezier(.42, 0, 1, 1)
  assert.equal(middle('out'), 684643);        // cubic-bezier(0, 0, .58, 1)
  assert.equal(middle('in-out'), 500000);     // cubic-bezier(.42, 0, .58, 1)
  assert.equal(middle([0.25, 0.1, 0.25, 1]), 802403);   // CSS "ease"
});

test('a delay is whole frames during which the element keeps its start', () => {
  for (const [fps, start] of [[25, 5], [30, 6], [50, 10], [60, 12]]) {
    const track = Motion.resolve({ el: 'a', x: [0, 100], seconds: 1, delay: 0.2 }, fps, PLATE);
    assert.equal(track.start, start);
    assert.deepEqual(Motion.stateAt([track], start), { a: { x: 0 } });
    assert.notEqual(Motion.stateAt([track], start + 1).a.x, 0);
    assert.deepEqual(Motion.stateAt([track], start + fps), { a: { x: 100 } });
  }
});

test('a crawl moves the same whole number of pixels on every frame and wraps on that grid', () => {
  const expected = { 25: 4, 30: 4, 50: 2, 60: 2 };   // 110 px/s; today's hand-written steps
  for (const fps of RATES) {
    const track = Motion.resolve({ el: 'crawl', crawl: { pxPerSecond: 110 } }, fps, { ...VIEW, content: 3999.2 });
    const step = expected[fps], loop = -track.to.x;
    assert.equal(track.loop, true);
    assert.deepEqual(track.from, { x: 0 });
    assert.equal(loop % step, 0, 'the loop is a whole number of steps');
    assert.ok(loop >= 3999.2 && loop < 3999.2 + step, 'the loop is the content, rounded up to the step');
    assert.equal(track.frames, loop / step);
    const x = Array.from({ length: 3 * track.frames + 1 }, (_, n) => Motion.stateAt([track], n).crawl.x);
    assert.ok(isWhole(x));
    // A wrap jumps back by one loop less a step, which shows the same pixels as one more step.
    assert.ok(x.every((value, n) => n === 0 || [-step, loop - step].includes(value - x[n - 1])), `${fps} fps`);
    assert.deepEqual([x[track.frames - 1], x[track.frames], x[track.frames + 1]], [step - loop, 0, -step]);
  }
});

test('a spin is a whole turn in a whole number of frames', () => {
  for (const fps of RATES) {
    const track = Motion.resolve({ el: 'ring', spin: { seconds: 4 } }, fps, VIEW);
    assert.equal(track.frames, 4 * fps);
    assert.deepEqual([track.loop, track.from, track.to], [true, { rotate: 0 }, { rotate: 360 }]);
    assert.equal(Motion.stateAt([track], fps).ring.rotate, 90);
    assert.equal(Motion.stateAt([track], 4 * fps).ring.rotate, 0);
    assert.equal(Motion.stateAt([track], 9 * fps).ring.rotate, 90);
  }
});

test('one table holds every element of an action', () => {
  const tracks = [
    Motion.resolve({ el: 'plate', x: [-100, 0], seconds: 1 }, 50, PLATE),
    Motion.resolve({ el: 'accent', opacity: [0, 1], y: ['100%', 0], seconds: 0.5, delay: 0.5 }, 50, PLATE),
  ];
  assert.deepEqual(Motion.stateAt(tracks, 25), { plate: { x: -50 }, accent: { opacity: 0, y: 131 } });
  assert.deepEqual(Motion.stateAt(tracks, 50), { plate: { x: 0 }, accent: { opacity: 1, y: 0 } });
});

test('a wrong move or loop names what is wrong and where', () => {
  const resolve = (spec) => () => Motion.resolve(spec, 50, PLATE, 'play[2]');
  assert.throws(resolve({ x: [0, 1], seconds: 1 }), /Motion: play\[2\]: el must be the id of an element/);
  assert.throws(resolve({ el: 'a', x: [0, '12em'], seconds: 1 }), /Motion: play\[2\]\.x: expected \[from, to\].*"12em"/);
  assert.throws(resolve({ el: 'a', x: 40, seconds: 1 }), /Motion: play\[2\]\.x: expected \[from, to\]/);
  assert.throws(resolve({ el: 'a', opacity: [0, 2], seconds: 1 }), /Motion: play\[2\]\.opacity: expected \[from, to\]/);
  assert.throws(resolve({ el: 'a', x: [0, 1] }), /Motion: play\[2\]\.seconds: must be a positive number/);
  assert.throws(resolve({ el: 'a', x: [0, 1], seconds: 1, delay: -1 }), /Motion: play\[2\]\.delay: must be zero or a positive number/);
  assert.throws(resolve({ el: 'a', x: [0, 1], seconds: 1, ease: 'bounce' }), /Motion: play\[2\]\.ease: expected linear, in, out, in-out or \[x1, y1, x2, y2\]/);
  assert.throws(resolve({ el: 'a', x: [0, 1], seconds: 1, ease: [2, 0, 1, 1] }), /Motion: play\[2\]\.ease/);
  assert.throws(resolve({ el: 'a', seconds: 1 }), /Motion: play\[2\]: moves nothing; give x, y, rotate or opacity/);
  assert.throws(resolve({ el: 'a', scale: [0, 1], seconds: 1 }), /Motion: play\[2\]: unknown key "scale"/);
  assert.throws(resolve({ el: 'a', crawl: { pxPerSecond: 0 } }), /Motion: play\[2\]\.crawl\.pxPerSecond: must be a positive number/);
  assert.throws(resolve({ el: 'a', spin: {} }), /Motion: play\[2\]\.spin\.seconds: must be a positive number/);
  assert.throws(resolve({ el: 'a', spin: { seconds: 4 }, x: [0, 1] }), /Motion: play\[2\]: unknown key "x"/);
});
