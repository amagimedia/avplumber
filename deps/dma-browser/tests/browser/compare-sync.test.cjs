const assert = require('node:assert/strict');
const { test } = require('node:test');
const { compare } = require('./compare-sync.cjs');

function trace(shift = 0) {
  const paints = Array.from({ length: 100 }, (_, i) => ({
    unixMs: 1000 + i * 40 + shift,
    monoNs: String(BigInt(i * 40 + shift) * 1000000n),
  }));
  return {
    kind: 'browser-sync-probe-v1',
    label: 'replica',
    fps: 25,
    epochMs: 0,
    phase: 0,
    startAtMs: 1000,
    endAtMs: 5000,
    paints,
    samples: [{ unixMs: 1200, loops: [{ elapsedMs: 1190 }] }],
    errors: [],
  };
}

test('accepts bounded wall-clock phase difference', () =>
  assert.equal(compare(trace(), trace(3)).pass, true));
test('rejects freezes despite perfect remaining timestamps', () => {
  const b = trace();
  b.paints.splice(20, 10);
  assert.match(compare(trace(), b).failures.join(), /paint delivery gap/);
});
test('rejects a stream that stops halfway through', () => {
  const b = trace();
  b.paints.splice(50);
  assert.equal(compare(trace(), b).pass, false);
});
test('rejects unaligned animation state and missing samples', () => {
  const b = trace();
  b.samples[0].loops[0].elapsedMs = 50;
  assert.equal(compare(trace(), b).pass, false);
  b.samples = [];
  assert.equal(compare(trace(), b).pass, false);
});
test('rejects different measurement intervals and probe errors', () => {
  const b = trace();
  b.startAtMs -= 200;
  assert.match(compare(trace(), b).failures.join(), /different startAtMs/);
  b.startAtMs = 1000;
  b.errors.push('no valid shared textures');
  assert.equal(compare(trace(), b).pass, false);
});
test('rejects excessive phase difference', () =>
  assert.equal(compare(trace(), trace(15)).pass, false));

test('rejects half-rate output even if each remaining frame is on the common grid', () => {
  const b = trace();
  b.paints = b.paints.filter((_, index) => index % 2 === 0);
  assert.match(compare(trace(), b).failures.join(), /paint count/);
});

test('rejects backwards clocks and invalid rates', () => {
  const b = trace();
  b.paints[10].unixMs = b.paints[0].unixMs;
  assert.throws(() => compare(trace(), b), /nonmonotonic/);
  const c = trace();
  c.fps = 0;
  assert.throws(() => compare(c, c), /rate/);
});
