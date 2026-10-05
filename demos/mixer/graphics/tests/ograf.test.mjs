// The element class as an EBU OGraf graphic (ograf.ebu.io, v1): method names, return payloads,
// the step rule of playAction, and the non-real-time pair setActionsSchedule / goToTime.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { ALPHA, LOWER_THIRD, LOWER_THIRD_LAYOUT, TICKER, TICKER_LAYOUT, UNEVEN, UNEVEN_LAYOUT, inline, mount, page } from './page.mjs';

const lowerThird = (options) => mount(LOWER_THIRD, { layout: LOWER_THIRD_LAYOUT, ...options });
const shown = (plate) => [plate.style.transform, plate.style.opacity];

test('graphic() returns an HTMLElement class with the OGraf methods and registers nothing', () => {
  const { Motion, registry } = page();
  const Graphic = Motion.graphic(LOWER_THIRD);
  for (const method of ['load', 'dispose', 'updateAction', 'playAction', 'stopAction', 'customAction', 'goToTime', 'setActionsSchedule']) {
    assert.equal(typeof Graphic.prototype[method], 'function', method);
  }
  // An OGraf renderer defines the element under a name of its own; a class can be defined once.
  assert.equal(registry.size, 0);
  assert.equal(new Graphic().shadowRoot, null, 'nothing is built before load()');
});

test('every action resolves to an OGraf return payload', async () => {
  const { element } = await mount({ ...ALPHA, ...LOWER_THIRD, html: `${ALPHA.html}${LOWER_THIRD.html}` }, { layout: LOWER_THIRD_LAYOUT });
  const skip = { skipAnimation: true };
  assert.deepEqual(await element.load({ renderCharacteristics: { frameRate: 50 } }), { statusCode: 200 });
  assert.deepEqual(await element.updateAction({ data: { name: 'Alan Turing' }, ...skip }), { statusCode: 200 });
  assert.deepEqual(await element.playAction(skip), { statusCode: 200, currentStep: 0 });
  assert.deepEqual(await element.customAction({ id: 'right', payload: null, ...skip }), { statusCode: 200 });
  assert.deepEqual(await element.customAction({ id: 'toString', ...skip }),
    { statusCode: 404, statusMessage: 'Motion: no custom action "toString"; this graphic has: right, left' });
  assert.deepEqual(await element.stopAction(skip), { statusCode: 200 });
  assert.deepEqual(await element.setActionsSchedule({ schedule: [] }), { statusCode: 200 });
  assert.deepEqual(await element.goToTime({ timestamp: 0 }), { statusCode: 200 });
  assert.deepEqual(await element.dispose({}), { statusCode: 200 });
});

test('playAction follows the OGraf step rule for a graphic of one step', async () => {
  const { element, parts } = await lowerThird();
  const play = async (params) => (await element.playAction({ ...params, skipAnimation: true })).currentStep;
  assert.equal(await play({}), 0, 'from the start, the default delta of 1 reaches step 0');
  assert.equal(parts.plate.style.opacity, '1');
  assert.equal(await play({ goto: 0 }), 0);
  assert.equal(await play({}), undefined, 'one more step is past the last one: the graphic ends');
  assert.equal(parts.plate.style.opacity, '0');
  assert.equal(await play({ delta: 1 }), 0);
  assert.equal(await play({ goto: 3 }), undefined);
  assert.equal(await play({ goto: 0, delta: 5 }), 0, 'goto wins over delta');
  await element.stopAction({ skipAnimation: true });
  assert.equal(await play({ delta: 2 }), undefined);
});

test('load needs the frame rate, and merges its data over the declared defaults', async () => {
  const { Motion } = page();
  const Graphic = Motion.graphic(LOWER_THIRD);
  await assert.rejects(new Graphic().load({ data: {} }), /Motion: load\(\) needs renderCharacteristics\.frameRate/);
  await assert.rejects(new Graphic().playAction({}), /Motion: load\(\) has not finished/);
  const { element, parts } = await lowerThird({ data: { role: 'Codebreaking correspondent' } });
  assert.deepEqual([parts.name.textContent, parts.role.textContent], ['Ada Lovelace', 'Codebreaking correspondent']);
  await element.updateAction({ data: { name: 'Alan Turing' } });
  assert.deepEqual([parts.name.textContent, parts.role.textContent], ['Alan Turing', 'Codebreaking correspondent']);
});

test('load fails with the name of an element the markup does not have', async () => {
  const { Motion } = page();
  const Graphic = Motion.graphic({ ...LOWER_THIRD, html: '<div id="plat"><div id="name"></div><div id="role"></div></div>' });
  await assert.rejects(new Graphic().load({ renderCharacteristics: { frameRate: 50 } }),
    /Motion: play\[0\]\.el: no element with id "plate" in html \(it has: plat, name, role\)/);
  const Crawl = Motion.graphic({ html: '<div id="crawl"></div>', loop: [{ el: 'crawl', crawl: { pxPerSecond: 110 } }] });
  await assert.rejects(new Crawl().load({ renderCharacteristics: { frameRate: 50 } }),
    /Motion: loop\[0\]\.el: a crawl needs at least two children in "crawl", each a copy of the content/);
});

test('goToTime shows the frame of the schedule at that time, with nothing running', async () => {
  for (const fps of [25, 30, 50, 60]) {
    const { element, parts, clock, Motion } = await lowerThird({ fps });
    const box = { vw: 1920, vh: 1080, width: 1016, height: 131 };
    const moveIn = [Motion.resolve(LOWER_THIRD.play[0], fps, box)], moveOut = [Motion.resolve(LOWER_THIRD.stop[0], fps, box)];
    const expected = (tracks, n) => { const { x, opacity } = Motion.stateAt(tracks, n).plate; return [`translate(${x}px, 0px)`, String(opacity)]; };
    await element.setActionsSchedule({ schedule: [
      { timestamp: 5000, action: { type: 'stopAction', params: {} } },
      { timestamp: 1000, action: { type: 'playAction', params: {} } },
      { timestamp: 3000, action: { type: 'updateAction', params: { data: { name: 'Grace Hopper' } } } },
    ] });
    const frame = 1000 / fps;
    await element.goToTime({ timestamp: 0 });
    assert.deepEqual(shown(parts.plate), ['translate(-1057px, 0px)', '0']);
    for (const n of [0, 1, 2, 7, Math.round(0.6 * fps) - 1, Math.round(0.6 * fps)]) {
      await element.goToTime({ timestamp: 1000 + n * frame });
      assert.deepEqual(shown(parts.plate), expected(moveIn, n), `${fps} fps, frame ${n} of the move in`);
    }
    await element.goToTime({ timestamp: 2999 });
    assert.deepEqual([...shown(parts.plate), parts.name.textContent], ['translate(0px, 0px)', '1', 'Ada Lovelace']);
    await element.goToTime({ timestamp: 3000 });
    assert.equal(parts.name.textContent, 'Grace Hopper');
    for (const n of [0, 1, 5, Math.round(0.4 * fps)]) {
      await element.goToTime({ timestamp: 5000 + n * frame });
      assert.deepEqual(shown(parts.plate), expected(moveOut, n), `${fps} fps, frame ${n} of the move out`);
    }
    await element.goToTime({ timestamp: 1000 + 3 * frame });   // seeking back
    assert.deepEqual([...shown(parts.plate), parts.name.textContent], [...expected(moveIn, 3), 'Ada Lovelace']);
    assert.deepEqual(parts.plate.animations, [], 'seeking never starts an animation');
    assert.deepEqual(clock.pending, []);
  }
});

test('goToTime places loops by the frames since play, and stops what was running', async () => {
  const { element, parts } = await mount({ ...TICKER, stop: [{ el: 'tag', opacity: [1, 0], seconds: 0.2 }] }, { layout: TICKER_LAYOUT });
  await element.playAction({});
  const [live] = parts.crawl.animations;
  await element.setActionsSchedule({ schedule: [
    { timestamp: 2000, action: { type: 'playAction', params: { goto: 0 } } },
    { timestamp: 50000, action: { type: 'stopAction', params: {} } },
  ] });
  await element.goToTime({ timestamp: 2000 + 7 * 20 });
  assert.equal(live.cancelled, true);
  assert.equal(parts.crawl.style.transform, 'translate(-14px, 0px)');   // 7 frames of 2 px at 50 fps
  await element.goToTime({ timestamp: 2000 + 2003 * 20 });
  assert.equal(parts.crawl.style.transform, 'translate(-6px, 0px)', 'wrapped after 2000 frames');
  await element.goToTime({ timestamp: 50000 + 5 * 20 });   // 2405 frames after play, 5 into the 10 of the stop move
  assert.equal(parts.crawl.style.transform, 'translate(-810px, 0px)', 'still crawling during the stop move');
  await element.goToTime({ timestamp: 50000 + 10 * 20 });
  assert.equal(parts.crawl.style.transform, 'translate(0px, 0px)', 'at rest once the stop move has ended');
  assert.deepEqual(parts.crawl.animations, [live]);
});

test('goToTime shows a play from the loaded state, as real time does', async () => {
  const { element, parts: { plate, text } } = await mount(UNEVEN, { layout: UNEVEN_LAYOUT });
  const at = (timestamp, type) => ({ timestamp, action: { type, params: {} } });
  await element.setActionsSchedule({ schedule: [at(0, 'playAction'), at(1000, 'stopAction'), at(2000, 'playAction')] });
  const seen = async (timestamp) => { await element.goToTime({ timestamp }); return inline(plate, text); };
  const willChange = 'transform, opacity', played = await seen(900);
  assert.deepEqual(played, [{ transform: 'translate(0px, 0px)', willChange }, { willChange }]);
  assert.deepEqual(await seen(1900), [{ transform: 'translate(0px, 0px)', opacity: '0', willChange }, { transform: 'translate(0px, 20px)', willChange }]);
  assert.deepEqual(await seen(2900), played, 'the second play ends as the first did');
  assert.deepEqual(await seen(900), played, 'seeking back leaves nothing of the stop');
});

test('a scheduled action that skips its animation is at its end at once', async () => {
  const { element, parts } = await lowerThird();
  await element.setActionsSchedule({ schedule: [{ timestamp: 1000, action: { type: 'playAction', params: { skipAnimation: true } } }] });
  await element.goToTime({ timestamp: 1000 });
  assert.deepEqual(shown(parts.plate), ['translate(0px, 0px)', '1']);
});

test('a wrong declaration fails when the page loads, naming the place', () => {
  const { Motion } = page();
  const graphic = (declaration) => () => Motion.graphic({ html: '<div id="a"></div><div id="b"></div>', ...declaration });
  const move = { el: 'a', x: [0, 10], seconds: 1 };
  assert.throws(() => Motion.graphic({}), /Motion: html: must be the markup of the graphic, a string/);
  assert.throws(graphic({ plya: [move] }), /Motion: graphic: unknown key "plya" \(known: html, css, data, update, play, stop, change, actions, loop, every, demo, stagger\)/);
  assert.throws(graphic({ play: move }), /Motion: play: must be a list/);
  assert.throws(graphic({ play: [{ ...move, x: [0, '3em'] }] }), /Motion: play\[0\]\.x: expected \[from, to\]/);
  assert.throws(graphic({ stop: [move, { ...move, opacity: [0, 1] }] }), /Motion: stop\[1\]\.el: "a" is already moved by stop\[0\]; one move per element/);
  assert.throws(graphic({ play: [{ el: 'a', spin: { seconds: 4 } }] }), /Motion: play\[0\]: a spin or crawl belongs in loop/);
  assert.throws(graphic({ loop: [move] }), /Motion: loop\[0\]: a loop is a crawl or a spin; a move belongs in play, stop, change or actions/);
  assert.throws(graphic({ play: [move], loop: [{ el: 'a', spin: { seconds: 4 } }] }),
    /Motion: play\[0\]\.el: "a" is turned or crawled by loop, which owns its transform; move a wrapper element/);
  assert.throws(graphic({ actions: { stop: [move] } }), /Motion: actions\.stop: the name is taken; choose another id/);
  assert.throws(graphic({ css: ['#a { color: red; }'] }), /Motion: css: must be the styles of the graphic, a string/);
  assert.throws(graphic({ update: 'name' }), /Motion: update: must be a function \(el, data\)/);
  assert.throws(graphic({ every: [{ seconds: 0, run() {} }] }), /Motion: every\[0\]\.seconds: must be a positive number/);
  assert.throws(graphic({ every: [{ seconds: 1 }] }), /Motion: every\[0\]\.run: must be a function \(el, n\)/);
  assert.throws(graphic({ every: [{ seconds: 1, run() {}, phase: 0.5 }] }), /Motion: every\[0\]: unknown key "phase" \(known: seconds, run\)/);
  assert.throws(graphic({ demo: [[0, 'slide']] }), /Motion: demo\[0\]: unknown action "slide" \(known: play, stop, update, repeat\)/);
});
