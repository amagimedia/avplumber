import assert from "node:assert/strict";
import { test } from "node:test";
import { createMediaWatchdog } from "../media-watchdog.mjs";

function fixture({callbacks = true} = {}) {
  let clock = 0, serial = 0, playing = 0;
  const tasks = new Map(), frames = new Map(), failures = [];
  const video = Object.assign(new EventTarget(), {currentTime: 42, paused: true, ended: false, readyState: 4});
  const visibility = Object.assign(new EventTarget(), {hidden: false});
  if (callbacks) {
    video.requestVideoFrameCallback = fn => { frames.set(++serial, fn); return serial; };
    video.cancelVideoFrameCallback = id => frames.delete(id);
  }
  const watchdog = createMediaWatchdog(video, {
    now: () => clock, visibility,
    timers: {
      setTimeout(fn, delay) { tasks.set(++serial, {fn, at: clock + delay}); return serial; },
      clearTimeout(id) { tasks.delete(id); },
    },
    onPlaying: () => playing++, onStall: error => failures.push(error.message),
  });
  return {
    video, visibility, watchdog, failures, frames, tasks,
    get playing() { return playing; },
    tick(ms) {
      const until = clock + ms;
      while (true) {
        const next = [...tasks].sort((a, b) => a[1].at - b[1].at)[0];
        if (!next || next[1].at > until) break;
        clock = next[1].at; tasks.delete(next[0]); next[1].fn();
      }
      clock = until;
    },
    frame() {
      video.paused = false;
      const next = [...frames][0];
      frames.delete(next[0]); next[1]();
    },
    hide(hidden) { visibility.hidden = hidden; visibility.dispatchEvent(new Event("visibilitychange")); },
    pause(paused) { video.paused = paused; video.dispatchEvent(new Event(paused ? "pause" : "playing")); },
  };
}

test("no track or stale readyState never cancels the first-frame deadline", () => {
  for (const attach of [false, true]) {
    const f = fixture();
    if (attach) f.watchdog.attach();
    f.tick(11000);
    assert.equal(f.playing, 0);
    assert.deepEqual(f.failures, []);
    f.tick(1000);
    assert.deepEqual(f.failures, ["No video received within 12 seconds"]);
    assert.equal(f.frames.size, 0);
    f.tick(60000);
    assert.equal(f.failures.length, 1);
  }
});

test("presentation keeps playing, then a frozen video triggers recovery once", () => {
  const f = fixture(); f.watchdog.attach(); f.watchdog.attach();
  assert.equal(f.frames.size, 1, "audio and video tracks share one watcher");
  for (let i = 0; i < 30; i++) { f.tick(1000); f.frame(); }
  assert.equal(f.playing, 1);
  assert.equal(f.watchdog.playing, true);
  f.tick(7000); assert.deepEqual(f.failures, []);
  f.tick(1000); assert.deepEqual(f.failures, ["Video stalled for 8 seconds"]);
  f.tick(30000); assert.equal(f.failures.length, 1);
});

test("background tabs and intentional pauses get a fresh grace period on return", () => {
  const f = fixture(); f.watchdog.attach();
  f.hide(true); f.tick(60000); f.hide(false);
  f.tick(11000); assert.deepEqual(f.failures, []);
  f.frame();
  f.pause(true); f.tick(60000); f.pause(false);
  f.tick(7000); assert.deepEqual(f.failures, []);
  f.frame(); f.tick(7000); assert.deepEqual(f.failures, []);
  f.tick(1000); assert.equal(f.failures.length, 1);
});

test("ended live video is recoverable even if the browser marks it paused", () => {
  const f = fixture(); f.watchdog.attach(); f.frame();
  f.video.ended = true; f.video.paused = true;
  f.tick(8000); assert.equal(f.failures.length, 1);
});

test("disconnect cancels callbacks; late old-session callbacks cannot revive it", () => {
  const f = fixture(); f.watchdog.attach();
  const lateFrame = [...f.frames.values()][0];
  f.watchdog.stop(); lateFrame(); f.tick(60000);
  assert.equal(f.playing, 0);
  assert.equal(f.frames.size, 0);
  assert.equal(f.tasks.size, 0);
  assert.deepEqual(f.failures, []);
});

test("fallback browsers require advancing media time, not buffered old video", () => {
  const f = fixture({callbacks: false}); f.watchdog.attach(); f.video.paused = false;
  f.tick(11000); assert.equal(f.playing, 0);
  f.video.currentTime++; f.tick(1000); assert.equal(f.playing, 1);
  f.tick(8000); assert.deepEqual(f.failures, ["Video stalled for 8 seconds"]);
});
