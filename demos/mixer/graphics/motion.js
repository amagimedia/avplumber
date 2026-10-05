// The one motion engine of the mixer's HTML graphics.
//
// A graphic declares its motion in seconds and CSS-like lengths. The engine turns that into a frame
// table (whole frames of the window's rate, whole pixels) and hands the table to the browser's
// compositor as Web Animations with one step keyframe per frame, so no script runs per frame. At the
// end of a move the end state is written to inline style and the animation is cancelled: a resting
// page has nothing running but the timer to its next cue.
//
// No dependencies and no module syntax: graphic_pages.py inlines this text, then one graphic's, into
// host.html. README.md beside this file shows how to declare a graphic.
const Motion = (() => {
  'use strict';

  const fail = (where, problem) => { throw new Error(`Motion: ${where}: ${problem}`); };
  const positive = (value, where) => (value > 0 && Number.isFinite(value) ? value : fail(where, 'must be a positive number'));
  const known = (object, keys, where) => {
    for (const key of Object.keys(object)) if (!keys.includes(key)) fail(where, `unknown key "${key}" (known: ${keys.join(', ')})`);
  };

  // ---- Frames and pixels -----------------------------------------------------------------------

  // The browser ticks every floor(1e6 / fps) microseconds (16,666 at 60 fps), Electron's own integer
  // division. Durations are whole ticks of exactly that length, so a move ends on a tick.
  const tickUs = (fps) => Math.floor(1e6 / fps);
  const frames = (seconds, fps) => Math.max(1, Math.round(seconds * fps));
  // Halves round away from zero, so a move and its mirror image travel the same distance.
  const whole = (value) => Math.sign(value) * Math.round(Math.abs(value)) + 0;

  // A length is a number of pixels or a string in px, % of the moved element's own width (x) or
  // height (y), vw or vh. Anything else reads as NaN.
  const length = (side) => (value, box) => {
    const [, number, unit = 'px'] = /^(-?\d*\.?\d+)(px|%|vw|vh)?$/.exec(value) ?? [];
    return whole(number * { px: 1, '%': box[side] / 100, vw: box.vw / 100, vh: box.vh / 100 }[unit]);
  };
  // What a move can change, and how a declared end of it becomes a number.
  const PROPS = {
    x: { read: length('width'), hint: 'a length (pixels, "50%" of the element itself, "82vw" or "10vh")' },
    y: { read: length('height'), hint: 'a length (pixels, "50%" of the element itself, "82vw" or "10vh")' },
    rotate: { read: (value) => (typeof value === 'number' ? value : NaN), hint: 'a number of degrees' },
    opacity: { read: (value) => (typeof value === 'number' && value >= 0 && value <= 1 ? value : NaN), hint: 'a number from 0 to 1' },
  };

  const EASES = { linear: null, in: [0.42, 0, 1, 1], out: [0, 0, 0.58, 1], 'in-out': [0.42, 0, 0.58, 1] };
  // CSS cubic-bezier(x1, y1, x2, y2) at progress p. x(t) rises with t, so bisection finds t.
  function bezier([x1, y1, x2, y2], p) {
    const at = (a, b, t) => 3 * a * t * (1 - t) ** 2 + 3 * b * t * t * (1 - t) + t ** 3;
    let low = 0, high = 1;
    for (let i = 0; i < 50; i++) {
      const middle = (low + high) / 2;
      if (at(x1, x2, middle) < p) low = middle; else high = middle;
    }
    return at(y1, y2, (low + high) / 2);
  }

  // Motion that never ends. Its steps are equal by construction, so the browser plays it as two
  // keyframes with steps() easing instead of one keyframe per frame.
  const LOOPS = {
    // Content moves left by the same whole number of pixels on every frame and wraps after one copy
    // of itself: box.content, rounded up to a whole number of steps.
    crawl({ pxPerSecond }, fps, box, where) {
      const step = Math.max(1, Math.round(positive(pxPerSecond, `${where}.crawl.pxPerSecond`) / fps));
      const steps = Math.max(1, Math.ceil(box.content / step));
      return { frames: steps, from: { x: 0 }, to: { x: -steps * step } };
    },
    // One clockwise turn.
    spin: ({ seconds }, fps, box, where) => (
      { frames: frames(positive(seconds, `${where}.spin.seconds`), fps), from: { rotate: 0 }, to: { rotate: 360 } }),
  };

  /**
   * A declared move or loop as a track: whole frames at `fps`, whole pixels in `box`. box is the moved
   * element's { width, height }, the viewport's { vw, vh } and, for a crawl, the width of its
   * { content }. `where` names the declaration in errors.
   * A track is { el, start, frames, ease, from, to } and, for a loop, loop: true.
   */
  function resolve(spec, fps, box, where = 'track') {
    const { el, seconds, delay = 0, ease = 'linear', ...ends } = spec;
    if (typeof el !== 'string' || !el) fail(where, 'el must be the id of an element in html');
    const kind = Object.keys(LOOPS).find((name) => name in spec);
    if (kind) {
      known(spec, ['el', kind], where);
      return { el, loop: true, start: 0, ease: null, ...LOOPS[kind](spec[kind] ?? {}, fps, box, where) };
    }
    known(spec, ['el', 'seconds', 'delay', 'ease', ...Object.keys(PROPS)], where);
    positive(seconds, `${where}.seconds`);
    if (!(delay >= 0 && Number.isFinite(delay))) fail(`${where}.delay`, 'must be zero or a positive number');
    const curve = ease in EASES ? EASES[ease] : ease;
    if (curve !== null && !(Array.isArray(curve) && curve.length === 4 && curve.every(Number.isFinite)
        && curve[0] >= 0 && curve[0] <= 1 && curve[2] >= 0 && curve[2] <= 1)) {
      fail(`${where}.ease`, 'expected linear, in, out, in-out or [x1, y1, x2, y2] with x1 and x2 from 0 to 1');
    }
    const from = {}, to = {};
    for (const [prop, value] of Object.entries(ends)) {
      const pair = Array.isArray(value) && value.length === 2 ? value.map((end) => PROPS[prop].read(end, box)) : [NaN];
      if (!pair.every(Number.isFinite)) fail(`${where}.${prop}`, `expected [from, to], each ${PROPS[prop].hint}; got ${JSON.stringify(value)}`);
      [from[prop], to[prop]] = pair;
    }
    if (!Object.keys(from).length) fail(where, 'moves nothing; give x, y, rotate or opacity');
    return { el, start: Math.round(delay * fps), frames: frames(seconds, fps), ease: curve, from, to };
  }

  /**
   * The frame table: the state of every element of `tracks` on frame n, as { id: { x, y, rotate,
   * opacity } } with only the properties its tracks change. x and y are whole pixels on every frame.
   * A move holds its start before its first frame and its target after its last; a loop wraps.
   */
  function stateAt(tracks, n) {
    const states = {};
    for (const { el, loop, start, frames: count, ease, from, to } of tracks) {
      const k = loop ? ((n - start) % count + count) % count : Math.min(Math.max(n - start, 0), count);
      const eased = ease && k > 0 && k < count ? bezier(ease, k / count) : null;
      const state = states[el] ??= {};
      for (const prop of Object.keys(from)) {
        const distance = to[prop] - from[prop];
        // Multiplied before dividing: the product is exact, so a linear move rounds true halves the
        // same way every time and its last frame is its target, not a sum of steps.
        const value = from[prop] + (eased === null ? distance * k / count : distance * eased);
        state[prop] = prop === 'x' || prop === 'y' ? whole(value) : Math.round(value * 1000) / 1000 + 0;
      }
    }
    return states;
  }

  // ---- What the browser is asked to do ---------------------------------------------------------

  // A state as style, in keyframes and inline alike. The engine owns transform and opacity of every
  // element it moves.
  function css({ x, y, rotate, opacity }) {
    const transform = [];
    if (x !== undefined || y !== undefined) transform.push(`translate(${x ?? 0}px, ${y ?? 0}px)`);
    if (rotate !== undefined) transform.push(`rotate(${rotate}deg)`);
    return {
      ...(transform.length && { transform: transform.join(' ') }),
      ...(opacity !== undefined && { opacity: String(opacity) }),
    };
  }

  /**
   * A track as the two arguments of element.animate. Each keyframe of a move holds until the next
   * (step-end), and half a tick of negative delay puts every sample of the compositor in the middle
   * of a step: tick k shows frame k of the table, and frame 0 is the state the element rests in.
   */
  function bake(track, fps) {
    const ms = (ticks) => ticks * tickUs(fps) / 1000;
    if (track.loop) {
      return {
        keyframes: [css(track.from), css(track.to)],
        timing: { duration: ms(track.frames), delay: ms(-0.5), iterations: Infinity, easing: `steps(${track.frames})` },
      };
    }
    const keyframes = Array.from({ length: track.frames + 1 }, (_, k) => (
      { ...css(stateAt([track], track.start + k)[track.el]), offset: k / track.frames, easing: 'step-end' }));
    return { keyframes, timing: { duration: ms(track.frames), delay: ms(track.start - 0.5), fill: 'both' } };
  }

  // ---- The demo schedule -----------------------------------------------------------------------

  const VERBS = {
    play: () => ({ type: 'playAction', params: { goto: 0 } }),
    stop: () => ({ type: 'stopAction', params: {} }),
    update: (data) => ({ type: 'updateAction', params: { data } }),
  };

  /**
   * A demo, [[seconds, action, data], ...], as an OGraf actions schedule (timestamps in ms) plus the
   * cycle length in ms: null unless the demo ends with [seconds, 'repeat']. `actions` are the ids of
   * the graphic's custom actions.
   */
  function schedule(cues, actions = []) {
    if (!Array.isArray(cues)) fail('demo', 'must be a list of cues [seconds, action, data]');
    const names = [...Object.keys(VERBS), 'repeat', ...actions];
    const result = { schedule: [], cycle: null };
    let previous = 0;
    cues.forEach((cue, i) => {
      const where = `demo[${i}]`, [seconds, name, data] = Array.isArray(cue) ? cue : [];
      if (typeof seconds !== 'number' || !(seconds >= 0) || typeof name !== 'string') {
        fail(where, `expected [seconds, action, data]; got ${JSON.stringify(cue)}`);
      }
      if (!names.includes(name)) fail(where, `unknown action "${name}" (known: ${names.join(', ')})`);
      if (seconds < previous) fail(where, `cues must be in time order (${seconds} s comes after ${previous} s)`);
      previous = seconds;
      if (name !== 'repeat') {
        const action = Object.hasOwn(VERBS, name) ? VERBS[name](data) : { type: 'customAction', params: { id: name, payload: data } };
        result.schedule.push({ timestamp: Math.round(seconds * 1000), action });
      } else if (i !== cues.length - 1) {
        fail(where, 'repeat must be the last cue');
      } else if (seconds === 0) {
        fail(where, 'repeat needs a cycle longer than 0 s');
      } else {
        result.cycle = Math.round(seconds * 1000);
      }
    });
    return result;
  }

  // Where in its cycle an instance starts, as a fraction of the cycle: golden-ratio steps of the
  // trailing number of its source id (browser_007 is 7) spread any count of windows evenly, so they
  // do not all move, and paint, on the same frames. No number, no shift.
  const phase = (source) => (/\d+$/.exec(source ?? '')?.[0] ?? 0) * 0.6180339887 % 1;

  /**
   * Runs a demo, { schedule, cycle }, on a loaded graphic by calling its OGraf actions, as a
   * controller would. A repeating demo is placed on the wall clock: a cycle starts whenever
   * Date.now() is `fraction` of a cycle (see phase) past a multiple of the cycle. What an instance
   * shows therefore depends on the time and its fraction, not on when its page loaded; cues already
   * behind in the running cycle are applied without animation. One timer waits for the next cue.
   * Returns a function that stops the demo.
   */
  function demo(element, { schedule: cues, cycle }, fraction = 0) {
    if (!cues.length) return () => {};
    const fire = ({ type, params }, skipAnimation) => element[type]({ ...params, skipAnimation });
    const shift = cycle ? fraction * cycle : 0;
    let start = cycle ? Math.floor((Date.now() - shift) / cycle) * cycle + shift : Date.now();
    let next = 0, timer;
    while (next < cues.length && start + cues[next].timestamp < Date.now()) fire(cues[next++].action, true);
    const arm = () => {
      if (next === cues.length) {
        if (!cycle) return;
        start += cycle;
        next = 0;
      }
      const due = cues[next].timestamp;
      timer = setTimeout(() => {
        // Cues due together run in one task, so no frame is painted between them.
        while (next < cues.length && cues[next].timestamp === due) fire(cues[next++].action, false);
        arm();
      }, start + due - Date.now());
    };
    arm();
    return () => clearTimeout(timer);
  }

  // ---- The graphic -----------------------------------------------------------------------------

  const KEYS = ['html', 'css', 'data', 'update', 'play', 'stop', 'actions', 'loop', 'every', 'demo'];
  // resolve() against this box checks a declaration when the page loads, before any element exists.
  const DRY_BOX = { width: 100, height: 100, vw: 100, vh: 100, content: 100 };
  // The graphic fills the window (or whatever an OGraf renderer positions it in).
  const BASE_CSS = ':host{position:absolute;inset:0;display:block}';

  /**
   * Declares a graphic (README.md lists the keys) and returns its custom element class, an EBU OGraf
   * graphic: load, playAction, stopAction, updateAction, customAction, goToTime, setActionsSchedule
   * and dispose. The class is not registered here: its renderer, host() on our pages, names it.
   * A wrong declaration throws now, when the page loads.
   */
  function graphic(declaration) {
    known(declaration, KEYS, 'graphic');
    const { html, css: style = '', data: defaults = {}, update = () => {}, play = [], stop = [],
      actions = {}, loop = [], every = [], demo: cues } = declaration;
    if (typeof html !== 'string' || !html) fail('html', 'must be the markup of the graphic, a string');
    if (typeof update !== 'function') fail('update', 'must be a function (el, data)');
    // Every list of tracks, under the name errors use for it: play, stop, loop and actions.<id>.
    const lists = { play, stop, loop };
    for (const [id, moves] of Object.entries(actions)) {
      if ([...Object.keys(lists), 'update', 'repeat'].includes(id)) fail(`actions.${id}`, 'the name is taken; choose another id');
      lists[`actions.${id}`] = moves;
    }
    for (const [name, specs] of Object.entries(lists)) {
      if (!Array.isArray(specs)) fail(name, 'must be a list');
      specs.forEach((spec, i) => {
        const where = `${name}[${i}]`, track = resolve(spec, 50, DRY_BOX, where);
        if (name === 'loop' && !track.loop) fail(where, 'a loop is a crawl or a spin; a move belongs in play, stop or actions');
        if (name !== 'loop' && track.loop) fail(where, 'a spin or crawl belongs in loop');
        const earlier = specs.findIndex((other) => other.el === spec.el);
        if (earlier < i) fail(`${where}.el`, `"${spec.el}" is already moved by ${name}[${earlier}]; one move per element`);
        if (name !== 'loop' && Array.isArray(loop) && loop.some((other) => other.el === spec.el)
            && ['x', 'y', 'rotate'].some((prop) => prop in track.from)) {
          fail(`${where}.el`, `"${spec.el}" is turned or crawled by loop, which owns its transform; move a wrapper element`);
        }
      });
    }
    if (!Array.isArray(every)) fail('every', 'must be a list of { seconds, run }');
    every.forEach(({ seconds, run } = {}, i) => {
      positive(seconds, `every[${i}].seconds`);
      if (typeof run !== 'function') fail(`every[${i}].run`, 'must be a function (el, n)');
    });

    return class extends HTMLElement {
      static demo = cues === undefined ? null : schedule(cues, Object.keys(actions));

      #fps;                     // set by load(), with the next four
      #els;                     // id → element, for every element of html that has an id
      #loaded;                  // the data of load() over the declared defaults
      #data;                    // #loaded with every update since
      #tracks;                  // name of a list → its tracks, as last measured
      #step;                    // 0 while played; undefined before play and after stop
      #turn = 0;                // counts actions: a stop overtaken by a play must not end the loops
      #moving = new Map();      // element → the animation of its move in flight
      #looping = new Map();     // id → { key, animation } of its running loop
      #timers = [];             // one pending timeout per `every`, while played
      #schedule = [];           // setActionsSchedule(), for goToTime()

      // The measured tracks; an action before load() has no elements to act on.
      #loadedTracks() {
        if (!this.#tracks) throw new Error('Motion: load() has not finished');
        return this.#tracks;
      }

      async load({ data, renderCharacteristics: { frameRate } = {} } = {}) {
        if (!(frameRate > 0)) throw new Error('Motion: load() needs renderCharacteristics.frameRate (the page shell passes data-fps)');
        this.#halt();
        this.style.visibility = 'hidden';   // nothing is painted before the start state is in place
        const root = this.shadowRoot ?? this.attachShadow({ mode: 'open' });
        root.innerHTML = `<style>${BASE_CSS}${style}</style>${html}`;
        const els = Object.fromEntries(Array.from(root.querySelectorAll('[id]'), (element) => [element.id, element]));
        for (const [name, specs] of Object.entries(lists)) {
          specs.forEach(({ el }, i) => {
            if (!els[el]) fail(`${name}[${i}].el`, `no element with id "${el}" in html (it has: ${Object.keys(els).join(', ')})`);
            // Its own compositor layer for good: starting or ending a move then rasters nothing.
            els[el].style.willChange = 'transform, opacity';
          });
        }
        this.#fps = frameRate;
        this.#els = els;
        this.#loaded = { ...defaults, ...data };
        this.#data = { ...this.#loaded };
        this.#step = undefined;
        update(els, this.#data);
        await document.fonts.ready;   // text has its final width before anything is measured
        this.#measure();
        this.#write(stateAt(this.#tracks.play, 0));   // rest where play starts
        this.style.visibility = '';
        return { statusCode: 200 };
      }

      async dispose() {
        this.#halt();
        this.#step = undefined;
        this.#tracks = undefined;
        if (this.shadowRoot) this.shadowRoot.innerHTML = '';
        return { statusCode: 200 };
      }

      async updateAction({ data } = {}) {
        this.#loadedTracks();
        Object.assign(this.#data, data);
        update(this.#els, this.#data);
        this.#measure();
        if (this.#step === 0) this.#on();   // a loop whose measure changed starts again
        return { statusCode: 200 };
      }

      async playAction(params = {}) {
        const plan = this.#plan('playAction', params);
        await this.#run(plan, params.skipAnimation);
        return { statusCode: 200, currentStep: plan.step };
      }

      async stopAction(params = {}) {
        await this.#run(this.#plan('stopAction'), params.skipAnimation);
        return { statusCode: 200 };
      }

      async customAction(params = {}) {
        const plan = this.#plan('customAction', params);
        if (!Object.hasOwn(this.#loadedTracks(), plan.list)) {
          const ids = Object.keys(actions).join(', ') || 'none';
          return { statusCode: 404, statusMessage: `Motion: no custom action "${params.id}"; this graphic has: ${ids}` };
        }
        await this.#run(plan, params.skipAnimation);
        return { statusCode: 200 };
      }

      async setActionsSchedule({ schedule: scheduled = [] } = {}) {
        this.#schedule = [...scheduled].sort((a, b) => a.timestamp - b.timestamp);   // stable: order at one time is kept
        return { statusCode: 200 };
      }

      // Non-real-time: nothing runs. The frame at `timestamp` (ms) is read from the frame table for
      // the actions of setActionsSchedule() that have started by then. `every` is real-time only.
      async goToTime({ timestamp }) {
        this.#loadedTracks();
        this.#halt();
        const frame = (since) => Math.round((timestamp - since) * this.#fps / 1000);
        const started = new Map();   // list → the frame it has reached, in the order the lists last ran
        let played;                  // when play last took the graphic from rest
        this.#step = undefined;
        this.#data = { ...this.#loaded };
        for (const { timestamp: at, action: { type, params = {} } } of this.#schedule) {
          if (at > timestamp) break;
          if (type === 'updateAction') {
            Object.assign(this.#data, params.data);
            continue;
          }
          const { list, step } = this.#plan(type, params);
          if (step === 0 && this.#step !== 0) played = at;
          this.#step = step;
          started.delete(list);
          started.set(list, params.skipAnimation ? Infinity : frame(at));
        }
        update(this.#els, this.#data);
        this.#measure();
        // As in real time, loops run from play until the stop move has ended.
        const stopFrames = Math.max(0, ...this.#tracks.stop.map((track) => track.start + track.frames));
        const looping = played !== undefined && (this.#step === 0 || started.get('stop') < stopFrames);
        this.#write(stateAt(this.#tracks.loop, looping ? frame(played) : 0));
        this.#write(stateAt(this.#tracks.play, 0));
        for (const [list, n] of started) this.#write(stateAt(this.#tracks[list] ?? [], n));
        return { statusCode: 200 };
      }

      // The list an action plays and the step it leaves the graphic on. OGraf's step model with one
      // step: step 0 is "played", and a target past it is the end, which is what stop is.
      #plan(type, { goto, delta = 1, id } = {}) {
        if (type === 'customAction') return { list: `actions.${id}`, step: this.#step };
        const target = type === 'stopAction' ? 1 : Number.isInteger(goto) && goto >= 0 ? goto : (this.#step ?? -1) + delta;
        return target >= 1 ? { list: 'stop', step: undefined } : { list: 'play', step: 0 };
      }

      async #run({ list, step }, skipAnimation) {
        const tracks = this.#loadedTracks()[list], turn = ++this.#turn;
        this.#step = step;
        if (step === 0) this.#on();
        await this.#act(tracks, skipAnimation);
        if (this.#step === undefined && turn === this.#turn) this.#off();
      }

      // Declared lengths to whole pixels and seconds to whole frames, for the content as it is now.
      #measure() {
        const view = { vw: innerWidth, vh: innerHeight };
        const measured = (name) => lists[name].map((spec, i) => {
          const element = this.#els[spec.el], where = `${name}[${i}]`;
          if (!spec.crawl) return resolve(spec, this.#fps, { ...view, width: element.offsetWidth, height: element.offsetHeight }, where);
          // A crawl wraps by one copy of its content. The element's children are those copies; each
          // is given the loop's length, so the wrap lands on the step grid.
          const copies = Array.from(element.children);
          if (copies.length < 2) fail(`${where}.el`, `a crawl needs at least two children in "${spec.el}", each a copy of the content`);
          for (const copy of copies) copy.style.width = '';
          const track = resolve(spec, this.#fps, { content: copies[0].getBoundingClientRect().width }, where);
          for (const copy of copies) copy.style.width = `${-track.to.x}px`;
          return track;
        });
        this.#tracks = Object.fromEntries(Object.keys(lists).map((name) => [name, measured(name)]));
      }

      #write(states) {
        for (const [id, state] of Object.entries(states)) Object.assign(this.#els[id].style, css(state));
      }

      #animate(track) {
        const { keyframes, timing } = bake(track, this.#fps);
        const animation = this.#els[track.el].animate(keyframes, timing);
        animation.finished.catch(() => {});   // a cancel rejects it; that is no error here
        return animation;
      }

      // Plays moves. An element's move in flight is cancelled, not finished: the new move starts
      // from its own declared start. Resolves when every move has ended or was taken over.
      #act(tracks, skipAnimation) {
        return Promise.all(tracks.map((track) => {
          const element = this.#els[track.el], end = css(stateAt([track], Infinity)[track.el]);
          this.#moving.get(element)?.cancel();
          this.#moving.delete(element);
          if (skipAnimation) {
            Object.assign(element.style, end);
            return undefined;
          }
          const animation = this.#animate(track);
          this.#moving.set(element, animation);
          return animation.finished.then(() => {
            if (this.#moving.get(element) !== animation) return;
            Object.assign(element.style, end);   // commit, then cancel: at rest nothing runs
            animation.cancel();
            this.#moving.delete(element);
          }, () => {});
        }));
      }

      // Loops and timed changes live only while the graphic is played.
      #on() {
        for (const track of this.#tracks.loop) {
          const key = JSON.stringify(track), running = this.#looping.get(track.el);
          if (running?.key === key) continue;
          running?.animation.cancel();
          this.#looping.set(track.el, { key, animation: this.#animate(track) });
        }
        every.forEach(({ seconds, run }, i) => {
          if (this.#timers[i] !== undefined) return;
          const period = seconds * 1000;
          // n counts periods of the wall clock, so a change lands on its boundary (a clock on the
          // second) whenever the page loaded. A timer may fire a hair off its boundary: round.
          const tick = (n) => {
            run(this.#els, n);
            this.#timers[i] = setTimeout(() => tick(Math.round(Date.now() / period)), (n + 1) * period - Date.now());
          };
          tick(Math.floor(Date.now() / period));
        });
      }

      #off() {
        for (const { animation } of this.#looping.values()) animation.cancel();
        this.#looping.clear();
        for (const timer of this.#timers) clearTimeout(timer);
        this.#timers = [];
      }

      #halt() {
        for (const animation of this.#moving.values()) animation.cancel();
        this.#moving.clear();
        this.#off();
      }
    };
  }

  /**
   * The page shell's one call: mounts the graphic in the body, loads it with the data-* attributes of
   * <html> (data-fps is the window's frame rate, the others are the graphic's data) and runs its demo.
   */
  async function host(Graphic) {
    const { fps, ...data } = document.documentElement.dataset;
    customElements.define('motion-graphic', Graphic);
    const element = document.body.appendChild(document.createElement('motion-graphic'));
    await element.load({ data, renderType: 'realtime', renderCharacteristics: { frameRate: Number(fps) } });
    if (Graphic.demo) demo(element, Graphic.demo, phase(data.source));
    return element;
  }

  return { tickUs, frames, resolve, stateAt, bake, schedule, phase, graphic, demo, host };
})();
