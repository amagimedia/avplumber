// A page without a browser. The engine's text runs as graphic_pages.py inlines it (motion.js, then a
// graphic, then the shell's boot line), against fakes that only record what it asks of the browser:
// which keyframes and timing reach element.animate, what is written to inline style, which
// animations are cancelled and which timers are pending.
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

export const RATES = [25, 30, 50, 60];
export const read = (name) => readFileSync(new URL(`../${name}`, import.meta.url), 'utf8');
/** The inline scripts of host.html, in order: the two markers, then the line that mounts the graphic. */
export const shellScripts = () => Array.from(read('host.html').matchAll(/<script>([\s\S]*?)<\/script>/g), ([, text]) => text);
// Lets promise reactions run: an await inside the engine continues before the test looks again.
export const settle = () => new Promise((resolve) => setImmediate(resolve));

class Clock {
  #timers = new Map();
  #ids = 0;
  constructor(now) { this.now = now; }
  setTimeout = (run, delay = 0) => {
    this.#timers.set(++this.#ids, { at: this.now + Math.max(0, delay), run });
    return this.#ids;
  };
  clearTimeout = (id) => { this.#timers.delete(id); };
  /** Milliseconds until each pending timer, soonest first. */
  get pending() { return [...this.#timers.values()].map((timer) => timer.at - this.now).sort((a, b) => a - b); }
  async advance(ms) {
    const end = this.now + ms;
    for (;;) {
      const due = [...this.#timers].filter(([, timer]) => timer.at <= end)
        .sort(([idA, a], [idB, b]) => a.at - b.at || idA - idB)[0];
      if (!due) break;
      this.#timers.delete(due[0]);
      this.now = due[1].at;
      due[1].run();
      await settle();
    }
    this.now = end;
    await settle();
  }
}

class FakeAnimation {
  cancelled = false;
  #settled = false;
  #resolve;
  #reject;
  constructor(keyframes, timing, startedAt) {
    this.keyframes = keyframes;
    this.timing = timing;
    /** When a browser would reach the end: never for a loop. */
    this.endsAt = startedAt + (timing.delay ?? 0) + timing.duration * (timing.iterations ?? 1);
    this.finished = new Promise((resolve, reject) => { this.#resolve = resolve; this.#reject = reject; });
  }
  get settled() { return this.#settled; }
  /** The browser reaching the end of the animation. */
  finish() { this.#settled = true; this.#resolve(this); }
  cancel() {
    this.cancelled = true;
    if (this.#settled) return;
    this.#settled = true;
    this.#reject(Object.assign(new Error('The user aborted a request.'), { name: 'AbortError' }));
  }
}

// layout[id]: { width, height } is the element's own box; { copies, content } gives it that many
// children, each `content` px wide until the engine sets a width on it.
class FakeElement {
  style = {};
  animations = [];
  textContent = '';
  #clock;
  constructor(id, layout = {}, clock) {
    this.id = id;
    this.#clock = clock;
    this.offsetWidth = layout.width ?? 0;
    this.offsetHeight = layout.height ?? 0;
    this.children = Array.from({ length: layout.copies ?? 0 }, () => ({
      style: {},
      getBoundingClientRect() { return { width: this.style.width ? parseFloat(this.style.width) : layout.content }; },
    }));
  }
  animate(keyframes, timing) {
    const animation = new FakeAnimation(keyframes, timing, this.#clock.now);
    this.animations.push(animation);
    return animation;
  }
  get running() { return this.animations.filter((animation) => !animation.cancelled); }
}

/**
 * graphic and boot are script texts placed after motion.js, as in the linked page.
 * Returns Motion, the `graphic` that text declared (if any), and the fakes.
 */
export function page({ graphic = '', boot = '', dataset = {}, layout = {}, width = 1920, height = 1080, now = 0,
  fontsReady = Promise.resolve() } = {}) {
  const clock = new Clock(now);
  // The real Date on the fake clock: Date.now() and new Date() read it, new Date(ms) is as ever.
  class PageDate extends Date {
    constructor(...args) { super(...(args.length ? args : [clock.now])); }
    static now() { return clock.now; }
  }
  const registry = new Map();
  const errors = [];
  class HTMLElement {
    style = {};
    shadowRoot = null;
    attachShadow() {
      const elements = [];
      this.shadowRoot = {
        get innerHTML() { return this.html; },
        set innerHTML(html) {
          this.html = html;
          elements.splice(0, Infinity, ...Array.from(html.matchAll(/\bid="([^"]+)"/g), ([, id]) => new FakeElement(id, layout[id], clock)));
        },
        querySelectorAll: () => elements,
      };
      return this.shadowRoot;
    }
    /** The fake elements of the graphic, by id. */
    get parts() { return Object.fromEntries(this.shadowRoot.querySelectorAll('[id]').map((element) => [element.id, element])); }
  }
  const document = {
    documentElement: { dataset },
    body: { children: [], appendChild(element) { this.children.push(element); return element; } },
    createElement: (tag) => new (registry.get(tag))(),
    fonts: { ready: fontsReady },
  };
  const globals = {
    HTMLElement, document, innerWidth: width, innerHeight: height,
    customElements: { define(tag, constructor) { if (registry.has(tag)) throw new Error(`${tag} defined twice`); registry.set(tag, constructor); } },
    setTimeout: clock.setTimeout, clearTimeout: clock.clearTimeout, Date: PageDate,
    console: { error: (...args) => errors.push(args.join(' ')) },
  };
  const link = vm.compileFunction(
    `${read('motion.js')}\n${graphic}\n${boot}\n`
    + "return { Motion, graphic: typeof graphic === 'undefined' ? undefined : graphic };",
    Object.keys(globals), { filename: 'motion.js' });
  return { ...link(...Object.values(globals)), clock, document, registry, errors, layout };
}

/** A loaded instance of a declared graphic: { element, parts, Motion, clock, layout }. */
export async function mount(declaration, { fps = 50, data, ...options } = {}) {
  const fake = page(options);
  const element = new (fake.Motion.graphic(declaration))();
  await element.load({ data, renderType: 'realtime', renderCharacteristics: { frameRate: fps } });
  return { ...fake, element, parts: element.parts };
}

/**
 * Lets `ms` pass as a browser would: timers fire, and each move of `element` reaches its end when
 * its last frame has been shown. Loops run on.
 */
export async function elapse({ clock, element }, ms) {
  const end = clock.now + ms;
  do {
    const moves = Object.values(element.parts).flatMap((part) => part.running).filter((animation) => !animation.settled);
    const next = Math.min(end, clock.now + (clock.pending[0] ?? Infinity), ...moves.map((animation) => animation.endsAt));
    await clock.advance(next - clock.now);
    for (const animation of moves) if (animation.endsAt <= clock.now && !animation.cancelled) animation.finish();
    await settle();
  } while (clock.now < end);
}

// Small declarations that pin the engine's contract in the other test files. The graphics that ship
// live beside the engine, one directory each; graphics.test.mjs runs those.
const PEOPLE = [{ name: 'Ada Lovelace', role: 'Analyst · Engine No. 1' }, { name: 'Grace Hopper', role: 'Compiler desk' }];
export const LOWER_THIRD = {
  html: '<div id="plate"><div id="accent"></div><div id="text"><div id="name"></div><div id="role"></div></div></div>',
  css: '#plate { position: absolute; left: 0; top: 12%; width: 100%; height: 76%; display: flex; }',
  data: PEOPLE[0],
  update: (el, data) => { el.name.textContent = data.name; el.role.textContent = data.role; },
  play: [{ el: 'plate', x: ['-104%', 0], opacity: [0, 1], seconds: 0.6, ease: [0.2, 0.8, 0.2, 1] }],
  stop: [{ el: 'plate', x: [0, '-104%'], opacity: [1, 0], seconds: 0.4, ease: 'in' }],
  demo: [...PEOPLE.flatMap((person, i) => [[6 * i, 'update', person], [6 * i, 'play'], [6 * i + 5.2, 'stop']]), [12, 'repeat']],
};
export const LOWER_THIRD_LAYOUT = { plate: { width: 1016, height: 131 } };

export const TICKER = {
  html: '<div id="tag">LIVE</div><div id="band"><div id="crawl"><span></span><span></span></div></div>',
  loop: [{ el: 'crawl', crawl: { pxPerSecond: 110 } }],
  demo: [[0, 'play']],
};
export const TICKER_LAYOUT = { crawl: { copies: 2, content: 3999.2 } };

export const ALPHA = {
  html: '<header id="title"></header><div id="marker"></div>',
  data: { source: '' },
  update: (el, data) => { el.title.textContent = `${data.source} · alpha over video`; },
  actions: {
    right: [{ el: 'marker', x: [0, '82vw'], seconds: 2 }],
    left: [{ el: 'marker', x: ['82vw', 0], seconds: 2 }],
  },
  demo: [[0, 'right'], [6, 'left'], [12, 'repeat']],
};
