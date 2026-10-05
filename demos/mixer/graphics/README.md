# Mixer graphics

One motion engine for every HTML graphic the mixer shows in a browser window, test source or
downstream key. A graphic declares what moves; [motion.js](motion.js) plays it.

## How a graphic moves

- **Whole frames.** The browser ticks every `floor(1e6 / fps)` µs (16 666 at 60 fps). A duration is
  `max(1, round(seconds × fps))` ticks, so a move ends on a tick.
- **Whole pixels.** Lengths in `%`, `vw` or `vh` become pixels when the graphic loads and after each
  update; every frame of every move, eased or not, lands on a whole pixel.
- **Played by the compositor.** The engine bakes one step keyframe per frame and hands them to
  `element.animate`. No script runs per frame.
- **Rest costs nothing.** When a move ends its end state is written to inline style and the animation
  is cancelled. A resting page holds one timer, to its next cue.
- **Loops** run only while the graphic is played. A crawl moves `max(1, round(pxPerSecond / fps))`
  pixels per frame, the same on every frame (110 px/s is 4, 4, 2, 2 px at 25, 30, 50, 60 fps), and
  wraps on that grid.

Whole pixels reach the program only where the window maps 1:1 onto the canvas.

## Files

| File | Role |
| --- | --- |
| [motion.js](motion.js) | The engine. One global, `Motion`; no imports. |
| [host.html](host.html) | The one page shell: transparent body, the engine, one graphic, `Motion.host(graphic)`. |
| `<name>/graphic.js` | One graphic: `const graphic = Motion.graphic({...});` |
| `<name>/<name>.ograf.json` | Its EBU OGraf manifest. |
| [tests/](tests) | `node --test` suite; runs the engine's text against a fake page. |
| [../graphic_pages.py](../graphic_pages.py) | `graphic_url(name, fps, **data)`: links shell, engine and graphic into a `data:` URL. |

## Add a graphic

`graphics/score_bug/graphic.js`, complete:

```js
const graphic = Motion.graphic({
  html: '<div id="plate"><b id="home"></b><span id="score"></span><b id="away"></b></div>',
  css: ':host { color: white; font: 700 40vh system-ui, sans-serif; }'
     + '#plate { position: absolute; inset: 10% 0; display: flex; gap: 20vh; background: #101c2bee; }',
  data: { home: 'HOME', away: 'AWAY', score: '0 : 0' },
  update: (el, data) => { for (const id of ['home', 'score', 'away']) el[id].textContent = data[id]; },
  play: [{ el: 'plate', y: ['-110%', 0], opacity: [0, 1], seconds: 0.5, ease: 'out' }],
  stop: [{ el: 'plate', y: [0, '-110%'], opacity: [1, 0], seconds: 0.3, ease: 'in' }],
  demo: [[0, 'update', { score: '0 : 0' }], [0, 'play'], [8, 'update', { score: '1 : 0' }], [14, 'stop'], [16, 'repeat']],
});
```

Then:

1. Write `graphics/score_bug/score_bug.ograf.json`. The
   [OGraf schema](https://ograf.ebu.io/v1/specification/json-schemas/graphics/schema.json) requires
   `$schema`, `id`, `name`, `main`, `supportsRealTime` and `supportsNonRealTime`; `schema` describes
   `data`. It rejects unknown fields unless they start with `v_`, so what only the mixer needs
   (window size, canvas placement) goes under `v_avplumber`.
2. Allow the window size in `DMA_BROWSER_ALLOWED_DIMS` of [../compose.yaml](../compose.yaml).
3. Get its page with `graphic_url("score_bug", fps, source="…")` from
   [../graphic_pages.py](../graphic_pages.py). `fps` is the rate the window paints at; the keywords
   become the graphic's `data`, as strings.

A wrong declaration throws when the page loads, naming the place (`Motion: play[0].x: expected
[from, to], …`); the browser service logs the page's console.

## Declaration

| Key | Value |
| --- | --- |
| `html` | The markup. Every element the graphic refers to has an `id`. |
| `css` | Its styles. They and the ids are private to the graphic (a shadow root): style the root as `:host`, not `body`. |
| `data` | Default data. `load()` and updates merge over it. |
| `update(el, data)` | Writes data into the page; `el` maps each id to its element. Runs at load and on every update. |
| `play`, `stop` | Lists of moves for `playAction` and `stopAction`. A loaded graphic rests where its `play` moves start. |
| `actions` | `{ id: [moves] }`: custom actions. |
| `loop` | `{ el, crawl: { pxPerSecond } }` or `{ el, spin: { seconds } }` (one clockwise turn). The children of a crawl element are two or more copies of its content; the engine sizes them. |
| `every` | `{ seconds, run(el, n) }`: a timed change, such as a clock's text. It runs when the graphic is played and then on each multiple of `seconds` of the wall clock; `n` counts those multiples. A timer, not an animation; changes that must land on one frame belong in one `every`. |
| `demo` | `[[seconds, action, data], …]`: what the page does on its own. `action` is `play`, `stop`, `update` (with `data`) or a custom action's id; a last cue `[seconds, 'repeat']` makes it a cycle. |

A move is `{ el, x, y, opacity, rotate, seconds, delay, ease }`:

- `x`, `y`: `[from, to]` lengths. A number is pixels; `'-104%'` is of the element's own width (`x`)
  or height (`y`), as in CSS `translate`; `'82vw'` and `'10vh'` are of the window.
- `opacity`: `[from, to]`, 0 to 1. `rotate`: `[from, to]` in degrees.
- `seconds`: duration. `delay`: seconds before it starts, default 0.
- `ease`: `'linear'` (default), `'in'`, `'out'`, `'in-out'` (the CSS curves), or `[x1, y1, x2, y2]`
  as in `cubic-bezier()`.

## Rules for a graphic

- All motion goes through `play`, `stop`, `actions` and `loop`; all timed changes through `every`.
  No `requestAnimationFrame`, `setInterval`, CSS `animation` or `transition` in a graphic: each one
  is a second clock, and an animation left running makes the window paint when nothing changes.
- The engine owns `transform`, `opacity` and `will-change` of every element named in a move or loop.
  Put a layout transform on a wrapper.
- A move that names `x`, `y` or `rotate` rewrites the element's whole `transform`, so an axis it
  leaves out returns to 0. Give each such move of an element every axis that element ever moves on
  (`y: [40, 40]` keeps it at 40). The engine does not check this.
- One move per element in a list, and an element that loops is not also moved: wrap it.
- A new action takes over an element whose move is still in flight; the new move starts from its
  own `from`.
- A cycle of a repeating demo must end as it began. Cycles are placed on the wall clock, so a page
  that loads mid-cycle applies the cues already behind it, without animation, and continues; what a
  window shows depends on the time, not on when its page loaded.
- Many windows of one graphic do not move at once: a page whose `data-source` ends in a number
  starts its cycle that many golden-ratio steps (0.618 of a cycle each) later.

## OGraf

`Motion.graphic()` returns a custom element class with the methods of an
[EBU OGraf](https://ograf.ebu.io/) graphic: `load`, `playAction`, `stopAction`, `updateAction`,
`customAction`, `goToTime`, `setActionsSchedule`, `dispose`. Each resolves to `{ statusCode }`;
`playAction` adds `currentStep` (a graphic has one step: `0` while played, `undefined` at rest).

- The demo is an OGraf actions schedule (`Graphic.demo`), run by `Motion.host` through those methods.
  A controller can call the same methods; nothing reaches into a loaded page yet.
- `load()` needs `renderCharacteristics.frameRate`. On our pages that is `data-fps`.
- `goToTime({ timestamp })` cancels whatever runs and shows the frame of the `setActionsSchedule()`
  schedule at that time, read from the same frame table. `every` is not part of it.
- The class is not registered and not exported: the renderer names the element. An OGraf package's
  `main` is the text of `motion.js`, then `graphic.js`, then `export default graphic;`. No tool
  writes that file yet, and no third-party renderer has been tried.

## Tests

```sh
node --test 'demos/mixer/graphics/tests/*.test.mjs'
python3 -m pytest demos/mixer/tests/test_graphic_pages.py
```

The node tests pin frame counts and pixel positions at 25, 30, 50 and 60 fps, the keyframes and timing
passed to `element.animate`, rest, the demo schedule and the OGraf surface. They cannot show what
Chromium does with those keyframes; that needs a browser window.
