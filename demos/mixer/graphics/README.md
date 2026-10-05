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
| `<name>/<name>.ograf.json` | Its EBU OGraf manifest, with what the mixer needs under `v_avplumber`. |
| [tests/](tests) | `node --test` suite; runs the engine and every graphic against a fake page. |
| [../graphic_pages.py](../graphic_pages.py) | Links shell, engine and graphic into a `data:` URL; reads the manifests. |

The graphics here:

| Graphic | Shows | Motion |
| --- | --- | --- |
| [browser_alpha](browser_alpha/graphic.js) | Transparency test source, one window per source | Marker slides 2 s, rests 4 s |
| [lower_third](lower_third/graphic.js) | Name plate key | In 0.6 s, out 0.4 s, a new name every 6 s |
| [ticker](ticker/graphic.js) | Crawl strip key | Crawl, 110 px/s |
| [bug_left](bug_left/graphic.js) | Corner bug key | Ring turns once in 4 s |
| [bug_right](bug_right/graphic.js) | Clock bug key | None: a timer changes dot and clock twice a second |

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

`graphics/score_bug/score_bug.ograf.json`, complete:

```json
{
  "$schema": "https://ograf.ebu.io/v1/specification/json-schemas/graphics/schema.json",
  "id": "io.github.amagimedia.avplumber.score_bug",
  "name": "Score bug",
  "description": "above the lower third",
  "main": "graphic.js",
  "supportsRealTime": true,
  "supportsNonRealTime": true,
  "schema": {"type": "object", "properties": {
    "home": {"type": "string"}, "away": {"type": "string"}, "score": {"type": "string"}
  }},
  "v_avplumber": {
    "window": {"width": 640, "height": 96},
    "key": {"order": 5, "anchor": "bottom-right", "above": "lower_third"}
  }
}
```

Then allow the window size, `640x96`, in `DMA_BROWSER_ALLOWED_DIMS` of
[../compose.yaml](../compose.yaml). That is all: the setup page lists the key, a recipe's `dsk` list
accepts `score_bug`, [../prepare_demo.py](../prepare_demo.py) opens its window and places it, and
the tests under [tests/](tests) and [../tests/test_graphic_pages.py](../tests/test_graphic_pages.py)
run it at 25, 30, 50 and 60 fps without a line added to them.

A wrong declaration throws when the page loads, naming the place (`Motion: play[0].x: expected
[from, to], …`); the browser service logs the page's console. A wrong manifest stops the setup
server and `prepare_demo.py` when they start, naming its place the same way
(`score_bug/score_bug.ograf.json: v_avplumber.key.anchor: must be one of …`).

## Declaration

| Key | Value |
| --- | --- |
| `html` | The markup. Every element the graphic refers to has an `id`. |
| `css` | Its styles. They and the ids are private to the graphic (a shadow root): style the root as `:host`, not `body`. |
| `data` | Default data. `load()` and updates merge over it. On our pages the `data-*` attributes of `<html>` arrive as data, as strings: `source` is the id of the browser source showing the page. |
| `update(el, data)` | Writes data into the page; `el` maps each id to its element. Runs at load and on every update. |
| `play`, `stop` | Lists of moves for `playAction` and `stopAction`. A loaded graphic rests where its `play` moves start. |
| `actions` | `{ id: [moves] }`: custom actions. |
| `loop` | `{ el, crawl: { pxPerSecond } }` or `{ el, spin: { seconds } }` (one clockwise turn). The children of a crawl element are two or more copies of its content; the engine sizes them. |
| `every` | `{ seconds, run(el, n) }`: a timed change, such as a clock's text. It runs when the graphic is played and then on each multiple of `seconds` of the wall clock; `n` counts those multiples. A timer, not an animation; changes that must land on one frame belong in one `every`. |
| `demo` | `[[seconds, action, data], …]`: what the page does on its own. `action` is `play`, `stop`, `update` (with `data`) or a custom action's id; a last cue `[seconds, 'repeat']` makes it a cycle. |
| `stagger` | Seconds over which many windows of the graphic spread the starts of their cycles; by default the whole cycle. Shorter when parts of the cycle move alike: the test source's 12 s cycle is two like halves, so its windows spread over 6. |

A move is `{ el, x, y, opacity, rotate, seconds, delay, ease }`:

- `x`, `y`: `[from, to]` lengths. A number is pixels; `'-104%'` is of the element's own width (`x`)
  or height (`y`), as in CSS `translate`; `'82vw'` and `'10vh'` are of the window.
- `opacity`: `[from, to]`, 0 to 1. `rotate`: `[from, to]` in degrees.
- `seconds`: duration. `delay`: seconds before it starts, default 0.
- `ease`: `'linear'` (default), `'in'`, `'out'`, `'in-out'` (the CSS curves), or `[x1, y1, x2, y2]`
  as in `cubic-bezier()`.

## Manifest

`<name>.ograf.json` follows the
[OGraf schema](https://ograf.ebu.io/v1/specification/json-schemas/graphics/schema.json): it requires
`$schema`, `id`, `name`, `main`, `supportsRealTime` and `supportsNonRealTime`, and rejects any field
of its own it does not define unless the name starts with `v_`. `main` is `graphic.js`; `schema`
describes `data`; `customActions` lists the ids in `actions`; `supportsNonRealTime` is `false` for a
graphic with an `every`. What only the mixer needs is under `v_avplumber`:

| Field | Value |
| --- | --- |
| `window` | `{ "width", "height" }` of the browser window, the graphic's own rectangle. Without it the window has the size of its source, by default the canvas. |
| `key` | Makes the graphic a downstream key the setup page offers, labelled `name · description`. Needs `window`. A show keys at most four at once. |
| `key.order` | Position in the setup page's list, lowest first. |
| `key.anchor` | `top-left`, `top-right`, `bottom-left` or `bottom-right`: the window, scaled by the canvas's short side over 1080, sits one margin (3 % of the short side) inside that corner. `top` or `bottom`: a strip scaled to span the canvas width. |
| `key.above` | Name of another key graphic: this one sits one margin above that one's place, whether or not the show keys it. Bottom anchors only. |

## Rules for a graphic

- All motion goes through `play`, `stop`, `actions` and `loop`; all timed changes through `every`.
  No `requestAnimationFrame`, `setInterval`, `setTimeout`, CSS `animation` or `transition` in a
  graphic: each one is a second clock, and an animation left running makes the window paint when
  nothing changes. The tests refuse them.
- A graphic loads nothing at run time: no `src`, `href`, `fetch` or `url()` other than `data:`.
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
  starts its cycle that many golden-ratio steps (0.618 of the `stagger` each) later.

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
- The class is not registered and not exported: the renderer names the element. `graphic.js` alone is
  therefore not yet what an OGraf renderer imports: its module is the text of `motion.js`, then
  `graphic.js`, then `export default graphic;`. No tool writes that file yet, and no third-party
  renderer has been tried.

## Tests

```sh
node --test 'demos/mixer/graphics/tests/*.test.mjs'
python3 -m pytest demos/mixer/tests/test_graphic_pages.py
```

The node tests pin frame counts and pixel positions at 25, 30, 50 and 60 fps, the keyframes and timing
passed to `element.animate`, rest, the demo schedule and the OGraf surface, and run every graphic as
its page runs it. The pytest file covers linking, the manifests, key placement and the window
allowlist. Neither can show what Chromium does with those keyframes; that needs a browser window.
