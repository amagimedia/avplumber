// Transparency test source: patches of white, red, green and blue at 0 to 100 % opacity over
// whatever lies behind the window, and a half-transparent marker. Like a real graphic the marker
// rests most of the time: it slides 2 s and rests 4 s, so the window paints on a third of its ticks.
const graphic = Motion.graphic({
  html: '<header id="title"></header><div id="patches">'
    + [0, 25, 50, 75, 100].map((percent) => (
      `<div class="column"><div class="swatch" style="opacity: ${percent / 100}"></div><div class="label">${percent}%</div></div>`)).join('')
    + '</div><div id="marker"></div>',
  css: `
    :host { color: white; font: 2.5vw system-ui, sans-serif; }
    header { position: absolute; top: 5%; left: 5%; padding: .4em .7em; background: #172536cc; }
    #patches { position: absolute; left: 5%; top: 30%; width: 90%; height: 40%; display: flex; gap: 2%; }
    .column { flex: 1; position: relative; }
    .swatch { height: 80%; background: linear-gradient(to bottom, white 0% 25%, red 25% 50%, lime 50% 75%, blue 75%); }
    .label { position: absolute; top: 85%; width: 100%; text-align: center; text-shadow: 0 1px 3px black; }
    #marker { position: absolute; left: 5%; bottom: 8%; width: 8%; height: 6%; background: #ffffff80; }`,
  // The id of the browser source that shows this page, so every window names itself.
  data: { source: '' },
  update: (el, data) => { el.title.textContent = data.source ? `${data.source} · alpha over video` : 'Browser alpha over video'; },
  // 82vw takes the marker from 5 % to 87 % of the window.
  actions: {
    right: [{ el: 'marker', x: [0, '82vw'], seconds: 2 }],
    left: [{ el: 'marker', x: ['82vw', 0], seconds: 2 }],
  },
  demo: [[0, 'right'], [6, 'left'], [12, 'repeat']],
  // Both halves of the cycle move alike, so many sources spread over one half.
  stagger: 6,
});
