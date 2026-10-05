// Crawl strip, the second lower-third layer. The window is the strip itself.
const ITEMS = ['Downstream keyer on air: clean feed stays unkeyed', 'Four browser keys, one GPU pass',
  'Keys cut on and off at the next program frame', 'Scenes and transitions run underneath'];
const graphic = Motion.graphic({
  html: '<div id="tag">LIVE</div><div id="band"><div id="crawl"><span></span><span></span></div></div>',
  css: `
    :host { display: flex; color: white; font-family: system-ui, sans-serif; }
    #tag { flex: none; padding: 0 2vh; background: #ff5a1f; font: 700 52vh/100vh system-ui, sans-serif; }
    #band { flex: 1; overflow: hidden; background: #101c2bdd; }
    #crawl { display: flex; white-space: nowrap; font-size: 50vh; line-height: 100vh; }
    #crawl span { flex: none; }`,
  data: { text: ITEMS.map((item) => `${item}   •   `).join('') },
  update: (el, data) => { for (const copy of el.crawl.children) copy.textContent = data.text; },
  loop: [{ el: 'crawl', crawl: { pxPerSecond: 110 } }],
  demo: [[0, 'play']],
});
