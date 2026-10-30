// The graph stage: sigma draws the nodes (WebGL) at the Groups level; one 2D canvas below it draws tiles, groupBoxes and
// queue lines from cached Path2D batches; pooled HTML elements above it carry area frames, labels and node cards.
// Camera moves only re-stroke the cached batches and move the pooled elements; batches are rebuilt on level,
// selection, hover-free state or stats changes.
import Sigma from 'sigma';
import Graph from 'graphology';
import {CW, SOURCE_KINDS, selectionPath, isProblem, nodeInputs} from './model.mjs';
import {esc, fmtK, framesText, queueText, fillPct, fillClass, kindClass} from './format.mjs';

const LEVELS = [0, 12, 30, 90];  // px per row where Overview / Groups / Nodes / Details start
const JUMP_PY = [0, 18, 44, 120];       // px per row a level button or key jumps to
const MIN_RATIO = 0.003, MAX_RATIO = 1.5;
const MAX_CARDS = 350, MAX_GROUP_LABELS = 220;
const BAND = 0.2, KIND_FADE = 0.22, KIND_LINE_PX = 36;  // source kind band (share of the tile height), others' alpha while one kind is highlighted, tile size that shows the kind line

const rgba = (h, a) => {
  h = h.trim().replace('#', '');
  if (h.length === 3) h = [...h].map(c => c + c).join('');
  const n = parseInt(h.slice(0, 6), 16);
  return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`;
};
const ZONES = ['src', 'gpu', 'ctl', 'out'];
function readColors(el) {
  const s = getComputedStyle(el), v = k => s.getPropertyValue(k).trim();
  const C = {ink: v('--text'), muted: v('--muted'), paper: v('--bg'), card: v('--panel'), line: v('--edge'), teal: v('--pvw'),
    orange: v('--accent'), red: v('--pgm'), pkt: v('--wipe'), off: v('--c-off-s'), badF: v('--c-key-f'),
    zs: Object.fromEntries(ZONES.map(z => [z, v(`--c-${z}-s`)])), zf: Object.fromEntries(ZONES.map(z => [z, v(`--c-${z}-f`)])),
    src: Object.fromEntries(SOURCE_KINDS.map(([, k]) => [k, v(`--src-${k.toLowerCase()}`)]))};
  C.faded = rgba(v('--faint'), 0.14); C.tally = C.ink;
  return C;
}

export function createGraphStage(els, {onSelect, onLevel}) {
  const {stage, sigma: container, edges: ecv, zones: zonesEl, cards: cardsEl, tip} = els;
  // Colours come from the CSS tokens, read once the stage is in the document: a dock panel is mounted detached.
  const C = {};
  const syncColors = () => { if (!C.ink && stage.isConnected) Object.assign(C, readColors(stage)); };
  const st = {sel: null, selQ: null, path: null, level: -1, hoverNode: null, hover: null, kindHover: null, kindPin: null};
  let model = null, groupBoxes = new Map();
  let batches = {groupBoxes: [], edges: [], bands: [], tally: [], lines: []}, segs = [], groupBoxList = [], dirty = true;
  const G = new Graph({type: 'directed'});

  // ---------- sigma: nodes only ----------
  const renderer = new Sigma(G, container, {
    zIndex: true, allowInvalidContainer: true, minCameraRatio: MIN_RATIO, maxCameraRatio: MAX_RATIO, renderEdgeLabels: false, enableEdgeEvents: false,
    labelFont: 'Inter, ui-sans-serif, system-ui, sans-serif', labelSize: 11, labelWeight: '650', labelRenderedSizeThreshold: 6, labelDensity: 1,
    // node names sit on a dark plate, like the group labels, so lines under them never cut through the text
    defaultDrawNodeLabel: (ctx, d, set) => {
      if (!d.label) return;
      ctx.font = `${set.labelWeight} ${set.labelSize}px ${set.labelFont}`;
      const x = d.x + d.size + 4, w = ctx.measureText(d.label).width + 8, h = set.labelSize + 6;
      ctx.fillStyle = C.paper; ctx.beginPath(); ctx.roundRect(x - 4, d.y - h / 2, w, h, 4); ctx.fill();
      ctx.fillStyle = C.ink; ctx.fillText(d.label, x, d.y + set.labelSize / 3);
    },
    // sigma's default hover draws a white label box; the dark hover card shows the name, so only ring the node
    defaultDrawNodeHover: (ctx, d) => { ctx.beginPath(); ctx.arc(d.x, d.y, d.size + 3, 0, Math.PI * 2); ctx.strokeStyle = C.ink; ctx.lineWidth = 1.5; ctx.stroke(); },
    nodeReducer: (n, d) => {
      if (st.level !== 1) return {...d, hidden: true};  // Overview draws tiles, Nodes and Details draw cards
      syncColors();
      const m = model.meta.get(n), r = {...d, color: m.w ? C.zs[m.zone] : C.off, zIndex: 1};
      if (st.path && !st.path.nodes.has(n)) { r.color = C.faded; r.label = null; r.zIndex = 0; }
      if (n === st.sel?.id || n === st.hoverNode) { r.highlighted = true; r.size = d.size * 1.7; r.zIndex = 2; }
      return r;
    },
  });
  const cam = renderer.getCamera();
  const refresh = () => { dirty = true; renderer.refresh({skipIndexation: true}); };
  // Hover only changes the nodes' reducer output, never the cached batches.
  const redrawNodes = () => renderer.refresh({skipIndexation: true});

  // ---------- edge canvas: cached Path2D batches ----------
  const ectx = ecv.getContext('2d');
  const queueStyle = q => {
    if (q.st === 'idle') return null;  // idle queues (unsubscribed AUX taps) are listed on the Queues page
    let color = q.st === 'drop' ? C.red : q.st === 'full' ? C.orange : q.kind === 'pkt' ? C.pkt : C.teal;
    let w = q.st === 'drop' ? 2.2 : q.st === 'full' ? 1.8 : 1.2, z = isProblem(q.st) ? 3 : 1;
    if (st.path) { if (st.path.edges.has(q.n)) { w = Math.max(w, 2.2); z = 4; } else { color = C.faded; w = 0.8; z = 0; } }
    return {color, w, z};
  };
  const linkStyle = r => {
    if (!r.active) return null;
    let color = r.drop ? C.red : r.full ? C.orange : C.teal, z = 1;
    const w = st.level === 0 ? 3 + 1.4 * Math.log2(1 + r.active) : 1 + Math.log2(1 + r.active);
    if (st.path) { if (r.qs.some(q => st.path.edges.has(q.n))) z = 4; else { color = C.faded; z = 0; } }
    return {color, w, z};
  };
  const isInputTap = q => model.meta.get(q.a).d === 'inputs' && model.meta.get(q.b).d !== 'inputs';
  const tileRadius = (w, h) => st.level === 0 ? Math.min(w, h) * 0.14 : Math.min(h / 2, 1.2);

  function rebuild() {
    const by = new Map();
    segs = [];
    const add = (s, pts, ref) => {
      const key = `${s.z}|${s.color}|${s.w}`;
      let b = by.get(key);
      if (!b) by.set(key, b = {z: s.z, color: s.color, w: s.w, path: new Path2D()});
      b.path.moveTo(pts[0], pts[1]);
      for (let k = 2; k < pts.length; k += 2) b.path.lineTo(pts[k], pts[k + 1]);
      for (let k = 0; k + 3 < pts.length; k += 2) segs.push([pts[k], pts[k + 1], pts[k + 2], pts[k + 3], ref]);
    };
    // Overview: area links. Groups: queues inside the core and AUX, with input taps summed into trunks. Nodes and
    // Details: every queue.
    if (st.level === 0) for (const r of model.areaLinks) { const s = linkStyle(r); if (s) add(s, r.pts, r); }
    if (st.level === 1) for (const r of model.links) { if (r.from !== 'area:inputs') continue; const s = linkStyle(r); if (s) add(s, r.pts, r); }
    if (st.level >= 1) for (const q of model.edges) { if (st.level === 1 && isInputTap(q)) continue; const s = queueStyle(q); if (s) add(s, model.routes.get(q.n), q); }
    // Overview: one square tile per group, filled by health, in the mixer's bus-card look (dim fill, bright outline of the
    // same hue), numbered inputs with a band of their source kind's colour along the bottom. Deeper levels: a faint
    // groupBox under each group's nodes. Overview batches of input tiles are split by kind (k), so a highlighted kind
    // only changes their alpha.
    const lb = new Map(), eb = new Map(), bb = new Map(), tb = new Map();
    const batch = (map, k, color) => { const key = `${k}|${color}`; let b = map.get(key); if (!b) map.set(key, b = {k, color, path: new Path2D()}); return b.path; };
    groupBoxList = [];
    for (const g of model.groups.values()) {
      const l = st.level === 0 ? g.tile : groupBoxes.get(g.id);
      groupBoxList.push({g, ...l});
      const dim = st.path && !g.nodes.some(n => st.path.nodes.has(n));
      const bad = g.st === 'drop', idle = g.off || g.st === 'idle';
      const fill = dim ? C.faded : bad ? C.badF : idle ? C.card : C.zf[g.zone];
      const edge = st.level !== 0 || dim ? null : bad ? C.red : g.st === 'full' ? C.orange : idle ? C.line : C.zs[g.zone];
      const r = tileRadius(l.w, l.h), k = st.level === 0 && g.d === 'inputs' ? g.src || '' : null;
      batch(lb, k, fill).roundRect(l.x, l.y, l.w, l.h, r);
      if (edge) batch(eb, k, edge).roundRect(l.x, l.y, l.w, l.h, r);
      if (st.level === 0 && g.src && !dim) batch(bb, k, C.src[g.src]).roundRect(l.x, l.y, l.w, l.h * BAND, [r, r, 0, 0]);  // graph y points up: y is the bottom edge
    }
    // Overview: a dot on every tile that some compositor draws right now.
    if (st.level === 0) for (const gid of model.onAir.keys()) {
      const g = model.groups.get(gid), t = g?.tile;
      if (!t) continue;
      const r = t.w * 0.075, cx = t.x + t.w - r * 1.7, cy = t.y + t.h - r * 1.7, p = batch(tb, g.d === 'inputs' ? g.src || '' : null, C.tally);
      p.moveTo(cx + r, cy); p.arc(cx, cy, r, 0, Math.PI * 2);
    }
    batches = {groupBoxes: [...lb.values()], edges: [...eb.values()], bands: [...bb.values()], tally: [...tb.values()], lines: [...by.values()].sort((a, b) => a.z - b.z)};
    dirty = false;
  }

  let xf = {sx: 1, sy: 1, ox: 0, oy: 0};
  const strokePts = (pts, w) => {
    ectx.beginPath(); ectx.moveTo(pts[0], pts[1]);
    for (let i = 2; i < pts.length; i += 2) ectx.lineTo(pts[i], pts[i + 1]);
    ectx.strokeStyle = C.ink; ectx.lineWidth = w / xf.sx; ectx.stroke();
  };
  function drawCanvas() {
    if (dirty) rebuild();
    const dpr = devicePixelRatio || 1, W = ecv.clientWidth, H = ecv.clientHeight;
    if (ecv.width !== Math.round(W * dpr) || ecv.height !== Math.round(H * dpr)) { ecv.width = Math.round(W * dpr); ecv.height = Math.round(H * dpr); }
    const o = renderer.graphToViewport({x: 0, y: 0}), ux = renderer.graphToViewport({x: 1, y: 0}), uy = renderer.graphToViewport({x: 0, y: 1});
    xf = {sx: ux.x - o.x, sy: uy.y - o.y, ox: o.x, oy: o.y};
    ectx.setTransform(1, 0, 0, 1, 0, 0); ectx.clearRect(0, 0, ecv.width, ecv.height);
    ectx.setTransform(xf.sx * dpr, 0, 0, xf.sy * dpr, xf.ox * dpr, xf.oy * dpr);
    const k = 1 / xf.sx, kind = st.level === 0 && (st.kindHover || st.kindPin), base = st.level === 0 ? 1 : 0.6;
    const alpha = b => kind && b.k !== null && b.k !== kind ? KIND_FADE : 1;
    for (const b of batches.groupBoxes) { ectx.globalAlpha = base * alpha(b); ectx.fillStyle = b.color; ectx.fill(b.path); }
    for (const b of batches.bands) { ectx.globalAlpha = alpha(b); ectx.fillStyle = b.color; ectx.fill(b.path); }
    for (const b of batches.edges) { ectx.globalAlpha = alpha(b); ectx.strokeStyle = b.color; ectx.lineWidth = 1.5 * k; ectx.stroke(b.path); }
    for (const b of batches.tally) { ectx.globalAlpha = alpha(b); ectx.fillStyle = b.color; ectx.fill(b.path); }
    ectx.globalAlpha = 1;
    ectx.lineJoin = 'round'; ectx.lineCap = 'round';
    for (const b of batches.lines) { ectx.strokeStyle = b.color; ectx.lineWidth = b.w * k; ectx.stroke(b.path); }
    if (st.selQ && st.level > 0 && model.routes.has(st.selQ)) strokePts(model.routes.get(st.selQ), 3.2);
    if (st.hover?.pts) strokePts(st.hover.pts, 2.6);
    if (st.hover?.groupBox) {
      const l = st.hover.groupBox;
      ectx.beginPath(); ectx.roundRect(l.x, l.y, l.w, l.h, tileRadius(l.w, l.h)); ectx.strokeStyle = C.muted; ectx.lineWidth = 2 * k; ectx.stroke();
    }
  }

  // ---------- HTML layers: area frames, labels, node cards (pooled, transform only) ----------
  let zoneDivs = [];
  const gl = (x, y) => ({x: xf.ox + x * xf.sx, y: xf.oy + y * xf.sy});
  const labelPool = [], cardPool = [], tilePool = [], linkPool = [];
  const take = (pool, i, cls, init) => {
    if (!pool[i]) { const e = document.createElement('div'); e.className = cls; init?.(e); cardsEl.append(e); pool[i] = e; }
    return pool[i];
  };
  const initCard = e => e.addEventListener('click', ev => { ev.stopPropagation(); select({kind: 'node', id: e.dataset.id}); });
  const centerAt = (el, p) => { el.hidden = false; el.style.transform = `translate(calc(${p.x}px - 50%), calc(${p.y}px - 50%))`; };
  const kindHTML = k => `<span class="${kindClass(k)}">${esc(k)}</span>`;
  const groupHTML = g => esc(g.id) + (g.src ? ` · ${kindHTML(g.src)}` : '');
  const cardHTML = (m, detail) => {
    const ins = nodeInputs(model, m.n).filter(q => q.st !== 'idle'), q = ins[0], src = model.groups.get(m.g).src;
    const tag = src ? kindHTML(src) : esc(m.zone === 'out' ? 'output' : model.areaById.get(m.d).tag);
    const name = detail || !m.n.startsWith(m.g + '_') ? m.n : m.n.slice(m.g.length + 1);
    return `<span class="gv-tag">${tag} · ${esc(m.g)}</span><span class="gv-name">${esc(name)}</span>`
      + `<span class="gv-sub">${esc(m.t)}${q ? ` · ${q.pps.toFixed(0)} pps` : ''}${m.w ? '' : ' · OFF'}</span>`
      + (q ? `<span class="gv-sub">${framesText(q)}${ins.length > 1 ? ` · +${ins.length - 1} in` : ''}</span><span class="gv-meter ${fillClass(q)}"><i style="width:${fillPct(q)}%"></i></span>` : '');
  };

  function layoutHTML(py) {
    model.areas.forEach((z, i) => {
      const a = gl(z.rect.x0, z.rect.top), b = gl(z.rect.x1, z.rect.bot), s = zoneDivs[i].style;
      s.left = Math.min(a.x, b.x) + 'px'; s.top = Math.min(a.y, b.y) + 'px'; s.width = Math.abs(b.x - a.x) + 'px'; s.height = Math.abs(b.y - a.y) + 'px';
    });
    const {width: W, height: H} = renderer.getDimensions(), v = String(model.version);
    let nl = 0, nc = 0, nt = 0, nr = 0;
    if (st.level === 0) {
      // One label size per area so each grid stays uniform: the largest at which every name that fits at 8 px or more
      // fits its tile. A name too long even for 8 px (e.g. Wipe clip among the inputs) hides only its own label.
      const fitOf = g => Math.min(g.d === 'inputs' ? 10.5 : 12, (g.tile.w * Math.abs(xf.sx) - 4) / (g.name.length * 0.62));
      const fit = new Map();
      for (const g of model.groups.values()) { const f = fitOf(g); if (f >= 8) fit.set(g.d, Math.min(fit.get(g.d) ?? f, f)); }
      for (const g of model.groups.values()) {
        const t = g.tile, px = t.w * Math.abs(xf.sx), fs = fit.get(g.d);
        if (px < (g.d === 'inputs' ? 22 : 34) || !fs || fitOf(g) < fs) continue;
        // a numbered input's label sits above its kind band, with the kind under the number once the tile is large enough
        const p = gl(t.x + t.w / 2, t.y + t.h * (g.src ? (1 + BAND) / 2 : 0.5));
        if (p.x < -60 || p.x > W + 60 || p.y < -20 || p.y > H + 20) continue;
        const el = take(tilePool, nt++, 'gv-tlabel'), kindLine = !!g.src && px >= KIND_LINE_PX, key = kindLine ? v + 'k' : v;
        if (el.dataset.g !== g.id || el.dataset.v !== key) {
          el.dataset.g = g.id; el.dataset.v = key;
          if (kindLine) el.innerHTML = `${esc(g.name)}<small class="${kindClass(g.src)}">${esc(g.src)}</small>`; else el.textContent = g.name;
          el.className = 'gv-tlabel ' + (g.st === 'idle' ? 'idle' : g.st === 'drop' ? 'd' : 'z-' + g.zone);
          if (g.d === 'inputs') el.dataset.k = g.src || ''; else delete el.dataset.k;
        }
        const fsPx = fs.toFixed(1) + 'px';
        if (el.style.fontSize !== fsPx) el.style.fontSize = fsPx;
        centerAt(el, p);
      }
      for (const r of model.areaLinks) {
        if (!r.active) continue;
        const el = take(linkPool, nr++, 'gv-llabel');
        const txt = `${fmtK(r.pps)} frames/s<small>${r.active.toLocaleString()} ${r.active === 1 ? 'queue' : 'queues'}</small>`;
        if (el.dataset.t !== txt) { el.dataset.t = txt; el.innerHTML = txt; }
        centerAt(el, gl(r.mid[0], r.mid[1]));
      }
    }
    if (st.level >= 1 && py >= 11) for (const g of model.groups.values()) {
      if (g.d === 'inputs' && py < 22) continue;  // input labels need the gap between the two bands
      const p = gl(g.x0 * CW - 2.6, -(g.y0 + g.y1) / 2);
      if (p.y < -20 || p.y > H + 20 || p.x < -10 || p.x > W + 400) continue;
      const el = take(labelPool, nl++, 'gv-glabel');
      if (el.dataset.g !== g.id || el.dataset.v !== v) { el.dataset.g = g.id; el.dataset.v = v; el.innerHTML = groupHTML(g); el.className = 'gv-glabel' + (g.st === 'drop' ? ' d' : ''); }
      el.hidden = false; el.style.transform = `translate(calc(${p.x}px - 100%), ${p.y - 7}px)`;
      if (nl > MAX_GROUP_LABELS) break;
    }
    if (st.level >= 2) {
      const detail = st.level === 3, w = Math.min(CW * py * 0.8, detail ? 250 : 180), h = Math.min(py * 0.8, detail ? 92 : 34);
      for (const m of model.meta.values()) {
        const p = gl(m.x * CW, -m.y);
        if (p.x < -w || p.x > W + w || p.y < -h || p.y > H + h) continue;
        if (nc >= MAX_CARDS) break;
        const el = take(cardPool, nc++, 'gv-card', initCard);
        const dim = st.path && !st.path.nodes.has(m.n);
        const cls = `gv-card k-${m.zone}${detail ? '' : ' small'}${m.n === st.sel?.id ? ' sel' : ''}${dim ? ' faded' : ''}`;
        const key = `${m.n}|${detail}|${v}`;
        if (el.dataset.k !== key) { el.dataset.k = key; el.dataset.id = m.n; el.innerHTML = cardHTML(m, detail); }
        if (el.className !== cls) el.className = cls;
        el.hidden = false; el.style.width = w + 'px'; el.style.height = h + 'px'; el.style.transform = `translate(${p.x - w / 2}px, ${p.y - h / 2}px)`;
      }
    }
    for (const [pool, used] of [[labelPool, nl], [tilePool, nt], [linkPool, nr], [cardPool, nc]]) for (let i = used; i < pool.length; i++) if (!pool[i].hidden) pool[i].hidden = true;
  }

  // ---------- frame loop ----------
  const rowPx = () => Math.abs(renderer.graphToViewport({x: 0, y: -1}).y - renderer.graphToViewport({x: 0, y: 0}).y);
  renderer.on('beforeRender', syncColors);
  renderer.on('afterRender', () => {
    if (!model) return;
    const py = rowPx(), level = py >= LEVELS[3] ? 3 : py >= LEVELS[2] ? 2 : py >= LEVELS[1] ? 1 : 0;
    if (level !== st.level) { st.level = level; onLevel?.(level); refresh(); return; }
    drawCanvas(); layoutHTML(py);
  });

  // ---------- instant zoom: trackpad pinch, wheel and scroll handled here, applied without animation ----------
  const clampR = r => Math.min(MAX_RATIO, Math.max(MIN_RATIO, r));
  function zoomAt(vx, vy, factor) {
    const s = cam.getState(), r = clampR(s.ratio * factor), f = renderer.viewportToFramedGraph({x: vx, y: vy}), k = r / s.ratio;
    cam.setState({x: f.x + (s.x - f.x) * k, y: f.y + (s.y - f.y) * k, ratio: r});
  }
  function panBy(dx, dy) {
    const s = cam.getState(), a = renderer.viewportToFramedGraph({x: 0, y: 0}), b = renderer.viewportToFramedGraph({x: dx, y: dy});
    cam.setState({x: s.x + (b.x - a.x), y: s.y + (b.y - a.y)});
  }
  const overlay = '.gv-float,.gv-hud,.gv-zoomctl';
  const onWheel = ev => {
    if (ev.target.closest(overlay)) return;
    ev.preventDefault(); ev.stopPropagation();
    const r = stage.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
    const dy = ev.deltaMode === 1 ? ev.deltaY * 16 : ev.deltaY, dx = ev.deltaMode === 1 ? ev.deltaX * 16 : ev.deltaX;
    // pinch (ctrlKey) and mouse wheels zoom; two-finger trackpad scroll pans
    const mouseWheel = ev.deltaMode === 1 || (Math.abs(dy) >= 50 && dx === 0 && Number.isInteger(dy));
    if (ev.ctrlKey || mouseWheel) zoomAt(x, y, Math.exp(dy * (ev.ctrlKey ? 0.012 : 0.0025))); else panBy(dx, dy);
  };
  let gScale = 1;
  const onGestureStart = ev => { ev.preventDefault(); gScale = 1; };
  const onGestureChange = ev => { ev.preventDefault(); const r = stage.getBoundingClientRect(); zoomAt(ev.clientX - r.left, ev.clientY - r.top, gScale / ev.scale); gScale = ev.scale; };
  const ratioForPy = py => clampR(cam.getState().ratio * rowPx() / py);
  function jumpLevel(l) {
    if (l === 0) return cam.animatedReset({duration: 180});
    const {width: W, height: H} = renderer.getDimensions(), f = renderer.viewportToFramedGraph({x: W / 2, y: H / 2});
    cam.animate({x: f.x, y: f.y, ratio: ratioForPy(JUMP_PY[l])}, {duration: 180});
  }

  // ---------- hover and hit testing on the canvas layer ----------
  let mouse = {x: 0, y: 0}, hitQueued = false;
  const placeTip = () => { tip.style.left = Math.min(mouse.x + 14, stage.clientWidth - 290) + 'px'; tip.style.top = (mouse.y + 14) + 'px'; };
  const showTip = html => { tip.innerHTML = html; tip.hidden = false; placeTip(); };
  const hideTip = () => { tip.hidden = true; };
  const placeName = x => model.areaById.get(x)?.title || (x === 'area:inputs' ? model.areaById.get('inputs').title : x);
  function hitTest(vx, vy) {
    const gx = (vx - xf.ox) / xf.sx, gy = (vy - xf.oy) / xf.sy, tol = 5 / Math.abs(xf.sx);
    let best = null, bd = tol;
    for (const [x1, y1, x2, y2, ref] of segs) {
      const d = x1 === x2
        ? (gy < Math.min(y1, y2) - tol || gy > Math.max(y1, y2) + tol ? Infinity : Math.abs(gx - x1))
        : (gx < Math.min(x1, x2) - tol || gx > Math.max(x1, x2) + tol ? Infinity : Math.abs(gy - y1));
      if (d < bd) { bd = d; best = ref; }
    }
    if (best) return best.key ? {link: best, pts: best.pts} : {q: best, pts: model.routes.get(best.n)};
    if (st.level <= 1) for (const l of groupBoxList) if (gx >= l.x && gx <= l.x + l.w && gy >= l.y && gy <= l.y + l.h) return {groupBox: l};
    return null;
  }
  const hoverId = h => h?.link || h?.q || h?.groupBox?.g;
  function hoverAt(target) {
    if (!model) return;
    if (target.closest(`.gv-card,.gv-chip,${overlay}`)) { if (st.hover) { st.hover = null; hideTip(); renderer.scheduleRender(); } return; }
    const h = st.hoverNode ? null : hitTest(mouse.x, mouse.y);
    if (hoverId(h) === hoverId(st.hover)) return;
    st.hover = h; container.style.cursor = h ? 'pointer' : '';
    if (h?.link) {
      const r = h.link;
      showTip(`<b>${esc(placeName(r.from))} → ${esc(placeName(r.to))}</b><code>${r.n} queues · ${r.active} flowing · ${r.pps.toFixed(0)} pps${r.drop ? ` · ${r.drop} dropped` : ''}</code><code>click: open</code>`);
    } else if (h?.q) showTip(`<b>${esc(h.q.a)} → ${esc(h.q.b)}</b><code>${esc(h.q.n)}</code><code>${esc(queueText(h.q))}</code><code>click: follow queue</code>`);
    else if (h?.groupBox) { const g = h.groupBox.g; showTip(`<b>${groupHTML(g)}</b><code>${g.n} nodes · ${g.st}${g.off ? ' · has OFF nodes' : ''}</code><code>click: open group</code>`); }
    else hideTip();
    renderer.scheduleRender();
  }
  const onMouseMove = ev => {
    const r = stage.getBoundingClientRect();
    mouse = {x: ev.clientX - r.left, y: ev.clientY - r.top};
    if (!tip.hidden) placeTip();
    if (hitQueued || ev.buttons) return;
    hitQueued = true;
    const target = ev.target;
    requestAnimationFrame(() => { hitQueued = false; hoverAt(target); });
  };
  const onMouseLeave = () => { st.hover = null; hideTip(); renderer.scheduleRender(); };
  renderer.on('enterNode', ({node: n}) => {
    st.hoverNode = n; st.hover = null;
    const m = model.meta.get(n);
    showTip(`<b>${esc(n)}</b><code>${esc(m.t)} · ${esc(m.g)} · ${(model.inQ.get(n) || []).length} in · ${(model.outQ.get(n) || []).length} out</code>`);
    container.style.cursor = 'pointer'; redrawNodes();
  });
  renderer.on('leaveNode', () => { st.hoverNode = null; hideTip(); container.style.cursor = ''; redrawNodes(); });
  renderer.on('clickNode', ({node: n}) => select({kind: 'node', id: n}));
  renderer.on('clickStage', () => {
    const h = st.hover;
    if (!h) return select(null);
    if (h.q) { const id = st.sel?.id === h.q.b ? h.q.a : h.q.b; select({kind: 'node', id, queue: h.q.n}, true); }
    else if (h.link) { if (model.groups.has(h.link.to)) select({kind: 'group', id: h.link.to}, true); else flyToArea(h.link.to); }
    else if (h.groupBox) select({kind: 'group', id: h.groupBox.g.id}, true);
  });

  // ---------- selection and camera flights ----------
  function flyToArea(d) {
    const r = model.areaById.get(d).rect, a = gl(r.x0, r.top), b = gl(r.x1, r.bot), {width: W, height: H} = renderer.getDimensions();
    const f = renderer.viewportToFramedGraph({x: (a.x + b.x) / 2, y: (a.y + b.y) / 2}), k = Math.max(Math.abs(b.x - a.x) / W, Math.abs(b.y - a.y) / H) * 1.15;
    cam.animate({x: f.x, y: f.y, ratio: clampR(cam.getState().ratio * k)}, {duration: 260});
  }
  function centerOf(names) {
    let x = 0, y = 0;
    for (const n of names) { const d = renderer.getNodeDisplayData(n); x += d.x; y += d.y; }
    return {x: x / names.length, y: y / names.length};
  }
  const valid = s => s && (s.kind === 'node' ? model.meta.has(s.id) : model.groups.has(s.id)) ? s : null;
  // A stage without a size (its page is hidden) flies once it is shown again.
  let pendingFly = null;
  function flyTo(s) {
    if (!stage.clientWidth || !stage.clientHeight) { pendingFly = s; return; }
    pendingFly = null;
    const c = centerOf(s.kind === 'node' ? [s.id] : model.groups.get(s.id).nodes);
    cam.animate({x: c.x, y: c.y, ratio: Math.min(cam.getState().ratio, ratioForPy(s.kind === 'node' ? 120 : 46))}, {duration: 260});
  }
  function apply(s, fly) {
    s = valid(s);
    st.selQ = s?.kind === 'node' ? s.queue || null : null;
    st.sel = s; st.path = selectionPath(model, s);
    st.hover = null; hideTip();
    pendingFly = null;
    if (s && fly) flyTo(s);
    refresh();
  }
  // From a click on the stage: applied here, then reported.
  function select(s, fly) { apply(s, fly); onSelect?.(st.sel); }

  // ---------- source kind highlight: a hovered or pinned header chip dims the other input tiles ----------
  let chips = [];
  function setKind(hover, pin, force) {
    if (!force && hover === st.kindHover && pin === st.kindPin) return;
    st.kindHover = hover; st.kindPin = pin;
    const kind = hover || pin;
    if (kind) stage.dataset.hl = kind; else delete stage.dataset.hl;  // dims the other inputs' tile labels (graph.css)
    for (const c of chips) { c.setAttribute('aria-pressed', String(c.dataset.kind === pin)); c.classList.toggle('on', c.dataset.kind === kind); }
    renderer.scheduleRender();
  }

  // ---------- model ----------
  function setModel(m) {
    model = m;
    G.clear();
    for (const n of m.meta.values()) G.addNode(n.n, {x: n.x * CW, y: -n.y, size: 2.6, label: n.n});
    groupBoxes = new Map([...m.groups.values()].map(g => [g.id, {x: g.x0 * CW - 2, y: -g.y1 - 0.42, w: (g.x1 - g.x0) * CW + 4, h: g.y1 - g.y0 + 0.84}]));
    for (const d of zoneDivs) d.remove();
    zoneDivs = m.areas.map(z => {
      const d = document.createElement('div');
      d.className = 'gv-zone';
      d.innerHTML = `<b>${esc(z.title.toUpperCase())}<span>${esc(z.count)}</span>${(z.kinds || []).map(([k, c]) => `<button class="gv-chip" data-kind="${esc(k)}"`
        + ` data-tip="click: pin · Esc: clear"><i class="${kindClass(k)}"></i>${c} ${esc(k)}</button>`).join('')}</b>`;
      zonesEl.append(d);
      return d;
    });
    chips = zoneDivs.flatMap(d => [...d.querySelectorAll('.gv-chip')]);
    for (const c of chips) {
      c.addEventListener('pointerenter', () => setKind(c.dataset.kind, st.kindPin));
      c.addEventListener('pointerleave', () => setKind(null, st.kindPin));
      c.addEventListener('click', () => setKind(st.kindHover, st.kindPin === c.dataset.kind ? null : c.dataset.kind));
    }
    setKind(null, chips.some(c => c.dataset.kind === st.kindPin) ? st.kindPin : null, true);
    st.hover = null; st.hoverNode = null; hideTip();
    apply(st.sel, false);
    renderer.refresh();
  }

  const onResize = () => requestAnimationFrame(() => {
    renderer.resize(); renderer.refresh();
    if (pendingFly && valid(pendingFly)) flyTo(pendingFly);
  });
  const resizeObserver = new ResizeObserver(onResize);
  resizeObserver.observe(stage);
  stage.addEventListener('wheel', onWheel, {capture: true, passive: false});
  stage.addEventListener('gesturestart', onGestureStart);
  stage.addEventListener('gesturechange', onGestureChange);
  stage.addEventListener('mousemove', onMouseMove);
  stage.addEventListener('mouseleave', onMouseLeave);

  return {
    setModel,
    // A new queues.json sample on the current model: colours, numbers and the flowing part of the path change.
    statsChanged() { if (!model) return; st.path = selectionPath(model, st.sel); refresh(); },
    // From outside (inspector, tables): applied and flown to unless the stage already shows that node or group; a
    // followed queue only adds its highlight.
    setSelection(s, fly = true) {
      s = valid(s);
      if (s && s.kind === st.sel?.kind && s.id === st.sel?.id) {
        if (s.queue && s.queue !== st.selQ) { st.selQ = s.queue; renderer.scheduleRender(); }  // stroked every frame
      } else if (s || st.sel) apply(s, fly);
    },
    select,
    // Esc: a pinned source kind highlight is cleared first; false when there was none.
    clearKindPin() { if (!st.kindPin) return false; setKind(st.kindHover, null); return true; },
    jumpLevel,
    zoomIn: () => cam.animatedZoom({duration: 140, factor: 2}),
    zoomOut: () => cam.animatedUnzoom({duration: 140, factor: 2}),
    fit: () => cam.animatedReset({duration: 180}),
    resize: onResize,
    destroy() {
      resizeObserver.disconnect();
      stage.removeEventListener('wheel', onWheel, {capture: true});
      stage.removeEventListener('gesturestart', onGestureStart);
      stage.removeEventListener('gesturechange', onGestureChange);
      stage.removeEventListener('mousemove', onMouseMove);
      stage.removeEventListener('mouseleave', onMouseLeave);
      renderer.kill();
      for (const pool of [labelPool, cardPool, tilePool, linkPool, zoneDivs]) for (const e of pool) e.remove();
    },
  };
}
