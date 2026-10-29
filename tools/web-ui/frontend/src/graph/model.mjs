// Overview view model of an avplumber graph: topology, layout, routing and live queue state, without any DOM.
//
// Layout (computed once per nodes.json): every node gets a column (its longest path from a source) and a row. Groups
// become group bands; groups are gathered into areas:
// - "inputs": a numbered group family (FAMILY_MIN or more members such as input_0 … input_191) whose members take no
//   queue from another group, plus every other such source group. Laid out as one row per group in bands side by side.
// - one area per other numbered family, keyed by the family's first name token (aux_aux0 … → "aux"); unnumbered
//   groups with that token join it (aux_mv).
// - "core": all remaining groups, stacked right of the inputs.
// Queue stats (every queues.json poll) only change colours and numbers, never positions.

export const CW = 4;  // graph units per column; one row is one unit
const FAMILY_MIN = 8;
const ROW_GAP = 0.6, AREA_GAP = 9, CORE_TOP_MAX = 18;

// Presentation hints for the mixer's known groups (pyplumber/mixer), so its graph reads like the operator page. Other
// graphs fall back to generic names: a numbered family member shows its number (prefixed by its area tag outside
// the inputs), any other group its id without the area's name token.
const GROUP_HINTS = {
  mixer_a: {name: 'Slot A', order: 1}, mixer_b: {name: 'Slot B', order: 2}, mixer: {name: 'Program', order: 3},
  mixer_wipe: {name: 'Wipe', order: 4}, output: {name: 'Outputs', order: 5}, mixer_wipe_load: {name: 'Wipe clip', order: 99},
  aux_mv: {name: 'MV'}, aux_mv2: {name: 'MV 2'},
};
const MIXER_CORE = {group: 'mixer', title: 'Program core', tag: 'mixer'};  // used when the graph has a "mixer" group
const AREA_TITLES = {inputs: 'Inputs', core: 'Core', aux: 'AUX + multiview'};
// Source kind (g.src) of a numbered input group by its nodes' types, the first match winning; other groups have none.
export const SOURCE_KINDS = [['dec_video', 'NVDEC'], ['ipc_dmabuf_source', 'browser'], ['raw_video_file_cuda', 'raw']];

const NB_FANOUT = 'one_to_many_nb';  // drops by design when a consumer still holds a newer frame
const NEAR_FULL = 0.9;               // average fill of capacity
const TILE_DROP_SHARE = 0.01;        // a group tile turns red only when a queue lost more than this share of its frames
const DROP_WARN_SHARE = 0.005;       // the instance's drop tile turns orange from this share, red above TILE_DROP_SHARE

const refs = v => typeof v === 'string' ? [v] : Array.isArray(v) ? v.filter(x => typeof x === 'string') : [];
const groupOf = n => String(n.params?.group || 'ungrouped');
const familyKey = g => g.replace(/\d+$/, '#');
const familyIndex = g => { const m = /\d+$/.exec(g); return m ? +m[0] : -1; };
const tokenOf = g => g.split('_')[0];
const cmp = (a, b) => a < b ? -1 : a > b ? 1 : 0;
const push = (map, key, value) => { let list = map.get(key); if (!list) map.set(key, list = []); list.push(value); };

// Producer and consumer of every queue named in the node params.
function wiring(nodes) {
  const prod = new Map(), cons = new Map();
  for (const n of nodes) {
    const p = n.params || {}, dst = new Set(refs(p.dst));
    for (const q of refs(p.src)) cons.set(q, n.name);
    if (p.routing && typeof p.routing === 'object') for (const v of Object.values(p.routing)) for (const q of refs(v)) dst.add(q);
    for (const q of dst) prod.set(q, n.name);
  }
  return {prod, cons};
}

// Column of every node: its longest path from a source; sources sit just before their first consumer and unconnected
// nodes just before the rest of their group.
// A cycle is cut at its back edge, found by a depth-first walk from the sources.
function layers(names, inn, out, groupNodes, group) {
  const back = new Map(), state = new Map();
  const walk = n => {
    state.set(n, 1);
    for (const c of out.get(n) || []) {
      if (state.get(c) === 1) { if (!back.has(n)) back.set(n, new Set()); back.get(n).add(c); }
      else if (!state.has(c)) walk(c);
    }
    state.set(n, 2);
  };
  for (const n of names) if (!inn.get(n)?.length) walk(n);
  for (const n of names) if (!state.has(n)) walk(n);
  const memo = new Map();
  const L = n => {
    if (!memo.has(n)) memo.set(n, Math.max(0, ...(inn.get(n) || []).filter(p => !back.get(p)?.has(n)).map(p => L(p) + 1)));
    return memo.get(n);
  };
  for (const n of names) L(n);
  for (const n of names) if (!inn.get(n)?.length && out.get(n)?.length) memo.set(n, Math.max(0, Math.min(...out.get(n).map(c => memo.get(c))) - 1));
  for (const n of names) if (!inn.get(n)?.length && !out.get(n)?.length) {
    const rest = groupNodes.get(group.get(n)).filter(m => m !== n).map(m => memo.get(m));
    if (rest.length) memo.set(n, Math.max(0, Math.min(...rest) - 1));
  }
  return memo;
}

// Every target owns a vertical lane in the gap before its column, so a fan-in reads as one trunk.
function laneMap(items) {
  const byCol = new Map(), lanes = new Map();
  for (const it of items) push(byCol, it.col, it);
  for (const list of byCol.values()) {
    list.sort((a, b) => a.row - b.row);
    list.forEach((it, k) => lanes.set(it.id, it.col * CW - CW * (0.2 + 0.55 * (k + 0.5) / list.length)));
  }
  return lanes;
}
function route(sx, sy, tx, ty, lx, sgap) {
  if (sy === ty && tx > sx && tx - sx <= CW) return [sx, sy, tx, ty];
  if (lx > sx) return [sx, sy, lx, sy, lx, ty, tx, ty];
  const o = sx + CW * 0.3, by = ty - 0.45 * sgap;
  return [sx, sy, o, sy, o, by, lx, by, lx, ty, tx, ty];
}
const link = (key, from, to) => ({key, from, to, qs: [], n: 0, active: 0, pps: 0, drop: 0, full: 0});  // qs: queue objects

export function buildGraphModel({nodes = [], queues = []} = {}, previous = null) {
  const {prod, cons} = wiring(nodes);
  const meta = new Map(), groupNodes = new Map(), group = new Map();
  for (const n of nodes) {
    const g = groupOf(n), p = {...n.params};
    delete p.name; delete p.type;
    meta.set(n.name, {n: n.name, t: n.type, g, w: n.working !== false, p, x: 0, y: 0});
    group.set(n.name, g); push(groupNodes, g, n.name);
  }

  // Queues: every name in the params plus every name in the stats sample; edges are those with both ends.
  const Q = new Map(), inQ = new Map(), outQ = new Map(), inn = new Map(), out = new Map();
  const addQueue = name => {
    if (Q.has(name)) return Q.get(name);
    const a = meta.has(prod.get(name)) ? prod.get(name) : null, b = meta.has(cons.get(name)) ? cons.get(name) : null;
    const q = {n: name, a, b, kind: 'frame', nb: !!a && meta.get(a).t === NB_FANOUT, c: 0, o: 0, avg: 0, mx: 0, pps: 0,
      dr: 0, enq: 0, deq: 0, drD: null, enqD: null, known: false, sampled: -1, st: 'idle'};
    Q.set(name, q);
    if (a && b) { push(outQ, a, name); push(inQ, b, name); push(out, a, b); push(inn, b, a); }
    return q;
  };
  for (const name of new Set([...prod.keys(), ...cons.keys()])) addQueue(name);
  const edges = [...Q.values()].filter(q => q.a && q.b);

  const names = [...meta.keys()];
  const memo = layers(names, inn, out, groupNodes, group);

  // Areas.
  const crossIn = new Map(), crossOut = new Map();
  for (const q of edges) { const ga = group.get(q.a), gb = group.get(q.b); if (ga !== gb) { push(crossIn, gb, ga); push(crossOut, ga, gb); } }
  const isSource = g => !crossIn.has(g);
  const families = new Map();
  for (const g of groupNodes.keys()) if (familyIndex(g) >= 0) push(families, familyKey(g), g);
  const bigTokens = new Map();  // token -> member groups of its big families
  for (const members of families.values()) if (members.length >= FAMILY_MIN) for (const g of members) push(bigTokens, tokenOf(g), g);
  let sourceToken = null, sourceScore = 0;
  for (const [tok, members] of bigTokens) {
    const s = members.filter(isSource).length;
    if (s * 2 > members.length && s > sourceScore) { sourceToken = tok; sourceScore = s; }
  }
  const areaOf = g => {
    const tok = tokenOf(g);
    if (sourceToken && tok === sourceToken) return 'inputs';
    if (bigTokens.has(tok)) return tok;
    if (sourceToken && isSource(g) && crossOut.has(g)) return 'inputs';
    return 'core';
  };
  const byArea = new Map();
  const order = g => GROUP_HINTS[g]?.order ?? 9;
  const sorted = [...groupNodes.keys()].sort((a, b) => order(a) - order(b) || cmp(familyKey(a), familyKey(b)) || familyIndex(a) - familyIndex(b) || cmp(a, b));
  for (const g of sorted) push(byArea, areaOf(g), g);
  const dOrder = ['inputs', 'core', ...[...byArea.keys()].filter(d => d !== 'inputs' && d !== 'core').sort()];
  const areaIds = dOrder.filter(d => byArea.has(d));

  // Columns: the input groups in bands side by side, everything else past them, leaving a lane gap for the taps.
  const inputs = byArea.get('inputs') || [];
  const bands = inputs.length >= 16 ? 2 : 1, perBand = Math.ceil(inputs.length / bands);
  const bandOf = new Map(inputs.map((g, i) => [g, Math.floor(i / perBand)]));
  const inW = inputs.length ? Math.max(...inputs.flatMap(g => groupNodes.get(g).map(n => memo.get(n)))) + 2 : 0;
  for (const n of names) { const g = group.get(n); memo.set(n, memo.get(n) + (bandOf.has(g) ? bandOf.get(g) : bands) * inW); }

  // Rows: one band of rows per group, sub-rows where a group has several nodes in one column.
  const groups = new Map();
  const place = (d, g, y) => {
    const cols = new Map();
    for (const n of [...groupNodes.get(g)].sort(cmp)) push(cols, memo.get(n), n);
    const h = Math.max(...[...cols.values()].map(v => v.length));
    for (const [c, ns] of cols) ns.forEach((n, k) => { const m = meta.get(n); m.x = c; m.y = y + k; });
    const xs = groupNodes.get(g).map(n => memo.get(n));
    groups.set(g, {id: g, d, x0: Math.min(...xs), x1: Math.max(...xs), y0: y, y1: y + h - 1, n: xs.length, nodes: groupNodes.get(g),
      st: 'ok', off: groupNodes.get(g).some(n => !meta.get(n).w), inQueues: groupNodes.get(g).flatMap(n => (inQ.get(n) || []).map(name => Q.get(name)))});
    return h;
  };
  const cursor = new Array(bands).fill(0);
  inputs.forEach(g => { const b = bandOf.get(g); cursor[b] += place('inputs', g, cursor[b]); });
  let y = inputs.length ? Math.min(CORE_TOP_MAX, Math.floor(Math.max(...cursor) / 5)) : 0;
  areaIds.forEach((d, k) => {
    if (d === 'inputs') return;
    if (k > (inputs.length ? 1 : 0)) y += AREA_GAP;
    for (const g of byArea.get(d)) y += place(d, g, y) + ROW_GAP;
  });

  // Areas: frame, title, tag and colour zone.
  const mixerCore = groups.has(MIXER_CORE.group);
  const areas = areaIds.map(d => {
    const gs = byArea.get(d).map(g => groups.get(g));
    const z = {d, groups: gs, x0: Math.min(...gs.map(g => g.x0)), x1: Math.max(...gs.map(g => g.x1)), y0: Math.min(...gs.map(g => g.y0)), y1: Math.max(...gs.map(g => g.y1))};
    z.rect = {x0: z.x0 * CW - 3, x1: z.x1 * CW + 3, top: -z.y0 + 1.2, bot: -z.y1 - 1.2};
    if (d === 'inputs') {
      const members = gs.filter(g => tokenOf(g.id) === sourceToken && familyIndex(g.id) >= 0);
      for (const g of members) g.src = SOURCE_KINDS.find(([t]) => g.nodes.some(n => meta.get(n).t === t))?.[1];
      const kinds = SOURCE_KINDS.map(([, k]) => [k, members.filter(g => g.src === k).length]).filter(([, c]) => c);
      Object.assign(z, {title: AREA_TITLES.inputs, tag: 'input', zone: 'src', count: `${members.length} sources`, kinds});
    } else if (d === 'core') Object.assign(z, {title: mixerCore ? MIXER_CORE.title : AREA_TITLES.core, tag: mixerCore ? MIXER_CORE.tag : 'core', zone: 'gpu'});
    else Object.assign(z, {title: AREA_TITLES[d] || d.toUpperCase(), tag: d, zone: 'ctl'});
    z.count ??= `${gs.length} groups`;
    return z;
  });
  const areaById = new Map(areas.map(z => [z.d, z]));
  for (const g of groups.values()) {
    const z = areaById.get(g.d), idx = familyIndex(g.id), member = idx >= 0 && (families.get(familyKey(g.id))?.length || 0) >= FAMILY_MIN;
    const stripped = g.id.startsWith(z.tag + '_') ? g.id.slice(z.tag.length + 1) : g.id;
    g.name = GROUP_HINTS[g.id]?.name || (member ? (g.d === 'inputs' ? String(idx) : `${z.tag.toUpperCase()} ${idx}`) : stripped);
    g.member = member;
    g.zone = g.d === 'core' && g.nodes.some(n => meta.get(n).t === 'output') ? 'out' : z.zone;
    for (const n of g.nodes) { const m = meta.get(n); m.d = g.d; m.zone = g.zone; }
  }

  // Overview tiles: each area is a grid of square tiles, one per group.
  for (const z of areas) {
    const r = z.rect, W = r.x1 - r.x0, H = r.top - r.bot, n = z.groups.length;
    const cols = Math.min(n, Math.ceil(Math.sqrt(n * W / H))), rows = Math.ceil(n / cols), size = Math.min(W / cols, H / rows), pad = size * 0.09;
    const ox = r.x0 + (W - cols * size) / 2, oy = r.top - (H - rows * size) / 2;
    z.groups.forEach((g, i) => { g.tile = {x: ox + (i % cols) * size + pad, y: oy - (Math.floor(i / cols) + 1) * size + pad, w: size - 2 * pad, h: size - 2 * pad}; });
  }

  // Routes of every queue, computed once.
  const nodeLanes = laneMap(names.filter(n => inQ.has(n)).map(n => ({id: n, col: meta.get(n).x, row: meta.get(n).y})));
  const routes = new Map();
  for (const q of edges) { const a = meta.get(q.a), b = meta.get(q.b); routes.set(q.n, route(a.x * CW, -a.y, b.x * CW, -b.y, nodeLanes.get(q.b), 1)); }

  // Groups: queues between groups summed into links; all input groups feed each consumer as one trunk.
  const links = new Map();
  for (const q of edges) {
    const ga = groups.get(group.get(q.a)), gb = groups.get(group.get(q.b));
    if (ga === gb) continue;
    const from = ga.d === 'inputs' && gb.d !== 'inputs' ? 'area:inputs' : ga.id, key = `${from}>${gb.id}`;
    if (!links.has(key)) links.set(key, link(key, from, gb.id));
    links.get(key).qs.push(q);
  }
  const gMid = g => -(g.y0 + g.y1) / 2;
  const groupLanes = laneMap([...new Set([...links.values()].map(r => r.to))].map(id => ({id, col: groups.get(id).x0, row: groups.get(id).y0})));
  const inX1 = areaById.get('inputs')?.x1;
  for (const r of links.values()) {
    const b = groups.get(r.to), ty = gMid(b), tx = b.x0 * CW - 1.6;
    if (r.from === 'area:inputs') r.pts = [inX1 * CW + 2, ty, tx, ty];
    else { const a = groups.get(r.from); r.pts = route(a.x1 * CW + 1.6, gMid(a), tx, ty, groupLanes.get(r.to), 1.6); }
  }

  // Overview: areas joined by a few summed links.
  const areaLinks = new Map();
  for (const q of edges) {
    const da = meta.get(q.a).d, db = meta.get(q.b).d;
    if (da === db) continue;
    const key = `${da}>${db}`;
    if (!areaLinks.has(key)) areaLinks.set(key, link(key, da, db));
    areaLinks.get(key).qs.push(q);
  }
  for (const r of areaLinks.values()) {
    const a = areaById.get(r.from).rect, b = areaById.get(r.to).rect;
    if (a.x1 <= b.x0) { const yy = (Math.max(a.bot, b.bot) + Math.min(a.top, b.top)) / 2; r.pts = [a.x1, yy, b.x0, yy]; }
    else { const x = (Math.max(a.x0, b.x0) + Math.min(a.x1, b.x1)) / 2, down = a.bot > b.top; r.pts = [x, down ? a.bot : a.top, x, down ? b.top : b.bot]; }
    r.mid = [(r.pts[0] + r.pts[2]) / 2, (r.pts[1] + r.pts[3]) / 2];
  }

  // Drawn now: which groups each compositor draws, from its active_inputs bit string over its src queues.
  const onAir = new Map(), drawing = new Map();
  for (const n of nodes) {
    const bits = n.params?.active_inputs, src = refs(n.params?.src);
    if (n.type !== 'mixer_compositor' || typeof bits !== 'string') continue;
    const drawn = new Set();
    for (let i = 0; i < bits.length && i < src.length; i++) if (bits[i] === '1' && meta.has(prod.get(src[i]))) drawn.add(group.get(prod.get(src[i])));
    drawing.set(n.name, [...drawn]);
    for (const g of drawn) push(onAir, g, n.name);
  }

  const model = {meta, groups, areas, areaById, queues: Q, edges, inQ, outQ, routes, links: [...links.values()],
    areaLinks: [...areaLinks.values()], onAir, drawing, nodeCount: meta.size, version: 0, sample: null, addQueue};
  // Carry the previous graph's counters over, so the first sample after a nodes.json refresh still has drop deltas.
  if (previous) for (const q of Q.values()) {
    const p = previous.queues.get(q.n);
    if (p?.known) Object.assign(q, {known: true, dr: p.dr, enq: p.enq});
  }
  updateQueueStats(model, queues);
  return model;
}

const num = v => typeof v === 'number' && Number.isFinite(v) ? v : 0;
const delta = (now, before) => now >= before ? now - before : null;  // null after a restart reset the counters

// One queues.json sample: copies the numbers, computes drops since the previous sample and classifies every queue.
// Returns false when this exact sample was already applied (several views share one model).
export function updateQueueStats(model, queues = []) {
  if (!Array.isArray(queues) || queues === model.sample) return false;
  model.sample = queues;
  const v = model.version + 1;
  for (const s of queues) {
    if (!s || typeof s.name !== 'string') continue;
    const q = model.queues.get(s.name) || model.addQueue(s.name), fq = s.frames_in_queue || {};
    const dr = num(s.dropped_total), enq = num(s.enqueued_total);
    q.drD = q.known ? delta(dr, q.dr) : null; q.enqD = q.known ? delta(enq, q.enq) : null;
    q.known = true; q.sampled = v; q.kind = s.type === 'av::Packet' ? 'pkt' : 'frame';
    q.c = num(s.capacity); q.o = num(s.occupied); q.avg = Math.round(num(fq.avg ?? s.occupied) * 100) / 100; q.mx = num(fq.max ?? s.occupied);
    q.pps = num(s.pps); q.dr = dr; q.enq = enq; q.deq = num(s.dequeued_total);
  }
  for (const q of model.queues.values()) {
    if (q.sampled !== v) { q.known = false; q.pps = 0; q.drD = null; q.enqD = null; }
    q.st = queueState(q);
  }
  for (const g of model.groups.values()) g.st = g.inQueues.length ? groupState(g.inQueues) : 'ok';
  for (const r of [...model.links, ...model.areaLinks]) {
    r.n = r.qs.length; r.active = 0; r.pps = 0; r.drop = 0; r.full = 0;
    for (const q of r.qs) {
      if (q.st !== 'idle') { r.active++; r.pps += q.pps; }
      if (q.st === 'drop') r.drop += q.drD;
      if (q.st === 'full') r.full++;
    }
  }
  model.version = v;
  return true;
}

// Worst state of the queues entering a group; a drop of at most TILE_DROP_SHARE of the frames stays visible on the
// queue lines only. The engine counts a dropped frame in dropped_total and never in enqueued_total.
function groupState(qs) {
  let drop = false, full = false, idle = true;
  for (const q of qs) {
    const st = q.st === 'drop' && q.drD / Math.max(1, (q.enqD || 0) + q.drD) <= TILE_DROP_SHARE ? 'ok' : q.st;
    if (st === 'drop') drop = true; else if (st === 'full') full = true;
    if (st !== 'idle') idle = false;
  }
  return drop ? 'drop' : full ? 'full' : idle ? 'idle' : 'ok';
}

// drop: lost frames since the previous sample (a non-blocking fan-out skipping a frame is by design, not a fault);
// idle: nothing flows (e.g. an unsubscribed AUX tap); full: average fill at 90% of capacity or more.
function queueState(q) {
  if (q.drD > 0 && !q.nb) return 'drop';
  if (!(q.pps >= 0.005)) return 'idle';
  return q.c > 0 && q.avg >= q.c * NEAR_FULL ? 'full' : 'ok';
}
export const isProblem = st => st === 'drop' || st === 'full';

// Everything upstream and downstream of a node through flowing queues.
export function nodePath(model, start) {
  const nodes = new Set([start]), edges = new Set();
  for (const down of [true, false]) {
    const stack = [start];
    while (stack.length) {
      const n = stack.pop();
      for (const name of (down ? model.outQ : model.inQ).get(n) || []) {
        const q = model.queues.get(name);
        if (q.st === 'idle') continue;
        edges.add(name);
        const m = down ? q.b : q.a;
        if (!nodes.has(m)) { nodes.add(m); stack.push(m); }
      }
    }
  }
  return {nodes, edges};
}
// A group's nodes and their direct neighbours through flowing queues.
export function groupPath(model, gid) {
  const g = model.groups.get(gid), nodes = new Set(g?.nodes || []), edges = new Set();
  for (const n of g?.nodes || []) for (const name of [...(model.inQ.get(n) || []), ...(model.outQ.get(n) || [])]) {
    const q = model.queues.get(name);
    if (q.st !== 'idle') { edges.add(name); nodes.add(q.a); nodes.add(q.b); }
  }
  return {nodes, edges};
}
export const selectionPath = (model, sel) => !sel ? null : sel.kind === 'node' ? (model.meta.has(sel.id) ? nodePath(model, sel.id) : null)
  : model.groups.has(sel.id) ? groupPath(model, sel.id) : null;

export const nodeInputs = (model, n) => (model.inQ.get(n) || []).map(name => model.queues.get(name));
export const nodeOutputs = (model, n) => (model.outQ.get(n) || []).map(name => model.queues.get(name));
// Queues entering a group from other groups.
export const groupInputs = (model, gid) => {
  const g = model.groups.get(gid);
  return (g?.nodes || []).flatMap(n => nodeInputs(model, n)).filter(q => model.meta.get(q.a).g !== gid);
};

// Drop health: the share of frames lost between nodes since start (moved = frames offered to the queues), and a level
// from the worse of that share and the share the worst queue lost since the previous sample.
export function summarizeInstance(model) {
  const s = {nodes: model.nodeCount, off: 0, queues: model.edges.length, flowing: 0, dropped: 0, dropQueues: 0, skipped: 0, full: 0,
    droppingNow: 0, moved: 0, share: 0, level: 'ok'};
  let worstNow = 0;
  for (const m of model.meta.values()) if (!m.w) s.off++;
  for (const q of model.edges) {
    if (q.st !== 'idle') s.flowing++;
    if (q.st === 'full') s.full++;
    if (q.st === 'drop') { s.droppingNow++; worstNow = Math.max(worstNow, q.drD / Math.max(1, (q.enqD || 0) + q.drD)); }
    if (q.nb) { s.skipped += q.dr; continue; }
    s.moved += (q.enq || 0) + (q.dr || 0);
    if (q.dr > 0) { s.dropped += q.dr; s.dropQueues++; }
  }
  s.share = s.moved ? s.dropped / s.moved : 0;
  const worst = Math.max(s.share, worstNow);
  s.level = worst > TILE_DROP_SHARE ? 'bad' : worst >= DROP_WARN_SHARE ? 'warn' : 'ok';
  return s;
}

// Search: groups first, then nodes; prefix matches before substring matches, shorter ids first.
export function searchEntries(model) {
  return [...[...model.groups.keys()].map(id => ({id, kind: 'group', sub: 'group'})), ...[...model.meta.values()].map(m => ({id: m.n, kind: 'node', sub: m.t}))];
}
export function search(entries, text, limit = 10) {
  const s = text.trim().toLowerCase();
  if (!s) return [];
  return entries.filter(x => x.id.toLowerCase().includes(s) || x.sub.includes(s))
    .sort((a, b) => (a.id.startsWith(s) ? 0 : 1) - (b.id.startsWith(s) ? 0 : 1) || a.id.length - b.id.length).slice(0, limit);
}

// One model per nodes.json response, shared by every view so a queues.json sample is applied exactly once. A new
// nodes.json keeps the previous drop counters, so the first sample after it still has deltas.
const shared = new WeakMap(), NONE = [];
let lastShared = null;
export function graphModel(nodes, queues) {
  nodes = Array.isArray(nodes) ? nodes : NONE;
  let model = shared.get(nodes);
  if (!model) { model = buildGraphModel({nodes, queues}, lastShared); shared.set(nodes, model); }
  else updateQueueStats(model, queues);
  lastShared = model;
  return model;
}
