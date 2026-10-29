import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {CW, buildGraphModel, updateQueueStats, nodePath, groupPath, groupInputs, summarizeInstance, searchEntries, search, graphModel} from './model.mjs';

// A small mixer-shaped graph: 16 numbered inputs fanning out (non-blocking) to a program compositor and eight AUX
// compositors, a clip source, a program chain with a tee to the outputs and a multiview.
const INPUTS = 16, AUX = 8;
const range = n => [...Array(n).keys()];
function synthetic({mixerNames = true} = {}) {
  const core = mixerNames ? 'mixer' : 'program', out = mixerNames ? 'output' : 'egress';
  const nodes = [];
  const node = (name, type, group, params = {}, extra = {}) => nodes.push({name, type, params: {name, type, group, ...params}, working: true, ...extra});
  for (const i of range(INPUTS)) {
    node(`in_${i}`, 'input_rec', `input_${i}`, {dst: `input_${i}_pkt`});
    node(`dec_${i}`, 'dec_video', `input_${i}`, {src: `input_${i}_pkt`, dst: `input_${i}_dec`});
    node(`fan_${i}`, 'one_to_many_nb', `input_${i}`, {src: `input_${i}_dec`, dst: [`comp_src_${i}`, ...range(AUX).map(k => `aux${k}_src_${i}`)]});
  }
  node('clip_in', 'input_rec', 'clip_load', {dst: 'clip_pkt'});
  node('clip_dec', 'dec_video', 'clip_load', {src: 'clip_pkt', dst: 'clip_out'});
  node('wipe', 'filter_video', 'wipe', {src: 'clip_out', dst: 'wipe_out'});
  node('comp', 'mixer_compositor', core, {src: [...range(INPUTS).map(i => `comp_src_${i}`), 'wipe_out'], dst: 'mix_out', active_inputs: '1010000000000000'});
  node('follow', 'mixer_pvw_follow', core);
  node('tee', 'one_to_many', out, {src: 'mix_out', dst: ['enc_in', 'mv_in']});
  node('enc', 'enc_video', out, {src: 'enc_in', dst: 'enc_out'});
  node('sink', 'output', out, {src: 'enc_out'});
  for (const k of range(AUX)) node(`aux${k}_comp`, 'mixer_compositor', `aux_aux${k}`, {src: range(INPUTS).map(i => `aux${k}_src_${i}`), active_inputs: k === 0 ? '1' : '0'}, k === 7 ? {working: false} : {});
  node('mv_comp', 'mixer_compositor', 'aux_mv', {src: ['mv_in'], dst: 'mv_out', active_inputs: '1'});
  node('mv_sink', 'output', 'aux_mv', {src: 'mv_out'});
  return nodes;
}
const queueNames = nodes => [...new Set(nodes.flatMap(n => [n.params.src, n.params.dst].flat().filter(Boolean)))];
// Everything flows at 30 pps unless overridden; AUX taps 1+ are unsubscribed (idle).
function sample(nodes, over = {}) {
  return queueNames(nodes).map(name => ({name, capacity: 4, occupied: 1, pps: /^aux[1-7]_src/.test(name) ? 0 : 30,
    dropped_total: 0, enqueued_total: 3000, dequeued_total: 2999, frames_in_queue: {avg: 1, max: 2, cur: 1},
    type: /_pkt$/.test(name) ? 'av::Packet' : 'av::VideoFrame', ...over[name]}));
}
const build = (opts, over) => { const nodes = synthetic(opts); return {nodes, model: buildGraphModel({nodes, queues: sample(nodes, over)})}; };
const orthogonal = pts => { for (let k = 0; k + 3 < pts.length; k += 2) if (pts[k] !== pts[k + 2] && pts[k + 1] !== pts[k + 3]) return false; return true; };

test('layering: columns are the longest path from a source, sources and unconnected nodes sit before their consumers', () => {
  const {model} = build();
  const m = n => model.meta.get(n);
  // 17 source groups in two bands; the bands are as wide as the deepest input column plus a two-column lane gap.
  const inW = 2 + 2;
  assert.equal(m('in_0').x, 0); assert.equal(m('dec_0').x, 1); assert.equal(m('fan_0').x, 2);
  const band1 = [...model.areaById.get('inputs').groups].slice(9).map(g => g.id);
  assert.ok(band1.includes('input_15'));
  assert.equal(m('fan_15').x, 2 + inW);
  assert.equal(m('comp').x, 3 + 2 * inW);
  assert.equal(m('follow').x, m('comp').x - 1, 'an unconnected node sits one column before the rest of its group');
  assert.equal(m('tee').x, m('comp').x + 1);
  assert.equal(m('sink').x, m('comp').x + 3);
  assert.equal(m('mv_comp').x, m('tee').x + 1);
});

test('areas: numbered source family plus other source groups, other families by name token, the rest is core', () => {
  const {model} = build();
  assert.deepEqual(model.areas.map(z => z.d), ['inputs', 'core', 'aux']);
  const ids = d => model.areaById.get(d).groups.map(g => g.id);
  assert.deepEqual(new Set(ids('inputs')), new Set([...range(INPUTS).map(i => `input_${i}`), 'clip_load']));
  assert.deepEqual(new Set(ids('aux')), new Set([...range(AUX).map(k => `aux_aux${k}`), 'aux_mv']));
  assert.deepEqual(ids('core'), ['mixer', 'output', 'wipe'], 'mixer hint order first, then by name');
  const [inputs, core, aux] = model.areas;
  assert.equal(inputs.title, 'Inputs'); assert.equal(inputs.count, `${INPUTS} sources · ${INPUTS} NVDEC`);
  assert.equal(core.title, 'Program core'); assert.equal(core.tag, 'mixer'); assert.equal(core.count, '3 groups');
  assert.equal(aux.title, 'AUX + multiview');
  // inputs: one row per group, bands side by side; core starts below the top; one area gap before AUX
  const inGroups = inputs.groups;
  assert.equal(inGroups[0].id, 'clip_load', 'unhinted groups sort by family name');
  assert.deepEqual([inGroups[0].y0, inGroups[1].y0, inGroups[9].y0, inGroups[10].y0], [0, 1, 0, 1], 'nine groups per band');
  assert.ok(inGroups[9].x0 > inGroups[0].x1);
  assert.ok(model.groups.get('mixer').y0 > 0);
  assert.ok(aux.y0 - core.y1 >= 9);
  assert.equal(model.groups.get('output').zone, 'out');
  assert.equal(model.groups.get('mixer').zone, 'gpu');
  assert.equal(model.groups.get('aux_aux3').zone, 'ctl');
  assert.equal(model.groups.get('input_3').zone, 'src');
});

test('tile names: mixer hints, numbered members by number, generic fallback strips the area token', () => {
  const {model} = build();
  const name = g => model.groups.get(g).name;
  assert.equal(name('input_7'), '7'); assert.equal(name('aux_aux3'), 'AUX 3'); assert.equal(name('aux_mv'), 'MV');
  assert.equal(name('mixer'), 'Program'); assert.equal(name('output'), 'Outputs'); assert.equal(name('wipe'), 'wipe');
  const generic = build({mixerNames: false}).model;
  assert.equal(generic.areaById.get('core').title, 'Core');
  assert.equal(generic.areaById.get('core').tag, 'core');
  assert.equal(generic.groups.get('program').name, 'program');
  assert.equal(generic.groups.get('egress').zone, 'out', 'a core group with an output node is coloured as output');
});

test('tiles: one square tile per group inside its area frame, no overlaps', () => {
  const {model} = build();
  for (const z of model.areas) {
    const r = z.rect;
    for (const g of z.groups) {
      const t = g.tile;
      assert.ok(Math.abs(t.w - t.h) < 1e-9, `${g.id} tile is square`);
      assert.ok(t.x >= r.x0 - 1e-9 && t.x + t.w <= r.x1 + 1e-9 && t.y >= r.bot - 1e-9 && t.y + t.h <= r.top + 1e-9, `${g.id} inside its area`);
    }
    const ts = z.groups.map(g => g.tile);
    for (let i = 0; i < ts.length; i++) for (let j = i + 1; j < ts.length; j++) {
      const a = ts[i], b = ts[j];
      assert.ok(a.x + a.w <= b.x + 1e-9 || b.x + b.w <= a.x + 1e-9 || a.y + a.h <= b.y + 1e-9 || b.y + b.h <= a.y + 1e-9, 'tiles do not overlap');
    }
  }
});

test('routes: orthogonal, from producer to consumer, one lane per target', () => {
  const {model} = build();
  for (const q of model.edges) {
    const pts = model.routes.get(q.n), a = model.meta.get(q.a), b = model.meta.get(q.b);
    assert.ok(orthogonal(pts), `${q.n} is orthogonal`);
    assert.deepEqual(pts.slice(0, 2), [a.x * CW, -a.y]);
    assert.deepEqual(pts.slice(-2), [b.x * CW, -b.y]);
  }
  // All 17 queues into the program compositor share the vertical lane just before it; another target in the same
  // column gets another lane.
  const lane = name => { const p = model.routes.get(name); return p[p.length - 4]; };
  const lanes = new Set(model.inQ.get('comp').map(lane));
  assert.equal(lanes.size, 1);
  const sameCol = [...model.meta.values()].filter(m => m.x === model.meta.get('mv_comp').x && model.inQ.has(m.n) && m.n !== 'mv_comp');
  for (const m of sameCol) assert.notEqual(lane(model.inQ.get(m.n)[0]), lane('mv_in'));
  // a straight neighbour inside one row is one segment
  assert.equal(model.routes.get('input_0_pkt').length, 4);
});

test('links: input taps summed into one trunk per consumer group, areas joined by summed links with labels', () => {
  const {model} = build();
  const link = key => model.links.find(r => r.key === key);
  assert.equal(link('area:inputs>mixer').n, INPUTS);
  assert.equal(link('area:inputs>wipe').n, 1);
  assert.equal(link('wipe>mixer').n, 1);
  assert.equal(link('area:inputs>aux_aux0').n, INPUTS);
  assert.equal(link('area:inputs>aux_aux3').active, 0, 'unsubscribed taps do not flow');
  assert.ok(orthogonal(link('mixer>output').pts));
  const byKey = Object.fromEntries(model.areaLinks.map(r => [r.key, r]));
  assert.deepEqual(Object.keys(byKey).sort(), ['core>aux', 'inputs>aux', 'inputs>core']);
  assert.equal(byKey['inputs>core'].active, INPUTS + 1);
  assert.equal(byKey['inputs>core'].pps, 30 * (INPUTS + 1));
  assert.equal(byKey['inputs>aux'].n, INPUTS * AUX);
  assert.equal(byKey['inputs>aux'].active, INPUTS, 'only aux_aux0 is subscribed');
  for (const r of model.areaLinks) { assert.equal(r.pts.length, 4); assert.ok(orthogonal(r.pts)); assert.equal(r.mid.length, 2); }
});

test('queue states: drop deltas, non-blocking skips, near full, idle', () => {
  const {nodes, model} = build({}, {input_0_dec: {dropped_total: 50}, comp_src_1: {dropped_total: 9}});
  const q = name => model.queues.get(name);
  assert.equal(q('input_0_dec').st, 'ok', 'drops before the first sample are history, not red');
  assert.equal(q('aux3_src_0').st, 'idle');
  const next = sample(nodes, {
    input_0_dec: {dropped_total: 80, enqueued_total: 3030},        // 30 of 30 lost: red
    comp_src_1: {dropped_total: 20, enqueued_total: 3030},         // a non-blocking fan-out skipping: by design
    input_2_dec: {frames_in_queue: {avg: 3.7, max: 4}},             // near full
    input_3_dec: {dropped_total: 0, enqueued_total: 10},            // counters reset by a restart: no delta
  });
  assert.equal(updateQueueStats(model, next), true);
  assert.equal(updateQueueStats(model, next), false, 'the same sample is applied once');
  assert.equal(q('input_0_dec').st, 'drop'); assert.equal(q('input_0_dec').drD, 30);
  assert.equal(q('comp_src_1').nb, true); assert.equal(q('comp_src_1').st, 'ok'); assert.equal(q('comp_src_1').drD, 11);
  assert.equal(q('input_2_dec').st, 'full');
  assert.equal(q('input_3_dec').enqD, null); assert.equal(q('input_3_dec').st, 'ok');
  assert.equal(model.groups.get('input_0').st, 'drop');
  assert.equal(model.groups.get('input_2').st, 'full');
  assert.equal(model.groups.get('aux_aux5').st, 'idle');
  assert.equal(model.areaLinks.find(r => r.key === 'inputs>core').drop, 0, 'skips are not link drops');
  // a queue missing from a sample is unknown and idle
  updateQueueStats(model, next.filter(s => s.name !== 'mix_out'));
  assert.equal(q('mix_out').known, false); assert.equal(q('mix_out').st, 'idle');
});

test('tile state: a drop of at most 1% of the frames stays on the queue line only', () => {
  const {nodes, model} = build();
  updateQueueStats(model, sample(nodes, {input_5_dec: {dropped_total: 1, enqueued_total: 3200}}));
  assert.equal(model.queues.get('input_5_dec').st, 'drop');
  assert.equal(model.groups.get('input_5').st, 'ok');
  updateQueueStats(model, sample(nodes, {input_5_dec: {dropped_total: 10, enqueued_total: 3300}}));
  assert.equal(model.groups.get('input_5').st, 'drop');
});

test('drawn now: compositors\' active_inputs map onto the groups producing their inputs', () => {
  const {model} = build();
  assert.deepEqual(model.drawing.get('comp'), ['input_0', 'input_2']);
  assert.deepEqual(model.onAir.get('input_0'), ['comp', 'aux0_comp']);
  assert.deepEqual(model.onAir.get('input_2'), ['comp']);
  assert.equal(model.onAir.has('input_1'), false);
  assert.deepEqual(model.onAir.get('output'), ['mv_comp']);
  assert.deepEqual(model.drawing.get('aux1_comp'), []);
});

test('paths follow flowing queues up- and downstream', () => {
  const {model} = build();
  const p = nodePath(model, 'dec_0');
  for (const n of ['in_0', 'fan_0', 'comp', 'tee', 'sink', 'mv_sink', 'aux0_comp']) assert.ok(p.nodes.has(n), n);
  assert.equal(p.nodes.has('aux3_comp'), false, 'idle taps are not part of the path');
  assert.equal(p.nodes.has('dec_1'), false, 'siblings are upstream of a shared consumer only');
  const g = groupPath(model, 'output');
  assert.deepEqual(new Set(g.nodes), new Set(['tee', 'enc', 'sink', 'comp', 'mv_comp']));
  assert.equal(groupInputs(model, 'output').length, 1);
});

test('instance summary, search and the shared model', () => {
  const {nodes, model} = build({}, {comp_src_0: {dropped_total: 4}, input_9_dec: {dropped_total: 7}});
  const s = summarizeInstance(model);
  assert.equal(s.nodes, nodes.length); assert.equal(s.off, 1);
  assert.equal(s.queues, model.edges.length);
  assert.equal(s.dropped, 7); assert.equal(s.dropQueues, 1); assert.equal(s.skipped, 4); assert.equal(s.droppingNow, 0);
  const hits = search(searchEntries(model), 'aux_aux');
  assert.equal(hits[0].kind, 'group'); assert.equal(hits.length, AUX);
  assert.deepEqual(search(searchEntries(model), 'mixer_comp').map(h => h.id), ['comp', 'mv_comp', 'aux0_comp', 'aux1_comp', 'aux2_comp', 'aux3_comp', 'aux4_comp', 'aux5_comp', 'aux6_comp', 'aux7_comp'],
    'type matches count too; shorter ids first');

  const shared = synthetic(), q1 = sample(shared), q2 = sample(shared, {input_0_dec: {dropped_total: 5, enqueued_total: 3030}});
  const a = graphModel(shared, q1);
  assert.equal(graphModel(shared, q1), a);
  graphModel(shared, q2);
  assert.equal(a.queues.get('input_0_dec').drD, 5);
  // a refreshed nodes.json keeps the counters, so the next sample still has deltas
  const b = graphModel(synthetic(), sample(shared, {input_0_dec: {dropped_total: 9, enqueued_total: 3060}}));
  assert.notEqual(b, a);
  assert.equal(b.queues.get('input_0_dec').drD, 4);
});

test('any graph: empty, without numbered families, with a cycle', () => {
  const empty = buildGraphModel({nodes: [], queues: []});
  assert.equal(empty.nodeCount, 0); assert.deepEqual(empty.areas, []); assert.deepEqual(summarizeInstance(empty).nodes, 0);
  assert.equal(graphModel(null, null).nodeCount, 0);
  assert.equal(graphModel(undefined, []), graphModel(null, []), 'a missing nodes list does not rebuild on every poll');

  const nodes = [
    {name: 'src', type: 'input', params: {dst: 'a'}},
    {name: 'loop', type: 'filter_video', params: {group: 'proc', src: ['a', 'back'], dst: 'b'}},
    {name: 'tail', type: 'filter_video', params: {group: 'proc', src: 'b', dst: 'back'}},
    {name: 'idle', type: 'filter_video', params: {group: 'proc'}},
  ];
  const m = buildGraphModel({nodes, queues: [{name: 'a', pps: 25, capacity: 2}, {name: 'b', pps: 25, capacity: 2}]});
  assert.deepEqual(m.areas.map(z => [z.d, z.title, z.tag]), [['core', 'Core', 'core']]);
  assert.equal(m.groups.get('ungrouped').name, 'ungrouped');
  assert.ok(m.meta.get('tail').x > m.meta.get('loop').x, 'a cycle is cut, not followed forever');
  assert.equal(m.meta.get('idle').x, Math.max(0, Math.min(m.meta.get('loop').x, m.meta.get('tail').x) - 1));
  assert.equal(m.queues.get('back').known, false);
  assert.equal(m.queues.get('back').st, 'idle', 'a queue missing from queues.json is idle');
  for (const q of m.edges) assert.ok(orthogonal(m.routes.get(q.n)));
});

// A real 192-input mixer snapshot, when AVP_GRAPH_DUMP names one: a JSON file {"nodes.json": {body}, "queues.json":
// {body}} of the engine's replies. A data.json beside it (the approved mock-up's preprocessed layout) must be
// reproduced exactly.
const DUMP = process.env.AVP_GRAPH_DUMP || '', MOCK = DUMP.replace(/[^/]*$/, 'data.json');
test('real 192-input mixer dump', {skip: !(DUMP && fs.existsSync(DUMP)) && 'set AVP_GRAPH_DUMP to a nodes/queues dump'}, () => {
  const dump = JSON.parse(fs.readFileSync(DUMP, 'utf8'));
  const nodes = JSON.parse(dump['nodes.json'].body), queues = JSON.parse(dump['queues.json'].body);
  const model = buildGraphModel({nodes, queues});
  assert.equal(model.nodeCount, nodes.length);
  assert.equal(model.edges.length, 5400);
  assert.deepEqual(model.areas.map(z => [z.d, z.title]), [['inputs', 'Inputs'], ['core', 'Program core'], ['aux', 'AUX + multiview']]);
  assert.equal(model.areaById.get('inputs').count, '192 sources · 105 NVDEC · 40 browser · 47 raw');
  assert.equal(model.groups.get('mixer_wipe_load').d, 'inputs');
  assert.equal(model.groups.get('aux_mv2').name, 'MV 2');
  assert.equal(model.groups.get('aux_aux12').name, 'AUX 12');
  for (const q of model.edges) assert.ok(orthogonal(model.routes.get(q.n)), q.n);
  assert.equal([...model.queues.values()].filter(q => q.st === 'drop').length, 0, 'no drops before a second sample');
  const idle = queues.filter(q => q.pps === 0).length;
  assert.equal([...model.queues.values()].filter(q => q.known && q.st === 'idle').length, idle);
  // drawn now: compositor input i is input_i for the 192 sources
  for (const n of nodes) {
    const bits = n.params.active_inputs;
    if (n.type !== 'mixer_compositor' || typeof bits !== 'string') continue;
    const expected = [...bits].flatMap((b, i) => b === '1' && i < 192 ? [`input_${i}`] : []);
    assert.deepEqual(model.drawing.get(n.name).filter(g => g.startsWith('input_')), expected, n.name);
  }
  // a second sample: one input queue losing frames, every fan-out skipping
  const next = queues.map(q => ({...q, enqueued_total: q.enqueued_total + 30,
    dropped_total: q.dropped_total + (q.name === 'input_7_decoded' ? 30 : model.queues.get(q.name).nb ? 2 : 0)}));
  updateQueueStats(model, next);
  assert.deepEqual([...model.queues.values()].filter(q => q.st === 'drop').map(q => q.n), ['input_7_decoded']);
  assert.equal(model.groups.get('input_7').st, 'drop');

  if (!fs.existsSync(MOCK)) return;
  const mock = JSON.parse(fs.readFileSync(MOCK, 'utf8'));
  for (const n of mock.nodes) assert.deepEqual([model.meta.get(n.n).x, model.meta.get(n.n).y], [n.x, n.y], n.n);
  for (const g of mock.groups) {
    const m = model.groups.get(g.id);
    assert.deepEqual([m.d, m.x0, m.x1, m.y0, m.y1, m.n], [g.d, g.x0, g.x1, g.y0, g.y1, g.n], g.id);
  }
  assert.deepEqual(model.areas.flatMap(z => z.groups.map(g => g.id)), mock.groups.map(g => g.id));
});
