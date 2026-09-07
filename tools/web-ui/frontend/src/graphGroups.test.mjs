import assert from 'node:assert/strict';
import test from 'node:test';
import { groupGraph, focusGroup } from './graphGroups.mjs';
const node = (name, group, src, dst, extra = {}) => ({name, working: true, params: {group, src, dst, ...extra}});
const nodes = [node('demux0', 'input_0', 'packet0', undefined, {routing: {'v:0': 'decoded0'}}),
  node('decode0', 'input_0', 'decoded0', 'video0'), node('decode1', 'input_1', 'packet1', 'video1'),
  node('comp', 'mix', ['video0', 'video1'], 'program'), node('enc', 'output', 'program', 'out')];
const queues = ['video0', 'video1'].map(name => ({name, capacity: 4, occupied: 1, pps: 60}));
test('overview bundles input family and conserves all nodes and queues', () => {
  const g = groupGraph(nodes, queues);
  assert.equal(g.nodes.length, 3);
  assert.equal(g.nodeCount, 5);
  assert.equal(g.queueCount, 7);
  assert.equal(g.internalCount, 1);
  assert.equal(g.queues.reduce((n, q) => n + q.members.length, 0) + g.internalCount, 7);
  const bundle = g.queues.find(q => q.members.includes('video0'));
  assert.deepEqual(bundle.members, ['video0', 'video1']);
  assert.equal(bundle.pps, 120);
  assert.equal(bundle.capacity, 8);
});
test('family and group expansion exposes native node identities', () => {
  const g = groupGraph(nodes, queues, new Set(['family:input_*', 'group:input_0']));
  assert(g.nodes.some(n => n.name === 'demux0'));
  assert(g.nodes.some(n => n.name === 'decode0'));
  assert.equal(g.internalCount, 0);
  assert(g.queues.some(q => q.name === 'decoded0'));
  assert(g.membership.has('@view:group:input_1'));
});
test('ungrouped nodes and same-node feedback remain explicit', () => {
  const g = groupGraph([node('a', undefined, ['loop'], ['loop', 'unused'])]);
  assert.equal(g.membership.size, 0);
  assert.equal(g.queueCount, 2);
  assert.equal(g.internalCount, 0);
  assert.deepEqual(g.nodes[0].params.src, ['loop']);
});
test('shared queue is counted once despite multiple consumers', () => {
  const g = groupGraph([node('a', 'first', [], 'q'), node('b', 'second', 'q', []), node('c', 'third', 'q', [])]);
  assert.equal(g.queueCount, 1);
  assert.equal(g.queues.length, 1);
  assert.equal(g.nodes.filter(n => n.params.src.includes('q')).length, 2);
});

test('focused group excludes sibling chains and bundles boundary inputs', async () => {
  const {focusGroup} = await import('./graphGroups.mjs');
  const view = focusGroup(nodes, queues, 'mix');
  assert.equal(view.nodes.length, 3);
  assert(view.nodes.some(n => n.name === 'comp'));
  assert(!view.nodes.some(n => n.name === 'decode0'));
  assert.equal(view.queues.find(q => q.members.includes('video0')).members.length, 2);
});

test('focused group preserves internal fanout also consumed outside the group', () => {
  const graph = [node('decoder', 'input_0', 'packets', 'video'),
    node('monitor', 'input_0', 'video', 'monitor_out'),
    node('compositor', 'mixer', 'video', 'program')];
  const view = focusGroup(graph, [], 'input_0');
  const video = view.queues.find(queue => queue.members.includes('video'));
  assert(video);
  assert.equal(view.nodes.filter(n => n.params.src.includes(video.name)).length, 2);
  assert(view.nodes.some(n => n.name === 'decoder'));
  assert(view.nodes.some(n => n.name === 'monitor'));
  assert(!view.nodes.some(n => n.name === 'compositor'));
});

test('each input in a sixteen-chain graph can be focused without exposing siblings', () => {
  const inputs = Array.from({length: 16}, (_, i) => [
    node(`source_${i}`, `input_${i}`, [], `packets_${i}`),
    node(`decode_${i}`, `input_${i}`, `packets_${i}`, `frame_${i}`),
    node(`fanout_${i}`, `input_${i}`, `frame_${i}`, [`a_${i}`, `b_${i}`]),
  ]).flat();
  const graph = [...inputs,
    node('a', 'mixer_a', inputs.filter(n => n.name.startsWith('fanout')).map((_, i) => `a_${i}`), 'scene_a'),
    node('b', 'mixer_b', inputs.filter(n => n.name.startsWith('fanout')).map((_, i) => `b_${i}`), 'scene_b')];
  for (let i = 0; i < 16; i++) {
    const view = focusGroup(graph, [], `input_${i}`);
    assert.deepEqual(view.nodes.filter(n => !view.membership.has(n.name)).map(n => n.name),
      [`source_${i}`, `decode_${i}`, `fanout_${i}`]);
    assert.equal(view.queueCount, 4);
    assert.equal(view.queues.reduce((count, q) => count + q.members.length, 0), 4);
  }
  const overview = groupGraph(graph);
  assert.equal(overview.nodes.length, 3);
  assert.equal(overview.membership.get('@view:family:input_*').nodes.length, 48);
});
