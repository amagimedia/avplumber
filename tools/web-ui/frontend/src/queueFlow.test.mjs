import assert from 'node:assert/strict';
import test from 'node:test';
import { sampleQueueFlow, summarizeQueueFlow } from './queueFlow.mjs';
import { groupGraph } from './graphGroups.mjs';
const queue = (extra = {}) => ({name: 'frame', type: 'VideoFrame', capacity: 10, occupied: 0,
  enqueued_total: 100, dequeued_total: 100, dropped_total: 0, ...extra});
const observe = (extra, previous) => sampleQueueFlow([queue(extra)], previous);
const state = samples => summarizeQueueFlow(samples).state;

test('new, missing and reset counters are unknown; an observed empty queue is idle', () => {
  const first = observe({});
  assert.equal(state(first), 'unknown');
  assert.equal(state(observe({}, first)), 'idle');
  assert.equal(state(observe({dequeued_total: 0}, first)), 'unknown');
  assert.equal(state(observe({dequeued_total: undefined}, first)), 'unknown');
  assert.equal(state(sampleQueueFlow([{name: 'legacy', pps: 60}])), 'unknown');
});

test('flow uses actual dequeue deltas, not an assumed frame rate', () => {
  const next = observe({dequeued_total: 101, enqueued_total: 101}, observe({}));
  assert.equal(state(next), 'flowing');
  assert.equal(summarizeQueueFlow(next).active, 1);
});

test('stable buffering with dequeue activity stays flowing; growth requires two intervals', () => {
  let samples = observe({occupied: 5});
  samples = observe({occupied: 5, enqueued_total: 160, dequeued_total: 160}, samples);
  assert.equal(state(samples), 'flowing');
  samples = observe({occupied: 6, enqueued_total: 221, dequeued_total: 220}, samples);
  assert.equal(state(samples), 'flowing');
  samples = observe({occupied: 7, enqueued_total: 282, dequeued_total: 280}, samples);
  assert.equal(state(samples), 'backlog');
  assert.equal(summarizeQueueFlow(samples).growing, 1);
});

test('nonempty queue not draining is descriptive backlog; recovery clears it', () => {
  let samples = observe({occupied: 2});
  samples = observe({occupied: 2}, samples);
  assert.equal(state(samples), 'idle');
  samples = observe({occupied: 2}, samples);
  assert.equal(state(samples), 'backlog');
  assert.equal(summarizeQueueFlow(samples).held, 1);
  assert.equal(state(observe({occupied: 0, dequeued_total: 102}, samples)), 'flowing');
});

test('only newly observed drops are red, with exact count and reset recovery', () => {
  let samples = observe({dropped_total: 200});
  samples = observe({dropped_total: 203}, samples);
  assert.equal(state(samples), 'dropped');
  assert.equal(summarizeQueueFlow(samples).dropped, 3);
  assert.equal(state(observe({dropped_total: 203}, samples)), 'idle');
  assert.equal(state(observe({dropped_total: 0}, samples)), 'unknown');
});

test('bundle keeps worst queue, unknown members, and one failing stream among fifteen healthy', () => {
  const healthy = observe({dequeued_total: 160, enqueued_total: 160}, observe({}))[0];
  const blocked = {...healthy, occupied: 10, flow: {...healthy.flow, active: false, held: true}};
  const summary = summarizeQueueFlow([...Array(15).fill(healthy), blocked, undefined]);
  assert.equal(summary.state, 'backlog');
  assert.equal(summary.active, 15);
  assert.equal(summary.held, 1);
  assert.equal(summary.known, 16);
  assert.equal(summary.total, 17);
  assert.equal(summary.maxFill, 1);
});

test('group projection preserves singleton diagnostics and global queue aliases', () => {
  const samples = observe({dequeued_total: 160, enqueued_total: 160}, observe({}));
  const graph = [{name: 'producer', params: {group: 'in', dst: '@frame'}},
    {name: 'consumer', params: {group: 'out', src: '@frame'}}];
  const projected = groupGraph(graph, samples).queues[0];
  assert.equal(projected.type, 'VideoFrame');
  assert.equal(projected.dequeued_total, 160);
  assert.equal(projected.flowSummary.state, 'flowing');
});

test('group projection surfaces a single dropping constituent instead of averaging it away', () => {
  const healthy = observe({dequeued_total: 160, enqueued_total: 160}, observe({}))[0];
  const dropping = {...healthy, name: 'other', occupied: 10, flow: {...healthy.flow, dropped: 1}};
  const graph = [{name: 'producer', params: {group: 'in', dst: ['frame', 'other', 'missing']}},
    {name: 'consumer', params: {group: 'out', src: ['frame', 'other', 'missing']}}];
  const bundle = groupGraph(graph, [healthy, dropping]).queues[0];
  assert.equal(bundle.flowSummary.state, 'dropped');
  assert.equal(bundle.flowSummary.dropping, 1);
  assert.equal(bundle.flowSummary.maxFill, 1);
  assert.equal(bundle.flowSummary.known, 2);
  assert.equal(bundle.flowSummary.total, 3);
});
