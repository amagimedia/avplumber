const test = require('node:test');
const assert = require('node:assert/strict');
const { generationCounter, noteConnectionLost, noteHeartbeat, heartbeatExpired } = require('../backend/instanceRestart.js');

test('regular heartbeats keep the generation', () => {
  const next = generationCounter(10);
  const inst = { id: 'mixer', generation: next() };
  assert.equal(noteHeartbeat(inst, next), false);
  assert.equal(noteHeartbeat(inst, next), false);
  assert.equal(inst.generation, 10);
});

test('a heartbeat after a lost control connection is a restart, reported once', () => {
  const next = generationCounter(10);
  const inst = { id: 'mixer', generation: next() };
  noteConnectionLost(inst);
  noteConnectionLost(inst); // refused reconnects while the process is down
  assert.equal(noteHeartbeat(inst, next), true);
  assert.equal(inst.generation, 11);
  assert.equal(noteHeartbeat(inst, next), false);
  assert.equal(inst.generation, 11);
});

test('a connection lost after its instance was removed from the registry is ignored', () => {
  assert.doesNotThrow(() => noteConnectionLost(undefined));
});

test('generations are seeded from the clock, so they change across backend restarts', () => {
  const before = Date.now();
  assert.ok(generationCounter()() >= before);
});

test('an instance created by heartbeats expires after the timeout', () => {
  const inst = { id: '127.0.0.1:22422', usesHeartbeat: true, lastHeartbeat: 1000, explicit: false };
  assert.equal(heartbeatExpired(inst, 1000 + 30000, 30000), false);
  assert.equal(heartbeatExpired(inst, 1000 + 30001, 30000), true);
});

test('an explicit instance survives the timeout and stops being heartbeat-tracked', () => {
  const inst = { id: 'default', usesHeartbeat: true, lastHeartbeat: 1000, explicit: true };
  assert.equal(heartbeatExpired(inst, 1000 + 30001, 30000), false);
  assert.equal(inst.usesHeartbeat, false);
  assert.equal(inst.lastHeartbeat, null);
  assert.equal(heartbeatExpired(inst, 1000 + 90000, 30000), false);
});

test('an instance without heartbeats never expires', () => {
  assert.equal(heartbeatExpired({ id: 'default', usesHeartbeat: false, lastHeartbeat: null, explicit: true }, 1e12, 30000), false);
});
