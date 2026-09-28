import test from 'node:test';
import assert from 'node:assert/strict';
import { instanceRestarted } from './instanceRestart.mjs';

const list = (...entries) => entries.map(([id, generation]) => ({ id, name: id, host: '127.0.0.1', port: 1, generation }));

test('a changed generation of the selected instance is a restart', () => {
  assert.equal(instanceRestarted(list(['mixer', 5], ['b', 1]), list(['mixer', 6], ['b', 1]), 'mixer'), true);
});

test('other instances and unchanged generations are not', () => {
  assert.equal(instanceRestarted(list(['mixer', 5], ['b', 1]), list(['mixer', 5], ['b', 2]), 'mixer'), false);
  assert.equal(instanceRestarted(list(['mixer', 5]), list(['mixer', 5], ['b', 1]), 'mixer'), false);
});

test('appearing, disappearing and generation-less instances are selection changes, not restarts', () => {
  assert.equal(instanceRestarted([], list(['mixer', 5]), 'mixer'), false);
  assert.equal(instanceRestarted(list(['mixer', 5]), [], 'mixer'), false);
  assert.equal(instanceRestarted(list(['mixer', undefined]), list(['mixer', undefined]), 'mixer'), false);
  assert.equal(instanceRestarted(undefined, list(['mixer', 5]), null), false);
});
