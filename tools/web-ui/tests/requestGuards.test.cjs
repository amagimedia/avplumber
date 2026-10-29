const test = require('node:test');
const assert = require('node:assert/strict');
const { readOnlyCommand, isSafeId, allowedHostnames, upgradeAllowed, isJsonRequest } = require('../backend/requestGuards.js');

test('the read-only queries the viewer and the cut-latency meter send are forwarded', () => {
  for (const command of ['nodes.json', 'queues.json', 'queues.stats', 'sync_groups.json',
    'correction_groups.json', 'node.object.get input_pl0 stream-limits', 'mixer.status mixer']) {
    assert.equal(readOnlyCommand(command), command);
  }
  assert.equal(readOnlyCommand('  node.object.get  in  streams '), 'node.object.get in streams');
});

test('commands that change an instance, and malformed queries, are rejected', () => {
  for (const command of ['node.delete in', 'node.add_start {"type":"output"}', 'group.stop main',
    'queues.stats.reset', 'stats.subscribe {"url":"http://x"}', 'retry nodes.json', 'detach nodes.json',
    'event.wait never', 'queue.drain q', 'NODES.JSON', '', ' ',
    'nodes.json extra', 'node.object.get in', 'node.object.get in streams extra', 'mixer.status',
    'node.object.get in x streams', undefined, null, 42, ['nodes.json'], { toString: () => 'nodes.json' }]) {
    assert.equal(readOnlyCommand(command), null, JSON.stringify(command));
  }
});

test('control characters cannot smuggle a second command line', () => {
  for (const command of ['nodes.json\nnode.delete in', 'nodes.json\r\ngroup.stop main', 'nodes.json\n',
    'mixer.status mixer\rnode.delete x', 'node.object.get in\tstreams', 'nodes.json\u0000', 'nodes.json\u007f']) {
    assert.equal(readOnlyCommand(command), null, JSON.stringify(command));
  }
});

test('arguments must be names of at most 128 characters', () => {
  assert.equal(readOnlyCommand(`mixer.status ${'m'.repeat(128)}`), `mixer.status ${'m'.repeat(128)}`);
  assert.equal(readOnlyCommand('mixer.status @global_mixer'), 'mixer.status @global_mixer');
  for (const command of [`mixer.status ${'m'.repeat(129)}`, `mixer.status ${'m'.repeat(100000)}`,
    'mixer.status {"a":1}', 'node.object.get in/x streams', 'mixer.status mixer;x', 'mixer.status mixér']) {
    assert.equal(readOnlyCommand(command), null, command.slice(0, 40));
  }
});

test('ids that name Object.prototype members are not stored', () => {
  for (const id of ['__proto__', 'constructor', 'prototype', '', undefined, 7]) assert.equal(isSafeId(id), false);
  for (const id of ['default', '127.0.0.1:7777', 'mixer', 'proto']) assert.equal(isSafeId(id), true);
});

test('Host must be loopback, the bind address or WEBUI_ALLOWED_HOSTS', () => {
  const loopback = allowedHostnames('127.0.0.1', '');
  assert.equal(upgradeAllowed({ host: '127.0.0.1:22222' }, loopback), true);
  assert.equal(upgradeAllowed({ host: 'localhost:22222' }, loopback), true);
  assert.equal(upgradeAllowed({ host: '[::1]:22222' }, loopback), true);
  assert.equal(upgradeAllowed({ host: 'LocalHost' }, loopback), true);
  // DNS rebinding: an attacker's name resolving to this server, with its own matching Origin.
  assert.equal(upgradeAllowed({ host: 'rebind.example:22222', origin: 'http://rebind.example:22222' }, loopback), false);
  for (const host of [undefined, '', 'evil.example@127.0.0.1', '127.0.0.1/x', '127.0.0.1 x']) {
    assert.equal(upgradeAllowed({ host }, loopback), false, String(host));
  }

  const wildcard = allowedHostnames('0.0.0.0', ' viewer.example , [fd00::5] ,');
  assert.equal(wildcard.has('0.0.0.0'), false);
  assert.equal(upgradeAllowed({ host: 'viewer.example:7681', origin: 'https://viewer.example:7681' }, wildcard), true);
  assert.equal(upgradeAllowed({ host: '[fd00::5]:22222' }, wildcard), true);
  assert.equal(upgradeAllowed({ host: '10.0.0.5:22222' }, wildcard), false);
  assert.equal(upgradeAllowed({ host: '10.0.0.5:22222' }, allowedHostnames('10.0.0.5', '')), true);
});

test('a browser Origin must name the same hostname as Host; clients without one are accepted', () => {
  const allowed = allowedHostnames('127.0.0.1', '');
  assert.equal(upgradeAllowed({ host: '127.0.0.1:22222' }, allowed), true);
  assert.equal(upgradeAllowed({ host: '127.0.0.1:22222', origin: 'http://127.0.0.1:22222' }, allowed), true);
  // The preview page on another port of the same host (doc/mixer_cut_latency.md).
  assert.equal(upgradeAllowed({ host: '127.0.0.1:22222', origin: 'http://127.0.0.1:8080' }, allowed), true);
  for (const origin of ['http://evil.example', 'http://localhost:22222', 'null', 'file://', '']) {
    assert.equal(upgradeAllowed({ host: '127.0.0.1:22222', origin }, allowed), false, origin);
  }
});

test('POST bodies must be declared JSON', () => {
  assert.equal(isJsonRequest({ 'content-type': 'application/json' }), true);
  assert.equal(isJsonRequest({ 'content-type': 'Application/JSON; charset=utf-8' }), true);
  for (const type of [undefined, '', 'text/plain', 'application/x-www-form-urlencoded', 'multipart/form-data',
    'application/jsonx']) {
    assert.equal(isJsonRequest({ 'content-type': type }), false, String(type));
  }
});
