// Run on Linux after rebuilding the addon: node tests/native/fdpass-lifetime.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const net = require('node:net');
const fdpass = require('../../addons/fdpass');
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));

async function until(predicate) {
  for (let i = 0; i < 200; i++) {
    if (predicate()) return;
    await wait(5);
  }
  assert.fail('timed out waiting for transport');
}

function ack(kind, frame = 0n) {
  const bytes = Buffer.alloc(16);
  bytes.write('ACK1');
  bytes.writeUInt32LE(kind, 4);
  bytes.writeBigUInt64LE(frame, 8);
  return bytes;
}

async function scenario(name, run) {
  const path = `/tmp/fdpass-${process.pid}-${name}.sock`;
  const releases = [], clients = [];
  const fd = fs.openSync('/dev/zero', 'r');
  fdpass.setReleaseCallback(path, (frame, reusable) => releases.push([frame, reusable]));
  assert.equal(fdpass.createServer(path), true);
  const send = async frame => {
    const bytes = Buffer.alloc(48);
    bytes.writeBigUInt64LE(frame, 40);
    return fdpass.broadcastFd(path, fd, bytes);
  };
  const connect = async (read = true) => {
    const client = net.createConnection(path);
    clients.push(client);
    await new Promise((resolve, reject) => {
      client.once('connect', resolve);
      client.once('error', reject);
    });
    if (read) client.resume();
    await wait(20); // accept() runs on the addon's polling thread.
    return client;
  };
  try {
    await run({path, releases, send, connect});
    console.log(`PASS ${name}`);
  } finally {
    clients.forEach(client => client.destroy());
    fdpass.closeServer(path);
    fs.closeSync(fd);
  }
}

(async () => {
  await scenario('drain', async ({releases, send, connect}) => {
    const client = await connect();
    assert.equal((await send(1n)).sent, 1);
    await wait(20);
    assert.deepEqual(releases, []);
    client.write(ack(0, 1n));
    await until(() => releases.length === 1);
    assert.deepEqual(releases, [[1n, true]]);
    await send(2n);
    client.write(ack(1));
    await wait(20);
    assert.equal((await send(3n)).clients, 0);
    await until(() => releases.length === 2);
    assert.deepEqual(releases, [[1n, true], [3n, true]]);
    client.write(ack(2));
    await until(() => releases.length === 3);
    assert.deepEqual(releases[2], [2n, true]);
  });
  await scenario('disconnect', async ({path, releases, send, connect}) => {
    const first = await connect(), second = await connect();
    assert.equal((await send(4n)).sent, 2);
    first.write(ack(0, 4n));
    second.destroy();
    await until(() => releases.length === 1);
    assert.deepEqual(releases, [[4n, false]]);
    assert.deepEqual(fdpass.closeServer(path), [4n]);
  });
  await scenario('stop', async ({path, send, connect}) => {
    await connect();
    await send(5n);
    assert.deepEqual(fdpass.closeServer(path), [5n]);
  });
  await scenario('backpressure', async ({send, connect}) => {
    await connect(false);
    let dropped = 0;
    for (let frame = 0n; frame < 50000n; frame++) {
      const result = await send(frame);
      assert.equal(result.clients, 1);
      assert.equal(result.sent + result.backpressure, 1);
      dropped += result.backpressure;
    }
    assert.ok(dropped > 0, 'paused consumer must eventually fill the socket');
  });
})().catch(error => { console.error(error); process.exitCode = 1; });
