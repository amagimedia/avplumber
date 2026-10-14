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
  await scenario('rebind', async ({path, send, connect}) => {
    await connect();
    fs.unlinkSync(path);
    assert.equal(fdpass.createServer(path), true);
    assert.ok(fs.existsSync(path), 'a vanished socket file must be re-bound');
    await connect();
    assert.equal((await send(6n)).sent, 2, 'existing and new consumers both receive');
  });
  await scenario('fence', async ({path, releases, connect}) => {
    // An acquire fence travels as a second fd of the same SCM_RIGHTS message; the header says
    // so in bit 31 of its pixel format word. The two must agree, and nothing may leak.
    const FENCE = 0x80000000;
    const header = (frame, fenced) => {
      const bytes = Buffer.alloc(48);
      bytes.writeUInt32LE(((fenced ? FENCE : 0) | 0x42475241) >>> 0, 12);   // | yields a signed 32-bit number
      bytes.writeBigUInt64LE(frame, 40);
      return bytes;
    };
    const fence = fs.openSync('/dev/null', 'r');
    const plane = fs.openSync('/dev/zero', 'r');
    const openFds = () => fs.readdirSync('/proc/self/fd').length;
    const client = await connect();
    await assert.rejects(fdpass.broadcastFd(path, plane, header(1n, true)), /fence/);
    await assert.rejects(fdpass.broadcastFd(path, plane, header(1n, false), fence), /fence/);
    assert.deepEqual(releases, [], 'a refused broadcast is the caller\'s to release');
    const before = openFds();
    assert.equal((await fdpass.broadcastFd(path, plane, header(2n, true), fence)).sent, 1);
    // Not a closed descriptor: the plane's duplicate would take the lowest free number first.
    const unopened = 2 ** 30;
    const failed = await fdpass.broadcastFd(path, plane, header(3n, true), unopened);
    assert.equal(failed.sent, 0);
    assert.equal(failed.errors, 1, 'a fence that cannot be duplicated sends nothing');
    await until(() => releases.length === 1);
    assert.deepEqual(releases, [[3n, true]]);
    assert.equal(openFds(), before, 'no duplicate survives a broadcast');
    client.write(ack(0, 2n));
    await until(() => releases.length === 2);
    fs.closeSync(fence);
    fs.closeSync(plane);
  });
  await scenario('fence-receive', async ({path, releases}) => {
    // Node's net cannot receive SCM_RIGHTS: a Python client counts the fds of a fenced and of a
    // plain message. Skipped without python3.
    const {spawn, spawnSync} = require('node:child_process');
    if (spawnSync('python3', ['-c', 'import socket, array']).status !== 0) {
      console.log('SKIP fence-receive (no python3)');
      return;
    }
    const script = `
import array, os, socket, struct, sys
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect(sys.argv[1])
print('connected', flush=True)
for _ in range(2):
    data, anc, flags, _ = s.recvmsg(48, socket.CMSG_SPACE(2 * 4))
    fds = array.array('i')
    for level, kind, payload in anc:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            fds.frombytes(payload[:len(payload) - len(payload) % 4])
    truncated = bool(flags & socket.MSG_CTRUNC)
    print('msg', len(data), len(fds), struct.unpack_from('<I', data, 12)[0] >> 31, int(truncated), flush=True)
    s.sendall(b'ACK1' + struct.pack('<IQ', 0, struct.unpack_from('<Q', data, 40)[0]))
    for fd in fds:
        os.close(fd)
`;
    const child = spawn('python3', ['-c', script, path], {stdio: ['ignore', 'pipe', 'inherit']});
    const lines = [];
    child.stdout.on('data', chunk => lines.push(...String(chunk).split('\n').filter(Boolean)));
    await until(() => lines.includes('connected'));
    await wait(20);
    const fence = fs.openSync('/dev/null', 'r');
    const plane = fs.openSync('/dev/zero', 'r');
    const fenced = Buffer.alloc(48), plain = Buffer.alloc(48);
    fenced.writeUInt32LE((0x80000000 | 0x42475241) >>> 0, 12);
    fenced.writeBigUInt64LE(10n, 40);
    plain.writeUInt32LE(0x42475241, 12);
    plain.writeBigUInt64LE(11n, 40);
    assert.equal((await fdpass.broadcastFd(path, plane, fenced, fence)).sent, 1);
    assert.equal((await fdpass.broadcastFd(path, plane, plain)).sent, 1);
    await until(() => lines.length >= 3);
    assert.deepEqual(lines.slice(1), ['msg 48 2 1 0', 'msg 48 1 0 0']);
    await until(() => releases.length === 2);
    assert.deepEqual(releases, [[10n, true], [11n, true]]);
    fs.closeSync(fence);
    fs.closeSync(plane);
    await new Promise(resolve => child.once('exit', resolve));
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
