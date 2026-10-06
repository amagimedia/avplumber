// End-to-end check of restart detection in backend/server.js: a real backend on a free port, a fake
// avplumber control server, and a WebSocket client standing in for an open graph tab.
// No browser and no connection to a native instance. Run: node tests/check-instance-restart.cjs
const assert = require('node:assert/strict');
const net = require('node:net');
const path = require('node:path');
const { spawn } = require('node:child_process');
const WebSocket = require('../node_modules/ws');
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const INTERNAL_FIELDS = ['lastHeartbeat', 'usesHeartbeat', 'explicit', 'connectionLost'];

function freePort() {
  return new Promise((resolve, reject) => {
    const probe = net.createServer().once('error', reject);
    probe.listen(0, '127.0.0.1', () => {
      const { port } = probe.address();
      probe.close(() => resolve(port));
    });
  });
}

// Answers every command with an empty JSON body, like an idle engine.
function fakeAvplumber(port) {
  const sockets = new Set();
  const server = net.createServer(socket => {
    sockets.add(socket);
    socket.on('close', () => sockets.delete(socket));
    socket.write('100 VTR READY\n');
    socket.on('data', data => {
      for (const line of String(data).split('\n').filter(Boolean)) {
        socket.write(line === 'bye' ? 'BYE\n' : '201 OK\n[]\n\n');
      }
    });
  });
  const kill = () => {
    for (const socket of sockets) socket.destroy();
    return new Promise(resolve => server.close(resolve));
  };
  return new Promise(resolve => server.listen(port, '127.0.0.1', () => resolve({ kill })));
}

function openTab(uiPort) {
  const ws = new WebSocket(`ws://127.0.0.1:${uiPort}/ws`);
  const broadcasts = [];
  const replies = new Map();
  ws.on('message', data => {
    const message = JSON.parse(data);
    if (message.type === 'instances') broadcasts.push(message.instances);
    if (message.type === 'response') replies.get(message.id)?.(message);
  });
  const command = (id, command, instanceId) => new Promise(resolve => {
    replies.set(id, resolve);
    ws.send(JSON.stringify({ type: 'command', id, command, instanceId }));
  });
  return new Promise(resolve => ws.on('open', () => resolve({ ws, broadcasts, command })));
}

function assertPublic(instance) {
  for (const field of INTERNAL_FIELDS) assert.equal(field in instance, false, `${field} leaked`);
}

(async () => {
  const [uiPort, avpPort] = [await freePort(), await freePort()];
  const env = { ...process.env, WEBUI_PORT: String(uiPort) };
  delete env.AVPLUMBER_PORT; // no env-configured default instance
  const backend = spawn(process.execPath, [path.join(__dirname, '../backend/server.js')],
    { env, stdio: ['ignore', 'ignore', 'inherit'] });
  const api = `http://127.0.0.1:${uiPort}/api/instances`;
  const heartbeat = async () => {
    const res = await fetch(`${api}/heartbeat`, { method: 'POST', body: JSON.stringify({ id: 'mixer', port: avpPort }) });
    assert.equal(res.status, 200);
    return res.json();
  };
  let avp;
  try {
    for (let i = 0; ; i++) {
      try { await fetch(api); break; } catch (e) { if (i > 50) throw e; await delay(100); }
    }
    avp = await fakeAvplumber(avpPort);
    assertPublic(await heartbeat());

    const tab = await openTab(uiPort);
    const { instances: [inst] } = await (await fetch(api)).json();
    assert.equal(inst.id, 'mixer');
    assertPublic(inst);
    assert.equal((await tab.command('1', 'nodes.json', 'mixer')).statusLine, '201 OK');

    await heartbeat(); await delay(200);
    assert.equal(tab.broadcasts.length, 0, 'a plain heartbeat must not broadcast');

    await avp.kill(); await delay(200);
    avp = await fakeAvplumber(avpPort);
    assertPublic(await heartbeat()); await delay(200);
    assert.equal(tab.broadcasts.length, 1, 'a restart must broadcast once');
    const [restarted] = tab.broadcasts[0];
    assert.equal(restarted.id, 'mixer');
    assert.notEqual(restarted.generation, inst.generation);
    assertPublic(restarted);

    await heartbeat(); await delay(200);
    assert.equal(tab.broadcasts.length, 1, 'the heartbeat after a restart must not broadcast');

    // Another tab opens a control connection and closes normally: not a restart.
    const other = await openTab(uiPort);
    assert.equal((await other.command('2', 'nodes.json', 'mixer')).statusLine, '201 OK');
    other.ws.close(); await delay(200);
    await heartbeat(); await delay(200);
    assert.equal(tab.broadcasts.length, 1, 'a tab closing its connection must not broadcast');

    // Only a refused connection marks the loss: the open tab closes normally, and a new tab
    // selects the instance while its process is down.
    tab.ws.close(); await delay(200);
    await avp.kill(); await delay(200);
    const late = await openTab(uiPort);
    assert.equal((await late.command('3', 'nodes.json', 'mixer')).statusLine, '500 ERROR');
    avp = await fakeAvplumber(avpPort);
    await heartbeat(); await delay(200);
    assert.equal(late.broadcasts.length, 1, 'a heartbeat after a refused connection must broadcast once');
    assert.notEqual(late.broadcasts[0][0].generation, restarted.generation);
    await heartbeat(); await delay(200);
    assert.equal(late.broadcasts.length, 1, 'the next heartbeat must not broadcast');

    late.ws.close();
    console.log('PASS');
  } finally {
    backend.kill();
    await avp?.kill();
  }
})().catch(error => {
  console.error(error);
  process.exit(1);
});
