// End-to-end check that backend/server.js stays a read-only viewer: /ws forwards only read-only
// queries, refuses cross-site and DNS-rebinding handshakes, survives malformed or oversized frames,
// and the HTTP API refuses non-JSON POSTs and prototype ids. A fake avplumber stands in for the
// engine; no browser and no native instance. Run: node tests/check-read-only-bridge.cjs
const assert = require('node:assert/strict');
const net = require('node:net');
const crypto = require('node:crypto');
const { delay, freePort, fakeAvplumber, startBackend, postJson, openTab } = require('./backendHarness.cjs');

// A non-browser client that completes the handshake and then sends one raw frame.
function sendRawFrame(uiPort, frame) {
  return new Promise((resolve, reject) => {
    const socket = net.createConnection({ host: '127.0.0.1', port: uiPort }, () => {
      socket.write(['GET /ws HTTP/1.1', `Host: 127.0.0.1:${uiPort}`, 'Upgrade: websocket', 'Connection: Upgrade',
        `Sec-WebSocket-Key: ${crypto.randomBytes(16).toString('base64')}`, 'Sec-WebSocket-Version: 13', '', ''].join('\r\n'));
    });
    let upgraded = false;
    socket.on('data', data => {
      if (!upgraded && String(data).startsWith('HTTP/1.1 101')) {
        upgraded = true;
        socket.write(frame);
      }
    });
    socket.on('close', () => resolve(upgraded));
    socket.on('error', reject);
  });
}

(async () => {
  const avpPort = await freePort();
  const { uiPort, api, backend, exited } = await startBackend();
  const base = `http://127.0.0.1:${uiPort}`;
  const alive = async () => {
    assert.equal(exited(), false, 'backend exited');
    assert.equal((await fetch(api)).status, 200);
  };
  let avp;
  try {
    avp = await fakeAvplumber(avpPort);

    // Registration: JSON only (a no-cors form or text/plain POST needs no preflight), no prototype ids.
    const plain = await fetch(`${api}/heartbeat`, { method: 'POST', body: JSON.stringify({ id: 'mixer', port: avpPort }) });
    assert.equal(plain.status, 415);
    assert.deepEqual((await (await fetch(api)).json()).instances, []);
    assert.equal((await postJson(api, { id: '__proto__', port: avpPort })).status, 400);
    assert.equal((await postJson(`${api}/heartbeat`, { id: 'mixer', port: avpPort })).status, 200);

    // Only read-only queries reach the engine; anything else is answered with an error.
    const tab = await openTab(uiPort);
    assert.equal((await tab.command('1', 'nodes.json', 'mixer')).statusLine, '201 OK');
    assert.equal((await tab.command('2', 'node.object.get in streams', 'mixer')).statusLine, '201 OK');
    for (const [i, command] of ['node.delete in', 'queues.stats.reset', 'nodes.json\nnode.delete in',
      'nodes.json\r\ngroup.stop main', 'nodes.json extra'].entries()) {
      const reply = await tab.command(`bad-${i}`, command, 'mixer');
      assert.equal(reply.statusLine, '500 ERROR', command);
      assert.match(reply.error, /read-only/);
    }
    assert.deepEqual(avp.received, ['nodes.json', 'node.object.get in streams']);

    // Cross-site page and DNS rebinding: the handshake is refused.
    await assert.rejects(openTab(uiPort, { origin: 'http://evil.example' }), /handshake refused: 403/);
    await assert.rejects(openTab(uiPort, { headers: { Host: `rebind.example:${uiPort}` } }), /handshake refused: 403/);
    (await openTab(uiPort, { origin: 'http://127.0.0.1:8080' })).ws.close();

    // Stats under a prototype name are refused and pollute nothing; the static server keeps working.
    for (const query of ['instance=__proto__', 'instance=constructor']) {
      const res = await postJson(`${base}/api/stats?${query}`, { name: 'start' });
      assert.equal(res.status, 400, query);
    }
    assert.equal((await postJson(`${base}/api/stats?instance=mixer`, { name: '__proto__' })).status, 400);
    assert.equal((await postJson(`${base}/api/stats?instance=mixer`, { name: 'stream1' })).status, 204);
    assert.deepEqual(Object.keys(await (await fetch(`${base}/api/stats/latest`)).json()), ['mixer']);
    assert.notEqual((await fetch(`${base}/`)).status, 500);
    await alive();

    // An oversized message closes that connection only.
    const big = await openTab(uiPort);
    const closed = new Promise(resolve => big.ws.on('close', code => resolve(code)));
    big.ws.send('x'.repeat(1024 * 1024 + 1));
    assert.equal(await closed, 1009);
    // An unmasked client frame is a protocol violation.
    assert.equal(await sendRawFrame(uiPort, Buffer.from([0x81, 0x02, 0x68, 0x69])), true);
    await delay(200);
    await alive();
    assert.equal((await tab.command('3', 'queues.json', 'mixer')).statusLine, '201 OK');

    tab.ws.close();
    console.log('PASS');
  } finally {
    backend.kill();
    await avp?.kill();
  }
})().catch(error => {
  console.error(error);
  process.exit(1);
});
