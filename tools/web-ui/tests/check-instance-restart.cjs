// End-to-end check of restart detection in backend/server.js: a real backend on a free port, a fake
// avplumber control server, and a WebSocket client standing in for an open graph tab.
// No browser and no connection to a native instance. Run: node tests/check-instance-restart.cjs
const assert = require('node:assert/strict');
const { delay, freePort, fakeAvplumber, startBackend, postJson, openTab } = require('./backendHarness.cjs');
const INTERNAL_FIELDS = ['lastHeartbeat', 'usesHeartbeat', 'explicit', 'connectionLost'];

function assertPublic(instance) {
  for (const field of INTERNAL_FIELDS) assert.equal(field in instance, false, `${field} leaked`);
}

(async () => {
  const avpPort = await freePort();
  const { uiPort, api, backend } = await startBackend();
  const heartbeat = async () => {
    const res = await postJson(`${api}/heartbeat`, { id: 'mixer', port: avpPort });
    assert.equal(res.status, 200);
    return res.json();
  };
  let avp;
  try {
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
