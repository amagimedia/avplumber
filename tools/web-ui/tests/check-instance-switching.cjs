// Node22+, npm run build first. Real Svelte browser, isolated mock backend.
// No connection to a native instance and no remote deployment.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const { WebSocketServer } = require('../node_modules/ws');
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const root = path.resolve(__dirname, '../frontend/dist');
const instances = ['a', 'b'].map(id => ({ id, name: id, host: '127.0.0.1', port: 1 }));
const nodes = source => [{ name: 'shared', type: 'input', params: { source } },
  ...(source === 'old' ? [{ name: 'old_extra', type: 'null', params: {} }] : [])];
const queue = source => [{ name: `${source}_queue`, type: 'video', occupied: 1, capacity: 3 }];
let registry = instances, initialReplies = true, socket;
const requests = [], errors = [];
const server = http.createServer((req, res) => {
  const pathname = new URL(req.url, 'http://localhost').pathname;
  if (pathname === '/api/instances') {
    res.setHeader('Content-Type', 'application/json');
    return res.end(JSON.stringify({ instances: registry }));
  }
  const file = path.join(root, pathname === '/' ? 'index.html' : pathname);
  if (!file.startsWith(root + '/') || !fs.existsSync(file)) { res.writeHead(404); return res.end(); }
  res.setHeader('Content-Type', file.endsWith('.js') ? 'text/javascript' :
    file.endsWith('.css') ? 'text/css' : 'text/html');
  res.end(fs.readFileSync(file));
});
const backend = new WebSocketServer({ server });
const reply = (request, body, instanceId = request.instanceId, statusLine = '201 OK') =>
  socket.send(JSON.stringify({ type: 'response', id: request.id, instanceId,
    statusLine, body: JSON.stringify(body) }));
backend.on('connection', client => {
  socket = client;
  client.on('message', data => {
    const request = JSON.parse(data);
    requests.push(request);
    if (initialReplies && request.instanceId === 'a') {
      reply(request, request.command === 'nodes.json' ? nodes('old') :
        request.command === 'queues.json' ? queue('old') : []);
    }
  });
});

(async () => {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const url = `http://127.0.0.1:${server.address().port}/`;
  const cdp = process.env.CDP_URL || 'http://127.0.0.1:9222';
  let page, ws;
  try {
    page = await fetch(`${cdp}/json/new?${encodeURIComponent(url)}`, { method: 'PUT' }).then(r => r.json());
    ws = new WebSocket(page.webSocketDebuggerUrl);
    await new Promise(resolve => ws.addEventListener('open', resolve, { once: true }));
    const pending = new Map(); let sequence = 0;
    ws.addEventListener('message', event => {
      const message = JSON.parse(event.data), request = pending.get(message.id);
      if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails);
      if (request) {
        pending.delete(message.id); clearTimeout(request.timer);
        message.error ? request.reject(message.error) : request.resolve(message.result);
      }
    });
    const call = (method, params = {}) => new Promise((resolve, reject) => {
      const id = ++sequence;
      pending.set(id, { resolve, reject, timer: setTimeout(() => {
        pending.delete(id); reject(new Error(`${method} timeout`));
      }, 10000) });
      ws.send(JSON.stringify({ id, method, params }));
    });
    const evaluate = async expression => {
      const result = await call('Runtime.evaluate', { expression, returnByValue: true });
      if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
      return result.result.value;
    };
    const until = async predicate => {
      for (let i = 0; i < 100; ++i) { if (await predicate()) return; await delay(50); }
      throw new Error(`condition timed out: ${predicate}`);
    };
    const count = () => evaluate(`Number(document.querySelector('.graph-native-counts')?.dataset.nodes)`);
    const select = id => evaluate(`(() => { const select = document.querySelector('.instance-select select');
      select.value = ${JSON.stringify(id)}; select.dispatchEvent(new Event('change', {bubbles:true})); })()`);
    const selectedEmpty = () => evaluate(`document.querySelector('.selected-node-panel').innerText.includes('Click a node')`);
    const findRequest = (from, instance, command) => requests.slice(from).find(r => r.instanceId === instance && r.command === command);
    await call('Runtime.enable');
    await until(async () => await count() === 2);
    await until(async () => await evaluate(`[...document.querySelectorAll('[data-testid=node]')].some(n => n.innerText.startsWith('shared'))`));
    await evaluate(`[...document.querySelectorAll('[data-testid=node]')].find(n => n.innerText.startsWith('shared')).click()`);
    await until(async () => !await selectedEmpty());
    initialReplies = false;
    let from = requests.length;
    await select('b');
    await until(() => findRequest(from, 'b', 'nodes.json'));
    assert.equal(await count(), 0, 'switch retained the old graph while target unavailable');
    assert.equal(await evaluate(`Number(document.querySelector('.graph-native-counts').dataset.queues)`), 0);
    assert(await selectedEmpty(), 'switch retained selected node');
    const bNodes = findRequest(from, 'b', 'nodes.json');
    const bQueues = findRequest(from, 'b', 'queues.json');
    reply(bNodes, nodes('old'), 'a'); // Exact pending ID, wrong instance.
    reply(bQueues, [], 'a', '500 wrong instance');
    await delay(100);
    assert.equal(await count(), 0, 'wrong-instance response populated graph');
    assert(!requests.slice(from).some(r => r.command === 'queues.stats'), 'wrong response resolved new pending request');
    reply(bNodes, nodes('new'));
    reply(bQueues, queue('new'));
    await until(async () => await count() === 1);
    assert(await selectedEmpty(), 'same node name selected itself in the new instance');
    assert.equal(await evaluate(`new URL(location.href).searchParams.get('instance')`), 'b');
    // A delayed response must stay invalid even after revisiting its instance.
    from = requests.length;
    await select('a');
    await until(() => findRequest(from, 'a', 'nodes.json'));
    await select('b');
    await until(() => findRequest(from, 'b', 'nodes.json'));
    const newB = findRequest(from, 'b', 'nodes.json');
    assert.notEqual(newB.id, bNodes.id, 'wire IDs reused across selection generations');
    reply(bNodes, nodes('old'));
    await delay(100);
    assert.equal(await count(), 0, 'old response revived a previous selection generation');
    reply(newB, nodes('new'));
    await until(async () => await count() === 1);
    // Missing deep link stays pending; registration later selects that ID.
    registry = [instances[0]];
    from = requests.length;
    await call('Page.navigate', { url: url + '?instance=b' });
    await until(async () => await evaluate(`!!document.querySelector('.instance-select select')`));
    await delay(150);
    assert.equal(await count(), 0, 'missing deep link silently selected another demo');
    assert.equal(await evaluate(`new URL(location.href).searchParams.get('instance')`), 'b');
    assert(!requests.slice(from).some(r => r.instanceId === 'a'), 'sent commands to fallback instance');
    socket.send(JSON.stringify({ type: 'instances', instances }));
    await until(() => findRequest(from, 'b', 'nodes.json'));
    reply(findRequest(from, 'b', 'nodes.json'), nodes('new'));
    await until(async () => await count() === 1);
    socket.send(JSON.stringify({ type: 'instances', instances: [] }));
    await until(async () => await count() === 0);
    assert(await selectedEmpty());
    assert.deepEqual(errors, []);
    console.log('PASS: switching, pending ownership, late replies, removal and deep links');
  } finally {
    if (ws) ws.close();
    if (page) await fetch(`${cdp}/json/close/${page.id}`);
    for (const client of backend.clients) client.terminate();
    backend.close(); server.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
