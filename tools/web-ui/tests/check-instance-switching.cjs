// Node22+, npm run build first. Real Svelte browser, isolated mock backend.
// No connection to a native instance and no remote deployment. The page is served under a path
// prefix, like behind a reverse proxy, so every asset, API and WebSocket URL must be relative.
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
let registry = instances, initialReplies = true, socket, connections = 0;
const requests = [], errors = [], prefix = '/graph';
const server = http.createServer((req, res) => {
  const full = new URL(req.url, 'http://localhost').pathname;
  if (!full.startsWith(prefix + '/')) { res.writeHead(404); return res.end(); }
  const pathname = full.slice(prefix.length);
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
backend.on('connection', (client, req) => {
  if (req.url !== `${prefix}/ws`) errors.push(`WebSocket outside the prefix: ${req.url}`);
  socket = client;
  connections += 1;
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
  const url = `http://127.0.0.1:${server.address().port}${prefix}/`;
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
    const count = () => evaluate(`Number(document.querySelector('main')?.dataset.nodes)`);
    const select = id => evaluate(`(() => { const select = document.querySelector('.inst select');
      select.value = ${JSON.stringify(id)}; select.dispatchEvent(new Event('change', {bubbles:true})); })()`);
    const selectedEmpty = () => evaluate(`document.querySelector('main').dataset.selected === ''`);
    const findRequest = (from, instance, command) => requests.slice(from).find(r => r.instanceId === instance && r.command === command);
    await call('Runtime.enable');
    await until(async () => await count() === 2);
    // Select a node from the Nodes page: the row opens it in the graph.
    await evaluate(`location.hash = 'nodes'`);
    await until(async () => await evaluate(`!!document.querySelector('tr[data-node="shared"]')`));
    await evaluate(`document.querySelector('tr[data-node="shared"]').click()`);
    await until(async () => await evaluate(`document.querySelector('main').dataset.selected === 'node:shared'`));
    assert.equal(await evaluate(`location.hash`), '', 'opening a node did not return to the graph page');
    initialReplies = false;
    let from = requests.length;
    await select('b');
    await until(() => findRequest(from, 'b', 'nodes.json'));
    assert.equal(await count(), 0, 'switch retained the old graph while target unavailable');
    assert.equal(await evaluate(`Number(document.querySelector('main').dataset.queues)`), 0);
    assert(await selectedEmpty(), 'switch retained selected node');
    const bNodes = findRequest(from, 'b', 'nodes.json');
    const bQueues = findRequest(from, 'b', 'queues.json');
    reply(bNodes, nodes('old'), 'a'); // Exact pending ID, wrong instance.
    reply(bQueues, [], 'a', '500 wrong instance');
    await delay(100);
    assert.equal(await count(), 0, 'wrong-instance response populated graph');
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
    // The process behind the selected instance restarted: a new generation reloads its graph
    // without clearing the shown one first; the first generation seen is not a restart.
    const generations = gens => instances.map(instance => ({ ...instance, generation: gens[instance.id] }));
    from = requests.length;
    socket.send(JSON.stringify({ type: 'instances', instances: generations({ a: 1, b: 1 }) }));
    socket.send(JSON.stringify({ type: 'instances', instances: generations({ a: 2, b: 1 }) }));
    await delay(150);
    assert(!findRequest(from, 'b', 'nodes.json'), 'unchanged generation reloaded the graph');
    socket.send(JSON.stringify({ type: 'instances', instances: generations({ a: 2, b: 2 }) }));
    await until(() => findRequest(from, 'b', 'nodes.json'));
    assert.equal(await count(), 1, 'restart cleared the graph before the new one arrived');
    reply(findRequest(from, 'b', 'nodes.json'), nodes('old'));
    await until(async () => await count() === 2);
    // Missing deep link stays pending; registration later selects that ID.
    registry = [instances[0]];
    from = requests.length;
    // Wait for the new document and its own WebSocket, which `socket` then is.
    const connectionsBefore = connections;
    await evaluate('window.beforeNavigation = true');
    await call('Page.navigate', { url: url + '?instance=b' });
    await until(async () => connections > connectionsBefore &&
      await evaluate(`!window.beforeNavigation && !!document.querySelector('.inst select')`));
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
    // Large graphs retain every row and complete parameter values, even when the overview is compact.
    from = requests.length;
    socket.send(JSON.stringify({ type: 'instances', instances }));
    await until(async () => await evaluate(`!!document.querySelector('.inst option[value="a"]')`));
    await select('a');
    await until(() => findRequest(from, 'a', 'nodes.json'));
    const longUrl = 'https://source.example/' + 'x'.repeat(240);
    const manyNodes = Array.from({length: 450}, (_, i) => ({name: `node_${i}`, type: 'input',
      params: {group: 'sources', dst: `queue_${i}`, url: longUrl, choices: Array.from({length: 20}, (_, j) => j)}}));
    const manyQueues = manyNodes.map((n, i) => ({name: n.params.dst, capacity: 4, occupied: 1, pps: i ? 30 : 0,
      enqueued_total: 100, dequeued_total: 99, dropped_total: 0, subscription_active: i !== 1}));
    reply(findRequest(from, 'a', 'nodes.json'), manyNodes);
    reply(findRequest(from, 'a', 'queues.json'), manyQueues);
    await until(async () => await count() === 450);
    const clickPage = name => evaluate(`[...document.querySelectorAll('nav.tabs button')].find(b => b.textContent.trim() === '${name}').click()`);
    await clickPage('Nodes');
    await until(async () => await evaluate(`document.querySelectorAll('[data-node]').length`) === 400);
    await evaluate(`document.querySelector('[aria-label="nodes pagination"] button:last-child').click()`);
    await until(async () => await evaluate(`document.querySelectorAll('[data-node]').length`) === 50);
    await evaluate(`document.querySelector('[data-node="node_449"]').click()`);
    await until(async () => await evaluate(`document.querySelector('main').dataset.selected`) === 'node:node_449');
    await evaluate(`Object.defineProperty(navigator, 'clipboard', {configurable: true, value: {writeText: async t => {window.copiedParams = t;}}});
      document.querySelector('[aria-label="Copy node parameters"]').click()`);
    await until(async () => !!await evaluate('window.copiedParams'));
    const copied = JSON.parse(await evaluate('window.copiedParams'));
    assert.equal(copied.url, longUrl);
    assert.equal(copied.choices.length, 20);
    assert(await evaluate(`document.querySelector('.gv-full-params pre').textContent.includes(${JSON.stringify(longUrl)})`));
    await clickPage('Queues');
    await until(async () => await evaluate(`document.querySelectorAll('[data-queue]').length`) === 400);
    // Refresh after inspecting params so expiry cannot be mistaken for a missing subscription.
    from = requests.length;
    await evaluate(`[...document.querySelectorAll('header button')].find(b => b.textContent.trim() === 'Refresh graph').click()`);
    await until(() => findRequest(from, 'a', 'queues.json'));
    reply(findRequest(from, 'a', 'queues.json'), manyQueues);
    reply(findRequest(from, 'a', 'nodes.json'), manyNodes);
    await until(async () => await evaluate(`document.querySelector('[data-queue="queue_0"]')?.textContent.includes('subscribed')`));
    assert(await evaluate(`document.querySelector('[data-queue="queue_1"]').textContent.includes('unsubscribed')`));
    await evaluate(`document.querySelector('[aria-label="queues pagination"] button:last-child').click()`);
    await until(async () => await evaluate(`document.querySelectorAll('[data-queue]').length`) === 50);
    await evaluate(`const f = document.querySelector('[aria-label="Filter queues"]'); f.value = 'queue_449'; f.dispatchEvent(new Event('input', {bubbles:true}));`);
    await until(async () => await evaluate(`document.querySelectorAll('[data-queue]').length`) === 1);
    assert(await evaluate(`document.querySelector('[aria-label="queues pagination"]').textContent.includes('Page 1 of 1')`));
    await evaluate(`[...document.querySelectorAll('header button')].find(b => b.textContent.trim() === 'Pause queue updates').click()`);
    await until(async () => await evaluate(`document.querySelector('[data-queue="queue_449"]').textContent.includes('unavailable')`));
    const requestsAtPause = requests.filter(r => r.command === 'queues.json').length;
    await delay(1200);
    assert.equal(requests.filter(r => r.command === 'queues.json').length, requestsAtPause, 'paused queue poll still runs');
    from = requests.length;
    await evaluate(`[...document.querySelectorAll('header button')].find(b => b.textContent.trim() === 'Resume queue updates').click()`);
    await until(() => findRequest(from, 'a', 'queues.json'));
    reply(findRequest(from, 'a', 'queues.json'), manyQueues);
    await until(async () => await evaluate(`document.querySelector('[data-queue="queue_449"]').textContent.includes('subscribed')`));
    await delay(4100);
    await until(async () => await evaluate(`document.querySelector('[data-queue="queue_449"]').textContent.includes('unavailable')`));
    await clickPage('Graph');
    if (process.env.SCREENSHOT_PATH) {
      const shot = await call('Page.captureScreenshot');
      fs.writeFileSync(process.env.SCREENSHOT_PATH, Buffer.from(shot.data, 'base64'));
    }

    // A browser without WebGL: the graph shows a notice and the other panels still mount.
    await call('Page.enable');
    await call('Page.addScriptToEvaluateOnNewDocument', { source: `{ const get = HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext = function (type, ...rest) { return /webgl/.test(type) ? null : get.call(this, type, ...rest); }; }` });
    await evaluate('window.beforeNavigation = true');
    await call('Page.navigate', { url });
    await until(async () => await evaluate(`!window.beforeNavigation && !!document.querySelector('.inst select')`));
    await delay(150);
    assert(await evaluate(`!!document.querySelector('.gv-nogl')`), 'no notice in place of the graph without WebGL');
    assert(await evaluate(`document.querySelectorAll('.lm_stack').length > 1`), 'the graph without WebGL took the other panels down');
    assert.deepEqual(errors, []);
    console.log('PASS: switching, late replies, pagination, full params, subscriptions, paused/stale telemetry and no WebGL');
  } finally {
    if (ws) ws.close();
    if (page) await fetch(`${cdp}/json/close/${page.id}`);
    for (const client of backend.clients) client.terminate();
    backend.close(); server.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
