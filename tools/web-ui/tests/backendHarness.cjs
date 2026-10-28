// Shared by the end-to-end backend checks: a real backend/server.js on a free port, a fake
// avplumber control server and WebSocket clients standing in for open tabs.
const net = require('node:net');
const path = require('node:path');
const { spawn } = require('node:child_process');
const WebSocket = require('../node_modules/ws');
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

function freePort() {
  return new Promise((resolve, reject) => {
    const probe = net.createServer().once('error', reject);
    probe.listen(0, '127.0.0.1', () => {
      const { port } = probe.address();
      probe.close(() => resolve(port));
    });
  });
}

// Answers every command with an empty JSON body, like an idle engine; `received` lists the lines.
function fakeAvplumber(port) {
  const sockets = new Set();
  const received = [];
  const server = net.createServer(socket => {
    sockets.add(socket);
    socket.on('close', () => sockets.delete(socket));
    socket.write('100 VTR READY\n');
    socket.on('data', data => {
      for (const line of String(data).split('\n').filter(Boolean)) {
        received.push(line);
        socket.write(line === 'bye' ? 'BYE\n' : '201 OK\n[]\n\n');
      }
    });
  });
  const kill = () => {
    for (const socket of sockets) socket.destroy();
    return new Promise(resolve => server.close(resolve));
  };
  return new Promise(resolve => server.listen(port, '127.0.0.1', () => resolve({ kill, received })));
}

// Starts the backend with `extraEnv` and waits until it answers; `api` is its /api/instances URL.
async function startBackend(extraEnv = {}) {
  const uiPort = await freePort();
  const env = { ...process.env, WEBUI_PORT: String(uiPort), ...extraEnv };
  delete env.AVPLUMBER_PORT; // no env-configured default instance
  const backend = spawn(process.execPath, [path.join(__dirname, '../backend/server.js')],
    { env, stdio: ['ignore', 'ignore', 'inherit'] });
  let exited = false;
  backend.on('exit', () => { exited = true; });
  const api = `http://127.0.0.1:${uiPort}/api/instances`;
  for (let i = 0; ; i++) {
    try { await fetch(api); break; } catch (e) { if (i > 50) throw e; await delay(100); }
  }
  return { uiPort, api, backend, exited: () => exited };
}

const postJson = (url, body) => fetch(url, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
});

function openTab(uiPort, options) {
  const ws = new WebSocket(`ws://127.0.0.1:${uiPort}/ws`, options);
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
  return new Promise((resolve, reject) => {
    ws.on('open', () => resolve({ ws, broadcasts, command }));
    ws.on('unexpected-response', (req, res) => reject(new Error(`handshake refused: ${res.statusCode}`)));
    ws.on('error', reject);
  });
}

module.exports = { delay, freePort, fakeAvplumber, startBackend, postJson, openTab };
