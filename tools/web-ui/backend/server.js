// Minimal Node.js backend for avplumber web UI.
// Responsibilities:
// - Serve static frontend files from ./public
// - Expose WebSocket endpoint (/ws) that proxies commands to avplumber TCP control server
// - Accept HTTP POST /api/stats from avplumber's stats.subscribe and fan out to WebSocket clients
// - Optionally tail avplumber logfile (LOG_FILE env) and stream lines to clients

/* eslint-disable no-console */

const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');
const net = require('net');
const WebSocket = require('ws');

const HTTP_PORT = parseInt(process.env.WEBUI_PORT || process.env.PORT || '22222', 10);
const AVP_HOST = process.env.AVPLUMBER_HOST || 'localhost';
const AVP_PORT = parseInt(process.env.AVPLUMBER_PORT || '0', 10);
const LOG_FILE = process.env.AVPLUMBER_LOGFILE || process.env.LOG_FILE || '';

const DIST_DIR = path.join(__dirname, '..', 'frontend', 'dist');
const PUBLIC_DIR = path.join(__dirname, '..', 'frontend', 'public');

// In-memory instance registry: { id: { id, name, host, port, logFile, lastHeartbeat, usesHeartbeat } }
const instances = new Map();
const HEARTBEAT_TIMEOUT_MS = 30000; // 30 seconds
const HEARTBEAT_CLEANUP_INTERVAL_MS = 5000; // Check every 5 seconds

function addInstance(def, usesHeartbeat = false) {
  if (!def) throw new Error('instance definition required');
  const host = def.host || 'localhost';
  const port = parseInt(def.port, 10);
  if (!port || Number.isNaN(port)) {
    throw new Error('valid port required');
  }
  let id = def.id || `${host}:${port}`;
  if (instances.has(id)) {
    // Make it unique
    id = `${id}-${Date.now()}`;
  }
  const name = def.name || `${host}:${port}`;
  const logFile = def.logFile || def.log_file || def.log || null;
  const inst = { 
    id, 
    name, 
    host, 
    port, 
    logFile, 
    lastHeartbeat: usesHeartbeat ? Date.now() : null,
    usesHeartbeat 
  };
  instances.set(id, inst);
  return inst;
}

function updateInstanceHeartbeat(def) {
  if (!def) throw new Error('instance definition required');
  const host = def.host || 'localhost';
  const port = parseInt(def.port, 10);
  if (!port || Number.isNaN(port)) {
    throw new Error('valid port required');
  }
  // Try to find existing instance by host:port
  let foundId = null;
  for (const [id, inst] of instances.entries()) {
    if (inst.host === host && inst.port === port) {
      foundId = id;
      break;
    }
  }
  
  if (foundId) {
    // Update existing instance
    const inst = instances.get(foundId);
    inst.lastHeartbeat = Date.now();
    inst.usesHeartbeat = true;
    // Update name and logFile if provided
    if (def.name) {
      inst.name = def.name;
    }
    if (def.logFile || def.log_file || def.log) {
      inst.logFile = def.logFile || def.log_file || def.log;
    }
    return { inst, wasNew: false };
  } else {
    // Create new instance with heartbeat tracking
    const inst = addInstance(def, true);
    return { inst, wasNew: true };
  }
}

// Cleanup stale instances periodically
setInterval(() => {
  const now = Date.now();
  const toRemove = [];
  for (const [id, inst] of instances.entries()) {
    // Only remove instances that use heartbeat and have timed out
    if (inst.usesHeartbeat && inst.lastHeartbeat && (now - inst.lastHeartbeat) > HEARTBEAT_TIMEOUT_MS) {
      toRemove.push(id);
    }
  }
  for (const id of toRemove) {
    instances.delete(id);
    // Stop log tailing for removed instance
    if (logTails.has(id)) {
      fs.unwatchFile(logTails.get(id).logFile);
      logTails.delete(id);
    }
    console.log(`Removed stale instance: ${id}`);
  }
  if (toRemove.length > 0) {
    broadcastInstancesUpdate();
  }
}, HEARTBEAT_CLEANUP_INTERVAL_MS);

// Initialize default instance from env for backwards compatibility
// Note: This instance won't be automatically removed since it doesn't send heartbeats
// It's kept for backwards compatibility with manual registration
if (AVP_PORT) {
  addInstance({
    id: 'default',
    name: 'default',
    host: AVP_HOST,
    port: AVP_PORT,
    logFile: LOG_FILE || null
  });
}

// Stats history: { [instanceId: string]: { [streamName: string]: { ts: number, data: any }[] } }
const statsHistory = {};
const STATS_HISTORY_LIMIT = 512;

function addStatsSample(obj, instanceId) {
  const instKey = instanceId || 'default';
  const streamKey = obj && typeof obj === 'object' && obj.name ? String(obj.name) : 'default';
  if (!statsHistory[instKey]) {
    statsHistory[instKey] = {};
  }
  if (!statsHistory[instKey][streamKey]) {
    statsHistory[instKey][streamKey] = [];
  }
  const arr = statsHistory[instKey][streamKey];
  arr.push({ ts: Date.now(), data: obj });
  if (arr.length > STATS_HISTORY_LIMIT) {
    arr.splice(0, arr.length - STATS_HISTORY_LIMIT);
  }
}

function latestStatsSnapshot() {
  const out = {};
  for (const [instKey, streams] of Object.entries(statsHistory)) {
    out[instKey] = {};
    for (const [name, arr] of Object.entries(streams)) {
      if (arr.length > 0) {
        out[instKey][name] = arr[arr.length - 1];
      }
    }
  }
  return out;
}

// Simple static file server
function serveStatic(req, res) {
  let pathname = url.parse(req.url).pathname || '/';
  if (pathname === '/') {
    pathname = '/index.html';
  }

  // Prefer built assets from dist/, fall back to public/
  const distPath = path.normalize(path.join(DIST_DIR, pathname));
  const publicPath = path.normalize(path.join(PUBLIC_DIR, pathname));

  const rootOk =
    (distPath.startsWith(DIST_DIR) || publicPath.startsWith(PUBLIC_DIR));
  if (!rootOk) {
    res.writeHead(403);
    res.end('Forbidden');
    return;
  }

  const tryServe = (p, onMissing) => {
    fs.stat(p, (err, stat) => {
      if (err || !stat.isFile()) {
        onMissing();
        return;
      }
      const ext = path.extname(p).toLowerCase();
      const type =
        ext === '.html' ? 'text/html; charset=utf-8' :
        ext === '.js' ? 'text/javascript; charset=utf-8' :
        ext === '.css' ? 'text/css; charset=utf-8' :
        'application/octet-stream';

      res.writeHead(200, { 'Content-Type': type });
      const stream = fs.createReadStream(p);
      stream.on('error', () => {
        res.writeHead(500);
        res.end('Read error');
      });
      stream.pipe(res);
    });
  };

  tryServe(distPath, () => {
    tryServe(publicPath, () => {
      res.writeHead(404);
      res.end('Not found');
    });
  });
}
const server = http.createServer((req, res) => {
  const parsed = url.parse(req.url, true);

  if (req.method === 'GET' && parsed.pathname === '/api/instances') {
    res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
    // Return instances without internal fields for API compatibility
    const instancesArray = Array.from(instances.values()).map(inst => {
      const { lastHeartbeat, usesHeartbeat, ...rest } = inst;
      return rest;
    });
    res.end(
      JSON.stringify({
        instances: instancesArray
      })
    );
    return;
  }

  if (req.method === 'POST' && parsed.pathname === '/api/instances') {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
      if (body.length > 1024 * 1024) {
        req.destroy();
      }
    });
    req.on('end', () => {
      try {
        const obj = JSON.parse(body || '{}');
        const inst = addInstance(obj);
        // start per-instance log tailing if logfile provided
        startLogTailForInstance(inst);
        broadcastInstancesUpdate();
        res.writeHead(201, { 'Content-Type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify(inst));
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ error: String(e && e.message ? e.message : e) }));
      }
    });
    return;
  }

  if (req.method === 'POST' && parsed.pathname === '/api/instances/heartbeat') {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
      if (body.length > 1024 * 1024) {
        req.destroy();
      }
    });
    req.on('end', () => {
      try {
        const obj = JSON.parse(body || '{}');
        const { inst, wasNew } = updateInstanceHeartbeat(obj);
        // start per-instance log tailing if logfile provided and not already started
        startLogTailForInstance(inst);
        // Only broadcast if this was a new instance (first heartbeat)
        if (wasNew) {
          broadcastInstancesUpdate();
        }
        res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify(inst));
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ error: String(e && e.message ? e.message : e) }));
      }
    });
    return;
  }

  if (req.method === 'POST' && parsed.pathname === '/api/stats') {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
      if (body.length > 10 * 1024 * 1024) {
        req.destroy();
      }
    });
    req.on('end', () => {
      try {
        const obj = JSON.parse(body);
        const instanceId = parsed.query && parsed.query.instance ? String(parsed.query.instance) : null;
        addStatsSample(obj, instanceId);
        // Broadcast to all clients
        const msg = JSON.stringify({
          type: 'stats',
          instanceId: instanceId || null,
          key: obj.name || 'default',
          payload: obj
        });
        for (const client of wss.clients) {
          if (client.readyState === WebSocket.OPEN) {
            client.send(msg);
          }
        }
      } catch (e) {
        console.error('Failed to parse stats JSON:', e);
      }
      res.writeHead(204);
      res.end();
    });
    return;
  }

  if (req.method === 'GET' && parsed.pathname === '/api/stats/latest') {
    res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify(latestStatsSnapshot()));
    return;
  }

  serveStatic(req, res);
});

// Lightweight parser for avplumber control protocol, one outstanding command at a time
class AvpConnection {
  constructor(host, port) {
    this.host = host;
    this.port = port;
    this.socket = null;
    this.buffer = '';
    this.handshakeDone = false;
    this.current = null; // { id, command, resolve, reject, state, bodyLines }
    this.queue = [];
    this.closed = false;

    this._connect();
  }

  _connect() {
    this.socket = net.createConnection({ host: this.host, port: this.port }, () => {
      // Connected, wait for 100 line
    });

    this.socket.setEncoding('utf8');

    this.socket.on('data', (chunk) => this._onData(chunk));
    this.socket.on('error', (err) => {
      console.error('TCP error:', err);
      if (this.current && this.current.reject) {
        this.current.reject(err);
        this.current = null;
      }
    });
    this.socket.on('close', () => {
      this.closed = true;
      if (this.current && this.current.reject) {
        this.current.reject(new Error('Connection closed'));
        this.current = null;
      }
    });
  }

  _onData(chunk) {
    this.buffer += chunk;
    while (true) {
      const idx = this.buffer.indexOf('\n');
      if (idx === -1) break;
      const line = this.buffer.slice(0, idx).replace(/\r$/, '');
      this.buffer = this.buffer.slice(idx + 1);
      this._handleLine(line);
    }
  }

  _handleLine(line) {
    if (!this.handshakeDone) {
      // Expecting "100 VTR READY"
      this.handshakeDone = true;
      return;
    }
    if (!this.current) {
      // Unexpected line; ignore
      return;
    }
    if (this.current.state === 'status') {
      const trimmed = line.trim();
      if (trimmed.startsWith('201 ')) {
        this.current.statusLine = trimmed;
        this.current.state = 'body';
        this.current.bodyLines = [];
      } else if (
        trimmed.startsWith('200 ') ||
        trimmed.startsWith('400 ') ||
        trimmed.startsWith('500 ') ||
        trimmed === 'BYE'
      ) {
        const cur = this.current;
        this.current = null;
        if (cur.resolve) {
          cur.resolve({ statusLine: trimmed, body: '' });
        }
        this._flushQueue();
      } else {
        // Fallback: treat as body-less status
        const cur = this.current;
        this.current = null;
        if (cur.resolve) {
          cur.resolve({ statusLine: trimmed, body: '' });
        }
        this._flushQueue();
      }
    } else if (this.current.state === 'body') {
      // Blank line terminates body
      if (line === '') {
        const cur = this.current;
        this.current = null;
        if (cur.resolve) {
          cur.resolve({
            statusLine: cur.statusLine || '201 OK',
            body: cur.bodyLines.join('\n')
          });
        }
        this._flushQueue();
      } else {
        this.current.bodyLines.push(line);
      }
    }
  }

  _flushQueue() {
    if (this.current || this.closed) return;
    const next = this.queue.shift();
    if (!next) return;
    this.current = {
      id: next.id,
      command: next.command,
      resolve: next.resolve,
      reject: next.reject,
      state: 'status',
      bodyLines: [],
      statusLine: ''
    };
    this.socket.write(next.command.trimEnd() + '\n');
  }

  sendCommand(id, command) {
    if (this.closed) {
      return Promise.reject(new Error('Connection closed'));
    }
    return new Promise((resolve, reject) => {
      this.queue.push({ id, command, resolve, reject });
      this._flushQueue();
    });
  }

  close() {
    try {
      this.socket.end('bye\n');
    } catch (_) {
      // ignore
    }
  }
}

const wss = new WebSocket.Server({ server, path: '/ws' });

// Broadcast instance list updates to all WebSocket clients
function broadcastInstancesUpdate() {
  const instancesArray = Array.from(instances.values()).map(inst => {
    const { lastHeartbeat, usesHeartbeat, ...rest } = inst;
    return rest;
  });
  const msg = JSON.stringify({
    type: 'instances',
    instances: instancesArray
  });
  for (const client of wss.clients) {
    if (client.readyState === WebSocket.OPEN) {
      client.send(msg);
    }
  }
}

// Per-instance log tailing: { instanceId -> { logFile, lastSize } }
const logTails = new Map();

function startLogTailForInstance(inst) {
  if (!inst || !inst.logFile) return;
  if (logTails.has(inst.id)) return;

  const logFile = inst.logFile;
  let lastSize = 0;

  fs.stat(logFile, (err, stat) => {
    if (!err && stat && typeof stat.size === 'number') {
      lastSize = stat.size;
    }
  });

  fs.watchFile(logFile, { interval: 1000 }, (prev, curr) => {
    if (curr.size <= lastSize) {
      lastSize = curr.size;
      return;
    }
    const start = lastSize;
    const end = curr.size;
    lastSize = end;
    const stream = fs.createReadStream(logFile, { start, end: end - 1, encoding: 'utf8' });
    let buf = '';
    stream.on('data', (chunk) => {
      buf += chunk;
      let idx;
      while ((idx = buf.indexOf('\n')) !== -1) {
        const line = buf.slice(0, idx).replace(/\r$/, '');
        buf = buf.slice(idx + 1);
        const msg = JSON.stringify({ type: 'log', instanceId: inst.id, line });
        for (const client of wss.clients) {
          if (client.readyState === WebSocket.OPEN) {
            client.send(msg);
          }
        }
      }
    });
  });

  logTails.set(inst.id, { logFile, lastSize });
}

// Start log tailing for any instances that already exist
for (const inst of instances.values()) {
  startLogTailForInstance(inst);
}

wss.on('connection', (ws) => {
  const connByInstance = new Map();

  function getOrCreateConn(instanceId) {
    const inst = instances.get(instanceId);
    if (!inst) {
      throw new Error(`Unknown instance: ${instanceId}`);
    }
    let conn = connByInstance.get(instanceId);
    if (!conn || conn.closed) {
      conn = new AvpConnection(inst.host, inst.port);
      connByInstance.set(instanceId, conn);
    }
    return conn;
  }

  ws.on('message', async (data) => {
    let msg;
    try {
      msg = JSON.parse(data.toString('utf8'));
    } catch {
      return;
    }
    if (!msg || typeof msg !== 'object') return;

    if (msg.type === 'command' && typeof msg.command === 'string') {
      const id = msg.id || null;
      const instanceId = msg.instanceId || 'default';
      try {
        const conn = getOrCreateConn(instanceId);
        const result = await conn.sendCommand(id, msg.command);
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(
            JSON.stringify({
              type: 'response',
              id,
              instanceId,
              statusLine: result.statusLine,
              body: result.body
            })
          );
        }
      } catch (err) {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(
            JSON.stringify({
              type: 'response',
              id,
              instanceId,
              statusLine: '500 ERROR',
              error: String(err && err.message ? err.message : err)
            })
          );
        }
      }
    } else if (msg.type === 'ping') {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'pong', ts: Date.now() }));
      }
    }
  });

  ws.on('close', () => {
    for (const conn of connByInstance.values()) {
      conn.close();
    }
  });
});


server.listen(HTTP_PORT, () => {
  console.log(`avplumber web-ui listening on http://localhost:${HTTP_PORT}`);
  if (instances.size > 0) {
    console.log(
      'Initial instances:',
      Array.from(instances.values()).map((i) => `${i.id}=${i.host}:${i.port}`).join(', ')
    );
  } else {
    console.log('No avplumber instances configured yet. Use POST /api/instances to add some.');
  }
});


