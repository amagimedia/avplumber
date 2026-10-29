// A stand-in for an avplumber control port: the backend checks use it as an idle engine, and with a dump of a real
// engine's replies it replays that graph so the UI can be viewed and tested without one.
//
// Protocol, as backend/server.js reads it: "100 VTR READY" on connect; then per command line "201 OK", the body and
// a blank line for queries (*.json, queues.stats, node.object.get, mixer.status), "BYE" for bye and "200 OK" for
// anything else. Without a dump every query answers [] ({} for node.object.get and mixer.status).
//
// A dump is {"nodes.json": {"status", "body"}, "queues.json": {...}, "queues.stats": {...}}. Its queues.json counters
// advance with time by each queue's rate (enqueued and dequeued, dropped_total stays), so the UI sees a live engine;
// queues named with --drop also lose a tenth of their frames, to show the dropped state.
//
//   node tests/fakeEngine.cjs <dump.json> [--port 5001] [--drop queue1,queue2]
//   AVPLUMBER_PORT=5001 npm start
const net = require('node:net');
const fs = require('node:fs');

const QUERY = /^(\S+\.json|queues\.stats|node\.object\.get|mixer\.status)(\s|$)/;
const DROP_SHARE = 0.1;

// The replayed replies by command. queues.json is rendered when asked, its counters advanced by each queue's rate
// since the start, so every poll sees new frames.
function replay(dump, drop) {
  const bodies = new Map(Object.entries(dump).map(([cmd, reply]) => [cmd, String(reply.body ?? '')]));
  const queues = bodies.has('queues.json') ? JSON.parse(bodies.get('queues.json')) : null;
  if (!queues) return bodies;
  const start = queues.map(q => ({ enq: q.enqueued_total, deq: q.dequeued_total, drop: q.dropped_total, ts: q.last_ts_seconds }));
  const began = Date.now();
  bodies.set('queues.json', () => {
    const s = (Date.now() - began) / 1000;
    queues.forEach((q, i) => {
      const b = start[i], frames = Math.round((q.enq_pps ?? q.pps ?? 0) * s);
      q.enqueued_total = b.enq + frames;
      q.dequeued_total = b.deq + Math.round((q.deq_pps ?? q.pps ?? 0) * s);
      if (drop.has(q.name)) q.dropped_total = b.drop + Math.round(frames * DROP_SHARE);
      if (typeof b.ts === 'number') q.last_ts_seconds = b.ts + s;
    });
    return JSON.stringify(queues);
  });
  return bodies;
}

function reply(bodies, line) {
  if (line === 'bye') return 'BYE\n';
  const m = QUERY.exec(line);
  if (!m) return '200 OK\n';
  let body = bodies.get(line) ?? bodies.get(m[1]) ?? (/\.json$|^queues\.stats$/.test(m[1]) ? '[]' : '{}');
  if (typeof body === 'function') body = body();
  // A blank line ends the body, so it must not contain one.
  return `201 OK\n${body.replace(/\n+/g, '\n')}\n\n`;
}

// Resolves to {port, received, kill}; `received` lists every command line, `kill` closes the port and its clients.
function startFakeEngine({ port = 0, host = '127.0.0.1', dump = null, drop = [] } = {}) {
  const bodies = dump ? replay(dump, new Set(drop)) : new Map();
  const sockets = new Set(), received = [];
  const server = net.createServer(socket => {
    sockets.add(socket);
    socket.on('close', () => sockets.delete(socket));
    socket.on('error', () => {});
    socket.setEncoding('utf8');
    socket.write('100 VTR READY\n');
    let buffer = '';
    socket.on('data', chunk => {
      buffer += chunk;
      for (let i; (i = buffer.indexOf('\n')) >= 0;) {
        const line = buffer.slice(0, i).trim();
        buffer = buffer.slice(i + 1);
        if (!line) continue;
        received.push(line);
        socket.write(reply(bodies, line));
      }
    });
  });
  const kill = () => {
    for (const socket of sockets) socket.destroy();
    return new Promise(resolve => server.close(resolve));
  };
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, host, () => resolve({ port: server.address().port, received, kill }));
  });
}

module.exports = { startFakeEngine };

if (require.main === module) {
  const args = process.argv.slice(2), opt = name => { const i = args.indexOf(name); return i >= 0 ? args.splice(i, 2)[1] : undefined; };
  const port = parseInt(opt('--port') || '0', 10), drop = (opt('--drop') || '').split(',').filter(Boolean);
  const dump = args[0] ? JSON.parse(fs.readFileSync(args[0], 'utf8')) : null;
  startFakeEngine({ port, dump, drop }).then(engine => {
    console.log(`fake avplumber on 127.0.0.1:${engine.port}${dump ? ` replaying ${args[0]}` : ''}`);
    const quit = () => engine.kill().then(() => process.exit(0));
    process.on('SIGINT', quit);
    process.on('SIGTERM', quit);
  }, error => { console.error(error.message); process.exit(1); });
}
