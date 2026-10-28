// Checks that keep the web UI a read-only viewer: which control commands /ws forwards, which
// WebSocket handshakes it accepts and which ids it stores. Pure functions, tested by
// tests/requestGuards.test.cjs.

// The read-only queries the viewer (frontend/src/App.svelte) and the preview's cut-latency meter
// (docker-compose/images/preview/cut-latency.mjs) send, with their number of arguments.
const READ_ONLY_COMMANDS = new Map([
  ['nodes.json', 0],
  ['queues.json', 0],
  ['queues.stats', 0],
  ['sync_groups.json', 0],
  ['correction_groups.json', 0],
  ['node.object.get', 2], // <node> <object>
  ['mixer.status', 1] // <mixer>
]);

const CONTROL_CHARS = /[\x00-\x1f\x7f]/;
// Node, object and mixer names as the bundled scripts and the Python mixer write them.
const NAME_ARGUMENT = /^[\w.@:-]{1,128}$/;

// Returns the command line to send to the engine, or null when it is not an allowed read-only
// query. Arguments must be names (NAME_ARGUMENT), and no control character may appear anywhere,
// so one message can never carry a second command line or a JSON payload.
function readOnlyCommand(command) {
  if (typeof command !== 'string' || CONTROL_CHARS.test(command)) return null;
  const [name, ...args] = command.trim().split(/ +/);
  if (READ_ONLY_COMMANDS.get(name) !== args.length) return null;
  if (!args.every((arg) => NAME_ARGUMENT.test(arg))) return null;
  return [name, ...args].join(' ');
}

const RESERVED_IDS = new Set(['__proto__', 'constructor', 'prototype']);

// Instance ids and stats stream names become dictionary keys here and in every tab.
function isSafeId(id) {
  return typeof id === 'string' && id !== '' && !RESERVED_IDS.has(id);
}

const unbracket = (hostname) => hostname.replace(/^\[(.*)\]$/, '$1');

// Host header ("host", "host:port" or "[v6]:port") -> lower-case hostname without brackets, or null.
function hostnameOf(authority) {
  if (typeof authority !== 'string' || !/^[^\s@/?#\\]+$/.test(authority)) return null;
  try {
    return unbracket(new URL(`http://${authority}`).hostname);
  } catch {
    return null;
  }
}

const WILDCARD_BINDS = new Set(['0.0.0.0', '::']);

// Hostnames a WebSocket handshake may name in its Host header: loopback, the bind address unless
// it is a wildcard, and the comma-separated hostnames (no ports) of WEBUI_ALLOWED_HOSTS.
function allowedHostnames(bind, extraHosts) {
  const names = ['localhost', '127.0.0.1', '::1', ...String(extraHosts || '').split(',')];
  if (bind && !WILDCARD_BINDS.has(bind)) names.push(bind);
  return new Set(names.map((name) => unbracket(name.trim().toLowerCase())).filter(Boolean));
}

// A Host outside the allow-list means DNS rebinding (an attacker's name resolving to this
// server). A browser Origin on another hostname means a cross-site page; clients without an
// Origin header are not browsers and are accepted.
function upgradeAllowed(headers, allowed) {
  const host = hostnameOf(headers.host);
  if (!host || !allowed.has(host)) return false;
  if (headers.origin === undefined) return true;
  try {
    return unbracket(new URL(headers.origin).hostname) === host;
  } catch {
    return false;
  }
}

// POST bodies must be declared JSON: browsers cannot send that cross-origin without a CORS
// preflight, which the backend never grants.
function isJsonRequest(headers) {
  const type = String(headers['content-type'] || '').split(';')[0].trim().toLowerCase();
  return type === 'application/json';
}

module.exports = { readOnlyCommand, isSafeId, allowedHostnames, upgradeAllowed, isJsonRequest };
