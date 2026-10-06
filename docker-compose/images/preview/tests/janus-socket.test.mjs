import assert from "node:assert/strict";
import test from "node:test";
import { JanusSocket } from "../janus-socket.mjs";

class Socket {
  sent = [];
  constructor(url, protocol) { Object.assign(this, { url, protocol }); }
  send(message) { this.sent.push(JSON.parse(message)); }
  close() { this.closed = true; }
  reply(payload) { this.onmessage({ data: JSON.stringify(payload) }); }
}

function fakeTimer() {
  const tasks = new Map();
  let id = 0;
  const add = (fn, ms, repeat = false) => { tasks.set(++id, { fn, ms, repeat }); return id; };
  const remove = task => { tasks.delete(task); };
  return { tasks, setTimeout: add, setInterval: (fn, ms) => add(fn, ms, true), clearTimeout: remove, clearInterval: remove,
    fire(ms) {
      for (const [key, task] of [...tasks]) if (task.ms === ms) {
        if (!task.repeat) tasks.delete(key);
        task.fn();
      }
    } };
}

// Lets the promise chains inside the transport run.
const settle = () => new Promise(resolve => setImmediate(resolve));

function connect({ url = "http://127.0.0.1/janus", onEvent, open = true } = {}) {
  const timer = fakeTimer(), events = [], errors = [];
  const janus = new JanusSocket(url, { Socket, timer, onError: error => errors.push(error.message),
    onEvent: onEvent || (event => { events.push(event); }) });
  if (open) janus.socket.onopen();
  return { janus, socket: janus.socket, timer, events, errors };
}

test("the WebSocket is at the HTTP API's address, with the Janus subprotocol", () => {
  assert.equal(connect().socket.url, "ws://127.0.0.1/janus");
  const { socket } = connect({ url: "https://mixer.example/preview/janus" });
  assert.equal(socket.url, "wss://mixer.example/preview/janus");
  assert.equal(socket.protocol, "janus-protocol");
});

test("a request waits for the socket and settles on the reply with its transaction", async () => {
  const { janus, socket, timer, events } = connect({ open: false });
  const attach = janus.request({ janus: "attach", plugin: "janus.plugin.streaming" });
  const trickle = janus.request({ janus: "trickle", handle_id: 22, candidate: { completed: true } });
  await settle();
  assert.equal(socket.sent.length, 0, "nothing is sent on a connecting socket");
  socket.onopen();
  await settle();
  const [first, second] = socket.sent;
  assert.deepEqual({ ...first, transaction: 0 },
    { janus: "attach", plugin: "janus.plugin.streaming", transaction: 0 });
  assert.equal(second.handle_id, 22);
  assert.notEqual(first.transaction, second.transaction);
  socket.reply({ janus: "ack", transaction: second.transaction });
  socket.reply({ janus: "success", transaction: first.transaction, data: { id: 22 } });
  assert.equal((await trickle).janus, "ack");
  assert.equal((await attach).data.id, 22);
  assert.equal(timer.tasks.size, 0, "a reply cancels the request's deadline");
  assert.deepEqual(events, [], "replies are not events");
});

test("the session's requests carry its id, and it is kept alive every 25 s", async () => {
  const { janus, socket, timer } = connect();
  const created = janus.createSession();
  await settle();
  assert.equal(socket.sent[0].janus, "create");
  assert.equal("session_id" in socket.sent[0], false);
  socket.reply({ janus: "success", transaction: socket.sent[0].transaction, data: { id: 11 } });
  await created;
  timer.fire(25000);
  timer.fire(25000);
  await settle();
  assert.deepEqual(socket.sent.slice(1).map(({ janus, session_id }) => [janus, session_id]),
    [["keepalive", 11], ["keepalive", 11]]);
  for (const request of socket.sent.slice(1)) socket.reply({ janus: "ack", transaction: request.transaction });
  await settle();
  janus.request({ janus: "attach" }).catch(() => {});
  await settle();
  assert.equal(socket.sent.at(-1).session_id, 11);
});

test("an error reply and a missing reply reject", async () => {
  const { janus, socket, timer, errors } = connect();
  const refused = janus.request({ janus: "attach" });
  await settle();
  socket.reply({ janus: "error", transaction: socket.sent[0].transaction,
    error: { code: 458, reason: "No such session" } });
  await assert.rejects(refused, /No such session/);
  const unanswered = janus.request({ janus: "message" });
  timer.fire(10000);
  await assert.rejects(unanswered, /Janus message timed out/);
  assert.deepEqual(errors, [], "a failed request is the caller's, not the connection's");
});

test("everything but a reply goes to onEvent, one at a time and in order", async () => {
  const seen = [];
  let release;
  const { janus, socket, errors } = connect({ onEvent: async event => {
    seen.push(event.janus);
    if (event.jsep) await new Promise(resolve => { release = resolve; });
    if (event.janus === "hangup") throw Error(event.reason);
    seen.push(`${event.janus} done`);
  } });
  const watch = janus.request({ janus: "message", handle_id: 22, body: { request: "watch", id: 1 } });
  await settle();
  const { transaction } = socket.sent[0];
  socket.reply({ janus: "ack", transaction });
  await watch;
  // Janus sends the plugin's answer to an acked request under the request's transaction.
  socket.reply({ janus: "event", transaction, jsep: { type: "offer", sdp: "" } });
  socket.reply({ janus: "webrtcup" });
  for (const janusType of ["media", "slowlink", "trickle", "timeout"]) socket.reply({ janus: janusType });
  await settle();
  assert.deepEqual(seen, ["event"], "the next event waits for the handler of the offer");
  release();
  await settle();
  assert.deepEqual(seen.slice(1, 4), ["event done", "webrtcup", "webrtcup done"]);
  assert.equal(seen.length, 2 + 2 * 5);
  socket.reply({ janus: "hangup", reason: "ICE failed" });
  socket.reply({ janus: "error", error: { reason: "unsolicited" } });
  await settle();
  assert.deepEqual(errors, ["ICE failed"], "a failing handler reports to onError");
  assert.deepEqual(seen.slice(-3), ["hangup", "error", "error done"]);
});

test("a lost socket rejects what is pending and reports once", async () => {
  const { janus, socket, timer, errors } = connect();
  const pending = janus.request({ janus: "create" });
  socket.onerror();
  socket.onclose();
  await assert.rejects(pending, /Janus connection lost/);
  assert.deepEqual(errors, ["Janus connection lost"]);
  assert.equal(timer.tasks.size, 0);
});

test("close() rejects what is pending and calls nothing afterwards", async () => {
  const { janus, socket, timer, events, errors } = connect();
  const created = janus.createSession();
  await settle();
  socket.reply({ janus: "success", transaction: socket.sent[0].transaction, data: { id: 11 } });
  await created;
  const pending = janus.request({ janus: "attach" });
  await settle();
  janus.close();
  assert.equal(socket.closed, true);
  await assert.rejects(pending, /Janus connection closed/);
  assert.equal(timer.tasks.size, 0, "no keepalive and no deadline is left");
  socket.reply({ janus: "success", transaction: socket.sent.at(-1).transaction, data: { id: 22 } });
  socket.reply({ janus: "hangup" });
  socket.onclose();
  await settle();
  assert.deepEqual([events, errors], [[], []]);
  await assert.rejects(janus.request({ janus: "keepalive" }), /Janus connection closed/);
  assert.equal(socket.sent.length, 2);
});

test("a send failure rejects promptly and reports one connection failure", async () => {
  const { janus, socket, timer, errors } = connect();
  socket.send = () => { throw Error("socket closed during send"); };
  await assert.rejects(janus.request({ janus: "create" }), /closed during send/);
  assert.deepEqual(errors, ["socket closed during send"]);
  assert.equal(timer.tasks.size, 0);
});

test("a lost keepalive triggers recovery", async () => {
  const { janus, socket, timer, errors } = connect();
  const created = janus.createSession();
  await settle();
  socket.reply({ janus: "success", transaction: socket.sent[0].transaction, data: { id: 11 } });
  await created;
  timer.fire(25000);
  await settle();
  timer.fire(10000);
  await settle();
  assert.deepEqual(errors, ["Janus keepalive timed out"]);
  assert.equal(timer.tasks.size, 0);
});

test("closing just after create succeeds cannot leave a keepalive running", async () => {
  const { janus, socket, timer, errors } = connect();
  const created = janus.createSession();
  await settle();
  socket.reply({ janus: "success", transaction: socket.sent[0].transaction, data: { id: 11 } });
  janus.close();
  await assert.rejects(created, /connection closed/);
  assert.equal(timer.tasks.size, 0);
  assert.deepEqual(errors, []);
});
