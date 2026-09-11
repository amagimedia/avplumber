import assert from "node:assert/strict";
import { CutLatencyMeter, renderCutSample } from "../cut-latency.mjs";

const element = { dataset: {}, setAttribute() {} };
for (const [ms, level] of [[0, "good"], [200, "good"], [201, "warn"], [400, "warn"], [401, "bad"]]) {
  renderCutSample(element, "Direct", { state: "measured", ms, scene: "B" });
  assert.equal(element.textContent, `${ms} ms`);
  assert.equal(element.dataset.level, level);
}
for (const ms of [null, undefined, NaN, Infinity, -1, "100"]) {
  renderCutSample(element, "Direct", { state: "measured", ms });
  assert.equal(element.textContent, "—");
  assert.equal(element.dataset.level, "unknown");
}
for (const state of ["pending", "interrupted", "timeout", "unmatched"]) {
  renderCutSample(element, "Direct", { state, ms: 50 });
  assert.equal(element.textContent, "—");
}

const timers = new Map();
let timerId = 0;
const timer = {
  setTimeout(fn, ms) { timers.set(++timerId, { fn, ms }); return timerId; },
  clearTimeout(id) { timers.delete(id); },
};
function tick(ms) {
  const entry = [...timers.entries()].find(([, task]) => task.ms === ms);
  assert.ok(entry, `missing ${ms} ms timer`);
  timers.delete(entry[0]);
  entry[1].fn();
}
class Socket {
  static all = [];
  sent = [];
  constructor() { Socket.all.push(this); }
  send(message) { this.sent.push(JSON.parse(message)); }
  close() { this.closed = true; this.onclose?.(); }
  reply(payload) { this.onmessage({ data: JSON.stringify(payload) }); }
}
const config = { ws_url: "ws://127.0.0.1:22222/ws", instance_id: "test", mixer: "mixer" };
const renders = [];
const meter = new CutLatencyMeter(config, (...args) => renders.push(args), { Socket, timer });
for (const bad of [{}, { ...config, mixer: "mixer\nnode.stop_all" }, { ...config, ws_url: "file:///test" }]) {
  assert.throws(() => new CutLatencyMeter(bad, () => {}, { Socket, timer }));
}
meter.start();
const first = Socket.all.at(-1);
first.onopen();
assert.equal(first.sent.length, 1);
assert.equal(first.sent[0].command, "mixer.status mixer");
assert.equal(first.sent[0].instanceId, "test");
meter.poll();
assert.equal(first.sent.length, 1, "only one request may be in flight");
const response = { type: "response", id: first.sent[0].id, instanceId: "test", statusLine: "201 OK",
  body: JSON.stringify({ cut_latency: { endpoint: "encoder_output", direct: { state: "measured", ms: 133 } } }) };
const before = renders.length;
first.reply({ ...response, instanceId: "another" });
first.reply({ ...response, id: "another" });
assert.equal(renders.length, before);
first.reply(response);
assert.equal(renders.at(-1)[0].direct.ms, 133);
assert.equal(timers.size, 1);
tick(1000);
first.reply({ ...response, id: first.sent[1].id, body: "{}" });
assert.equal(renders.at(-1)[0], null);
assert.equal(renders.at(-1)[1], "Mixer not instrumented");
tick(1000);
tick(3000);
assert.equal(renders.at(-1)[0], null);
assert.equal(first.closed, true);
const stale = renders.length;
first.reply({ ...response, id: first.sent[2].id });
assert.equal(renders.length, stale, "late responses cannot resurrect stale metrics");
tick(2000);
const second = Socket.all.at(-1);
second.onopen();
meter.stop();
second.reply({ ...response, id: second.sent[0].id });
assert.equal(renders.at(-1)[0], null);
assert.equal(timers.size, 0);

meter.start();
tick(5000);
assert.equal(renders.at(-1)[1], "Measurement connection timeout");
meter.stop();
assert.equal(timers.size, 0);
console.log("Cut latency rendering, read-only WS sampling, timeouts, reconnect and stale-response checks passed");
