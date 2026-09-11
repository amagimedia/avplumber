import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createContext, runInContext } from "node:vm";

const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map();
const timers = new Map();
let timerId = 0;
const context = createContext({
  location: { origin: "http://127.0.0.1" },
  document: {
    readyState: "loading",
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, { dataset: {}, setAttribute() {} });
      return elements.get(id);
    },
  },
  window: {
    addEventListener() {},
    setTimeout(fn) { timers.set(++timerId, fn); return timerId; },
    clearTimeout(id) { timers.delete(id); },
  },
});
runInContext(script, context);

function report(seconds, state = "succeeded") {
  return new Map([
    ["video", { type: "inbound-rtp", kind: "video", transportId: "transport" }],
    ["transport", { type: "transport", selectedCandidatePairId: "selected" }],
    ["selected", { type: "candidate-pair", state, currentRoundTripTime: seconds }],
    ["unselected", { type: "candidate-pair", state: "succeeded", nominated: true, currentRoundTripTime: 9 }],
  ]);
}

assert.equal(context.selectedRttMs(report(0.1234)), 123);
assert.equal(context.selectedRttMs(report(0)), 0);
assert.equal(context.selectedRttMs(report(0.2, "in-progress")), null);
for (const value of [undefined, null, NaN, Infinity, -1, "0.1"]) {
  assert.equal(context.selectedRttMs(report(value)), null);
}
const noSelected = report(0.1);
noSelected.delete("selected");
assert.equal(context.selectedRttMs(noSelected), null);
const noVideo = report(0.2);
noVideo.delete("video");
assert.equal(context.selectedRttMs(noVideo), 200);

const value = elements.get("rtt-value");
for (const [ms, level] of [[0, "good"], [200, "good"], [201, "warn"], [400, "warn"], [401, "bad"]]) {
  context.renderRtt(ms);
  assert.equal(value.textContent, `${ms} ms`);
  assert.equal(value.dataset.level, level);
}
context.renderRtt(null);
assert.equal(value.textContent, "—");
assert.equal(value.dataset.level, "unknown");

async function sample(connection) {
  context.connection = connection;
  runInContext("peer = connection; startRttMeter(peer)", context);
  await Promise.resolve();
  await Promise.resolve();
}
await sample({ connectionState: "connected", getStats: async () => report(0.25) });
assert.equal(value.textContent, "250 ms");
assert.equal(timers.size, 1);
context.stopRttMeter();
assert.equal(timers.size, 0);
assert.equal(value.textContent, "—");

await sample({ connectionState: "connected", getStats: async () => { throw Error("unavailable"); } });
assert.equal(value.textContent, "—");
assert.equal(timers.size, 1);
context.stopRttMeter();

let resolveStats;
await sample({ connectionState: "connected", getStats: () => new Promise(resolve => { resolveStats = resolve; }) });
context.stopRttMeter();
resolveStats(report(0.9));
await Promise.resolve();
await Promise.resolve();
assert.equal(value.textContent, "—");
assert.equal(timers.size, 0, "a stopped request must not restart sampling");
console.log("RTT selection, thresholds, unavailable data, failures and stop-race checks passed");
