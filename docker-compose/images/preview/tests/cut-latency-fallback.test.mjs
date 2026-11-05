import assert from "node:assert/strict";

const element = { dataset: {}, setAttribute() {} };
const requests = [];
const timers = [];
let unload;
let nextResponse = { status: { cut_latency: {
  endpoint: "encoder_output",
  direct: { state: "measured", recent: [{ id: 1, state: "measured", ms: 42 }] },
  previewed: { state: "measured", recent: [] },
} } };
const original = Object.fromEntries(["document", "window", "fetch", "setTimeout"]
  .map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
try {
  globalThis.document = { getElementById: () => element };
  globalThis.window = { addEventListener: (name, callback) => {
    assert.equal(name, "beforeunload");
    unload = callback;
  } };
  globalThis.setTimeout = (callback, delay) => {
    assert.equal(delay, 1000);
    timers.push(callback);
  };
  globalThis.fetch = async (url, options) => {
    requests.push(url);
    assert.equal(options.cache, "no-store");
    assert.ok(options.signal instanceof AbortSignal);
    return { ok: url !== "metrics.json", json: async () => nextResponse };
  };
  await import("../cut-latency.mjs");
  await new Promise(setImmediate);
  assert.deepEqual(requests, ["metrics.json", "mixer-state"]);
  assert.equal(element.textContent, "42 ms");

  nextResponse = { status: { cut_latency: { endpoint: "unrelated" } } };
  await timers.shift()();
  assert.equal(element.textContent, "—", "invalid state clears the previous measurement");
  assert.equal(element.dataset.level, "unknown");

  let resolve;
  nextResponse = new Promise(done => { resolve = done; });
  const pending = timers.shift()();
  unload();
  resolve({ status: { cut_latency: { endpoint: "encoder_output" } } });
  await pending;
  assert.equal(element.textContent, "—", "late responses cannot update an unloaded page");
  assert.equal(timers.length, 0, "unloading stops polling");
} finally {
  for (const [key, descriptor] of Object.entries(original)) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else delete globalThis[key];
  }
}
console.log("Embedded mixer state fallback, invalid samples and unload checks passed");
