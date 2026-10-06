import assert from "node:assert/strict";
import { ReceiverStats, PresentationStats } from "../receiver-stats.mjs";

const report = (overrides = {}) => new Map([
  ["v", { id: "v", type: "inbound-rtp", kind: "video", ssrc: 1, timestamp: 1000,
    framesDecoded: 30, jitterBufferDelay: 3, jitterBufferEmittedCount: 30,
    totalDecodeTime: 0.09, codecId: "c", freezeCount: 0, ...overrides }],
  ["c", { mimeType: "video/H265" }],
]);
const stats = new ReceiverStats();
assert.equal(stats.sample(report()).fps, null, "first sample has no interval");
let sample = stats.sample(report({ timestamp: 2000, framesDecoded: 60,
  jitterBufferDelay: 4.5, jitterBufferEmittedCount: 60, totalDecodeTime: 0.21 }));
assert.equal(sample.fps, 30);
assert.equal(sample.jitterBufferMs, 50, "use interval, not lifetime buffer average");
assert.ok(Math.abs(sample.decodeMs - 4) < 1e-10);
assert.equal(sample.codec, "video/H265");
assert.equal(sample.framesDropped, null, "unsupported counters are not zero");
sample = stats.sample(report({ timestamp: 3000, framesDecoded: 60,
  jitterBufferDelay: 4.5, jitterBufferEmittedCount: 60, totalDecodeTime: 0.21 }));
assert.equal(sample.fps, 0, "a stopped decoder is visible");
assert.equal(sample.jitterBufferMs, null, "no emitted frames means no residence sample");
assert.equal(sample.decodeMs, null);
const epoch = sample.epoch;
sample = stats.sample(report({ timestamp: 4000, ssrc: 2, framesDecoded: 1 }));
assert.equal(sample.fps, null);
assert.equal(sample.epoch, epoch + 1);
sample = stats.sample(report({ timestamp: 5000, ssrc: 2, framesDecoded: 0 }));
assert.equal(sample.fps, null);
assert.equal(sample.epoch, epoch + 2, "decoder reset starts a new epoch");
assert.equal(stats.sample(new Map()), null);
assert.equal(stats.sample(report()).fps, null);

const presentation = new PresentationStats();
const frame = (n, time) => presentation.frame(time, { presentedFrames: n, expectedDisplayTime: time });
frame(1, 1000); frame(2, 1033); frame(3, 1067);
assert.equal(presentation.sample(1100).presentationStalls, 0);
frame(4, 1567);
sample = presentation.sample(1600);
assert.equal(sample.presentationStalls, 1);
assert.equal(sample.maxPresentationGapMs, 500);
assert.equal(sample.frameAgeMs, 33);
assert.equal(presentation.sample(2000).frameAgeMs, 433, "an ongoing stall remains visible");
frame(20, 2200);
assert.equal(presentation.sample(2200).presentationStalls, 1,
  "missed JavaScript callbacks cannot establish a presentation freeze");
presentation.reset(); frame(21, 8000);
assert.equal(presentation.sample(8000).presentationStalls, 1, "hidden/paused interval is excluded");
console.log("Receiver interval metrics, missing counters, resets, and presentation gaps passed");
