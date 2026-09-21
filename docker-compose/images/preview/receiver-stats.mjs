// Receiver statistics are sampled as interval deltas: lifetime averages hide
// short stalls, and counters may restart when the SSRC/decoder changes.
export class ReceiverStats {
  previous = null;
  epoch = 0;

  sample(report) {
    const video = [...report.values()].find(s => s.type === "inbound-rtp"
      && (s.kind === "video" || s.mediaType === "video"));
    if (!video) { this.previous = null; return null; }
    const old = this.previous;
    this.previous = { ...video };
    if (!old || video.id !== old.id || video.ssrc !== old.ssrc
      || video.framesDecoded < old.framesDecoded) this.epoch += 1;
    const valid = old && video.id === old.id && video.ssrc === old.ssrc
      && video.timestamp > old.timestamp && video.framesDecoded >= old.framesDecoded;
    const delta = key => valid && Number.isFinite(video[key]) && Number.isFinite(old[key])
      && video[key] >= old[key] ? video[key] - old[key] : null;
    const averageMs = (total, count) => {
      const n = delta(count), value = delta(total);
      return n > 0 && value !== null ? value * 1000 / n : null;
    };
    const frames = delta("framesDecoded");
    const sample = {
      epoch: this.epoch,
      fps: frames === null ? null : frames * 1000 / (video.timestamp - old.timestamp),
      jitterBufferMs: averageMs("jitterBufferDelay", "jitterBufferEmittedCount"),
      decodeMs: averageMs("totalDecodeTime", "framesDecoded"),
      codec: report.get(video.codecId)?.mimeType ?? "",
    };
    for (const key of ["framesDecoded", "framesRendered", "framesDropped", "freezeCount",
      "pauseCount", "totalFreezesDuration", "totalPausesDuration", "packetsReceived",
      "packetsLost", "retransmittedPacketsReceived", "nackCount", "pliCount", "bytesReceived",
      "frameWidth", "frameHeight"]) {
      sample[key] = Number.isFinite(video[key]) ? video[key] : null;
    }
    return sample;
  }
}

export class PresentationStats {
  previous = null;
  stalls = 0;
  maxGapMs = 0;
  lastCallback = null;

  reset() { this.previous = null; this.lastCallback = null; }

  frame(now, metadata) {
    const old = this.previous;
    this.previous = metadata;
    this.lastCallback = now;
    // A skipped callback is not evidence of a skipped video presentation.
    if (old && metadata.presentedFrames === old.presentedFrames + 1) {
      const gap = metadata.expectedDisplayTime - old.expectedDisplayTime;
      this.maxGapMs = Math.max(this.maxGapMs, gap);
      if (gap >= 250) this.stalls += 1;
    }
  }

  sample(now) {
    const sample = {
      presentationStalls: this.stalls,
      maxPresentationGapMs: this.maxGapMs,
      frameAgeMs: this.lastCallback === null ? null : Math.max(0, now - this.lastCallback),
    };
    this.maxGapMs = 0;
    return sample;
  }
}

export function createReceiverMonitor(video) {
  const stats = new ReceiverStats(), presentation = new PresentationStats();
  const receiver = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  video.dataset.receiver = receiver;
  let stopped = false, callback = null, posting = false;
  const active = () => !document.hidden && !video.paused && !video.ended;
  const reset = () => presentation.reset();
  const frame = (now, metadata) => {
    if (stopped) return;
    if (active()) presentation.frame(now, metadata);
    else reset();
    callback = video.requestVideoFrameCallback(frame);
  };
  if (video.requestVideoFrameCallback) callback = video.requestVideoFrameCallback(frame);
  document.addEventListener("visibilitychange", reset);
  video.addEventListener("pause", reset);
  video.addEventListener("playing", reset);
  const render = sample => {
    for (const [id, value, unit] of [
      ["receiver-fps", sample?.fps, "fps"],
      ["receiver-buffer", sample?.jitterBufferMs, "ms"],
      ["receiver-freezes", sample?.freezeCount, ""],
    ]) {
      const element = document.getElementById(id);
      if (element) element.textContent = Number.isFinite(value)
        ? `${Math.round(value)}${unit ? ` ${unit}` : ""}` : "—";
    }
  };
  return {
    update(report, rttMs) {
      const sample = stats.sample(report);
      render(sample);
      if (!sample || stopped) return;
      Object.assign(sample, presentation.sample(performance.now()), {
        rttMs, visible: !document.hidden, paused: video.paused,
      });
      if (!posting) {
        posting = true;
        fetch("receiver-stats", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ receiver, sample }), signal: globalThis.AbortSignal?.timeout?.(3000),
        }).catch(() => {}).finally(() => { posting = false; });
      }
    },
    stop() {
      stopped = true;
      if (callback !== null) video.cancelVideoFrameCallback(callback);
      document.removeEventListener("visibilitychange", reset);
      video.removeEventListener("pause", reset);
      video.removeEventListener("playing", reset);
      render(null);
    },
  };
}
