export function renderCutSample(element, label, sample, unavailable = "No measurement") {
  const ms = sample?.state === "measured" && Number.isFinite(sample.ms) && sample.ms >= 0
    ? Math.round(sample.ms) : null;
  element.textContent = ms === null ? "—" : `${ms} ms`;
  element.dataset.level = ms === null ? "unknown" : ms <= 200 ? "good" : ms <= 400 ? "warn" : "bad";
  element.title = ms === null ? (sample?.state || unavailable)
    : `Last measured cut to ${sample.scene}; command receipt → encoder output (${sample.ms.toFixed(2)} ms)`;
  element.setAttribute("aria-label", `${label} cut latency ${ms === null ? "unavailable" : `${ms} milliseconds`}`);
}

// Uses the existing graph UI's WS→TCP bridge. This client only reads status;
// it never enables instrumentation, sends CUTs, or changes graph configuration.
export class CutLatencyMeter {
  constructor(config, render, { Socket = WebSocket, timer = globalThis } = {}) {
    if (!config || typeof config.ws_url !== "string" || !/^wss?:\/\//.test(config.ws_url)
        || typeof config.instance_id !== "string" || !config.instance_id
        || typeof config.mixer !== "string" || !/^[\w.-]+$/.test(config.mixer)) {
      throw Error("Invalid cut measurement configuration");
    }
    this.config = config;
    this.render = render;
    this.Socket = Socket;
    this.timer = timer;
    this.socket = null;
    this.pending = null;
    this.nextId = 0;
    this.pollTimer = null;
    this.deadline = null;
    this.stopped = true;
  }

  clearTimers() {
    this.timer.clearTimeout(this.pollTimer);
    this.timer.clearTimeout(this.deadline);
    this.pollTimer = this.deadline = null;
  }

  start() {
    this.stop();
    this.stopped = false;
    this.connect();
  }

  connect() {
    if (this.stopped) return;
    let socket;
    try { socket = new this.Socket(this.config.ws_url); }
    catch (_) { this.retry("Measurement connection unavailable"); return; }
    this.socket = socket;
    this.deadline = this.timer.setTimeout(() => this.retry("Measurement connection timeout"), 5000);
    socket.onopen = () => {
      if (this.socket !== socket || this.stopped) return;
      this.clearTimers();
      this.poll();
    };
    socket.onmessage = event => {
      if (this.socket !== socket || this.stopped) return;
      let response;
      try { response = JSON.parse(event.data); } catch (_) { return; }
      if (response.type !== "response" || response.id !== this.pending
          || response.instanceId !== this.config.instance_id) return;
      this.pending = null;
      this.clearTimers();
      try {
        if (!/^201\b/.test(response.statusLine)) throw Error("Status unavailable");
        const status = JSON.parse(response.body);
        const sample = status.cut_latency;
        if (!sample || sample.endpoint !== "encoder_output") throw Error("Mixer not instrumented");
        this.render(sample, "No cut measured yet");
      } catch (error) { this.render(null, error.message); }
      this.pollTimer = this.timer.setTimeout(() => this.poll(), 1000);
    };
    socket.onclose = socket.onerror = () => {
      if (this.socket === socket && !this.stopped) this.retry("Measurement connection lost");
    };
  }

  poll() {
    if (this.stopped || !this.socket || this.pending !== null) return;
    this.pending = `cut-latency-${++this.nextId}`;
    try {
      this.socket.send(JSON.stringify({ type: "command", id: this.pending,
        instanceId: this.config.instance_id, command: `mixer.status ${this.config.mixer}` }));
      this.deadline = this.timer.setTimeout(() => this.retry("Measurement data stale"), 3000);
    } catch (_) { this.retry("Measurement connection lost"); }
  }

  retry(reason) {
    this.clearTimers();
    const socket = this.socket;
    this.socket = null;
    this.pending = null;
    socket?.close();
    this.render(null, reason);
    if (!this.stopped) this.pollTimer = this.timer.setTimeout(() => this.connect(), 2000);
  }

  stop() {
    this.stopped = true;
    this.retry("Measurements disconnected");
  }
}

if (typeof document !== "undefined") {
  const direct = document.getElementById("cut-direct");
  const previewed = document.getElementById("cut-previewed");
  const render = (sample, reason) => {
    renderCutSample(direct, "AVP Direct", sample?.direct, reason);
    renderCutSample(previewed, "AVP Previewed", sample?.previewed, reason);
  };
  let meter;
  let stopped = false;
  window.addEventListener("beforeunload", () => { stopped = true; meter?.stop(); });
  fetch("metrics.json", { cache: "no-store", signal: AbortSignal.timeout(5000) })
    .then(response => { if (!response.ok) throw Error("Measurements not configured"); return response.json(); })
    .then(config => {
      if (stopped) return;
      meter = new CutLatencyMeter(config, render);
      meter.start();
    })
    .catch(error => render(null, error.message));
}
