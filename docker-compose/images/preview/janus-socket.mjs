// The Janus API over one WebSocket instead of HTTP long polling: a browser opens six HTTP/1.1
// connections per host, so the polls of six players on one page kept every further player from
// signalling. A request settles on the "success", "ack" or "error" that carries its transaction.
// Everything else goes to onEvent, also the plugin event that follows an "ack" under the same
// transaction (the offer answering "watch"). Janus destroys the session when the socket closes.
export class JanusSocket {
  pending = new Map();
  transactions = 0;
  events = Promise.resolve();
  closed = false;

  // `url` is the Janus HTTP API address, http(s)://…/janus: the WebSocket is at the same path.
  constructor(url, { onEvent, onError, Socket = WebSocket, timer = globalThis }) {
    this.onEvent = onEvent;
    this.onError = onError;
    this.timer = timer;
    this.socket = new Socket(url.replace(/^http/, "ws"), "janus-protocol");
    this.opened = new Promise(resolve => { this.socket.onopen = resolve; });
    this.socket.onmessage = ({ data }) => {
      if (this.closed) return;
      try { this.receive(JSON.parse(data)); }
      catch (error) { this.fail(error); }
    };
    this.socket.onclose = this.socket.onerror = () => this.fail(Error("Janus connection lost"));
  }

  fail(error) {
    if (this.closed) return;
    this.close(error);
    this.onError(error);
  }

  receive(message) {
    if (this.closed) return;
    const request = this.pending.get(message.transaction);
    if (request && ["success", "ack", "error"].includes(message.janus)) {
      this.pending.delete(message.transaction);
      this.timer.clearTimeout(request.deadline);
      if (message.janus === "error") request.reject(Error(message.error?.reason || "Janus error"));
      else request.resolve(message);
    } else {
      // One at a time, in arrival order: onEvent may await (it sets the peer connection up from
      // the offer) and the next event relies on that being done.
      this.events = this.events.then(() => this.closed || this.onEvent(message))
        .catch(error => this.closed || this.onError(error));
    }
  }

  request(body) {
    if (this.closed) return Promise.reject(Error("Janus connection closed"));
    const transaction = `avp-${Date.now()}-${++this.transactions}`;
    return new Promise((resolve, reject) => {
      const deadline = this.timer.setTimeout(() => {
        this.pending.delete(transaction);
        reject(Error(`Janus ${body.janus} timed out`));
      }, 10000);
      this.pending.set(transaction, { resolve, reject, deadline });
      this.opened.then(() => {
        if (this.pending.has(transaction)) {
          try { this.socket.send(JSON.stringify({ ...body, transaction, session_id: this.sessionId })); }
          catch (error) { this.fail(error); }
        }
      });
    });
  }

  // Every later request belongs to this session. Janus drops a session that sends nothing for 60 s.
  async createSession() {
    this.sessionId = (await this.request({ janus: "create" })).data.id;
    if (this.closed) throw Error("Janus connection closed");
    this.keepalive = this.timer.setInterval(
      () => this.request({ janus: "keepalive" }).catch(error => this.fail(error)), 25000);
  }

  // Pending requests reject; neither onEvent nor onError is called afterwards.
  close(error = Error("Janus connection closed")) {
    if (this.closed) return;
    this.closed = true;
    this.timer.clearInterval(this.keepalive);
    for (const { reject, deadline } of this.pending.values()) {
      this.timer.clearTimeout(deadline);
      reject(error);
    }
    this.pending.clear();
    this.socket.close();
  }
}
