// Detects that the avplumber process behind a registered instance restarted on the same
// host:port, so open graph tabs can reload nodes.json once instead of polling it.
//
// The engine's heartbeat carries no process identity, so the signal is: a control connection
// to the instance was closed by the peer (or refused), and then a heartbeat arrived. A restarted
// engine may send its first heartbeat before its graph exists (registerWithWebUI runs before the
// script or the mixer builds it), but its control server holds commands until setReady(), so the
// reload still receives every node. A connection lost without a restart costs one extra
// nodes.json fetch on the next heartbeat; an instance whose control port the backend cannot
// reach costs one at every heartbeat while a tab keeps it selected.

// Generations are seeded from the clock so they also change across backend restarts: a tab
// that reconnects to a restarted backend reloads when its instance re-registers.
function generationCounter(start = Date.now()) {
  let next = start;
  return () => next++;
}

// inst is undefined when the instance was removed (heartbeat timeout) before the connection was lost.
function noteConnectionLost(inst) {
  if (inst) inst.connectionLost = true;
}

// Called on every heartbeat of an already registered instance. Returns true, and assigns a new
// generation, when the heartbeat follows a lost control connection.
function noteHeartbeat(inst, nextGeneration) {
  if (!inst.connectionLost) return false;
  inst.connectionLost = false;
  inst.generation = nextGeneration();
  return true;
}

// Called by the periodic cleanup. Returns true when inst's heartbeats stopped and it should be
// removed. An instance registered explicitly (AVPLUMBER_PORT or POST /api/instances) is kept and
// only stops being heartbeat-tracked: the engine's heartbeat carries no id, so after a removal its
// next heartbeat would create a `host:port` entry that tabs showing the explicit id never select.
function heartbeatExpired(inst, now, timeoutMs) {
  if (!inst.usesHeartbeat || !inst.lastHeartbeat || now - inst.lastHeartbeat <= timeoutMs) return false;
  if (!inst.explicit) return true;
  inst.usesHeartbeat = false;
  inst.lastHeartbeat = null;
  return false;
}

module.exports = { generationCounter, noteConnectionLost, noteHeartbeat, heartbeatExpired };
