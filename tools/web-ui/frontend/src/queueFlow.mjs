const finite = value => typeof value === 'number' && Number.isFinite(value);
const delta = (current, previous, key) => finite(current[key]) && finite(previous?.[key]) && current[key] >= previous[key]
  ? current[key] - previous[key] : null;

// One observation per existing telemetry response, never per rendered frame.
export function sampleQueueFlow(queues, previous = []) {
  const old = new Map(previous.map(queue => [queue.name, queue]));
  return queues.map(queue => {
    const before = old.get(queue.name);
    const dequeued = delta(queue, before, 'dequeued_total');
    const enqueued = delta(queue, before, 'enqueued_total');
    const dropped = delta(queue, before, 'dropped_total');
    const known = dequeued !== null && enqueued !== null && dropped !== null;
    const growing = known && queue.occupied > before.occupied
      ? (before.flow?.growingSamples || 0) + 1 : 0;
    const held = known && dequeued === 0 && queue.occupied > 0
      ? (before.flow?.heldSamples || 0) + 1 : 0;
    return {...queue, flow: {
      known, active: known && dequeued > 0, dropped: known ? dropped : 0,
      growingSamples: growing, heldSamples: held,
      growing: growing >= 2, held: held >= 2,
    }};
  });
}

export function summarizeQueueFlow(queues) {
  const count = predicate => queues.filter(queue => queue && predicate(queue)).length;
  const dropped = queues.reduce((sum, queue) => sum + (queue?.flow?.dropped || 0), 0);
  const summary = {
    total: queues.length,
    known: count(queue => queue.flow?.known),
    active: count(queue => queue.flow?.active),
    growing: count(queue => queue.flow?.growing),
    held: count(queue => queue.flow?.held),
    dropping: count(queue => queue.flow?.dropped > 0),
    dropped,
    maxFill: Math.max(0, ...queues.map(queue => queue?.capacity > 0 && finite(queue.occupied)
      ? Math.max(0, Math.min(1, queue.occupied / queue.capacity)) : 0)),
  };
  summary.state = dropped > 0 ? 'dropped' : summary.growing || summary.held ? 'backlog'
    : summary.active ? 'flowing' : summary.known === summary.total && summary.total ? 'idle' : 'unknown';
  return summary;
}
