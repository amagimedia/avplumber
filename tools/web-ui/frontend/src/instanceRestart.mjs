// True when the backend reports a new generation for an instance that was already listed: the
// avplumber process behind it, or the backend itself, restarted under the same id
// (backend/instanceRestart.js). Backends without generations never report a restart.
export function instanceRestarted(previous, next, id) {
  const generation = list => (list || []).find(instance => instance && instance.id === id)?.generation;
  const before = generation(previous);
  const after = generation(next);
  return before !== undefined && after !== undefined && before !== after;
}
