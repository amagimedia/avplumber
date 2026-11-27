// Run with node --test. Display topology and Wayland placement need no window server.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const { openWorkspace } = require('../../demos/mixer/desktop/workspace.cjs');

test('restore relative to the saved display and recover when it moves, shrinks or is removed', t => {
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'mixer-workspace-'));
  t.after(() => fs.rmSync(profile, { recursive: true, force: true }));
  const primary = { id: 1, workArea: { x: 0, y: 20, width: 1600, height: 900 } };
  const second = { id: 2, workArea: { x: -1920, y: 0, width: 1920, height: 1080 } };
  let displays = [primary, second];
  const screen = { getAllDisplays: () => displays, getPrimaryDisplay: () => primary, getDisplayMatching: () => second };
  const view = { width: 1280, height: 820 };
  const win = { getNormalBounds: () => ({ x: -1400, y: 210, width: 1100, height: 800 }), isMaximized: () => true };
  const initial = openWorkspace(profile, screen, true);
  assert.deepEqual(initial.bounds('graph', view), view);
  initial.remember('graph', win, true);
  initial.flush();

  const restored = openWorkspace(profile, screen, true);
  assert.deepEqual(restored.bounds('graph', view), win.getNormalBounds());
  assert.equal(restored.state.graph.maximized, true);
  second.workArea = { x: 1600, y: 20, width: 1200, height: 750 };
  assert.deepEqual(restored.bounds('graph', view), { x: 1700, y: 20, width: 1100, height: 750 });
  displays = [primary];
  assert.deepEqual(restored.bounds('graph', view), { x: 250, y: 70, width: 1100, height: 800 });
  restored.closed('graph');
  restored.flush();
  assert.equal(openWorkspace(profile, screen, true).state.graph.open, false);

  const wayland = openWorkspace(profile, {}, false);
  assert.deepEqual(wayland.bounds('graph', view), { width: 1100, height: 800 });
  wayland.remember('graph', win, true);
  wayland.flush();
  assert.deepEqual(wayland.state.graph, { width: 1100, height: 800, maximized: true, open: true });
});
