// Only the desktop shell owns the operator's window layout.
const fs = require('node:fs');
const path = require('node:path');

function openWorkspace(profile, screen, keepPositions) {
  const file = path.join(profile, 'mixer-workspace.json');
  let state = {}, timer;
  try {
    const saved = JSON.parse(fs.readFileSync(file, 'utf8'));
    if (saved && typeof saved === 'object' && !Array.isArray(saved)) state = saved;
  } catch (error) {
    if (error.code !== 'ENOENT') console.warn('Could not read mixer workspace:', error.message);
  }

  function flush() {
    clearTimeout(timer);
    try {
      fs.writeFileSync(file + '.tmp', JSON.stringify(state), { mode: 0o600 });
      fs.renameSync(file + '.tmp', file);
    } catch (error) {
      console.warn('Could not save mixer workspace:', error.message);
    }
  }
  function saveSoon() {
    clearTimeout(timer);
    timer = setTimeout(flush, 250);
  }
  function bounds(key, view) {
    const saved = state[key];
    if (!Number.isSafeInteger(saved?.width) || !Number.isSafeInteger(saved.height) ||
        saved.width <= 0 || saved.height <= 0) return { width: view.width, height: view.height };
    // Wayland lets the compositor choose the output and constrain the requested size.
    if (!keepPositions) return { width: saved.width, height: saved.height };
    const display = screen.getAllDisplays().find(d => d.id === saved.display);
    const area = (display || screen.getPrimaryDisplay()).workArea;
    const width = Math.min(saved.width, area.width), height = Math.min(saved.height, area.height);
    // After unplugging a monitor, center its windows on the primary display.
    const offset = (pos, room) => display && Number.isSafeInteger(pos)
      ? Math.max(0, Math.min(pos, room)) : Math.floor(room / 2);
    return { width, height, x: area.x + offset(saved.left, area.width - width),
             y: area.y + offset(saved.top, area.height - height) };
  }
  function remember(key, win, open) {
    const rect = win.getNormalBounds();
    const saved = { width: rect.width, height: rect.height, maximized: win.isMaximized(), open };
    if (keepPositions) {
      const display = screen.getDisplayMatching(rect);
      Object.assign(saved, { display: display.id, left: rect.x - display.workArea.x, top: rect.y - display.workArea.y });
    }
    state[key] = saved;
    saveSoon();
  }
  function closed(key) {
    if (state[key]) {
      state[key].open = false;
      saveSoon();
    }
  }
  return { state, bounds, remember, closed, flush };
}
module.exports = { openWorkspace };
