// Svelte action: dark hover labels (the mixer page's .meter-tip look) for every [data-tip] element inside the node it
// is used on, instead of native title tooltips. One shared label element; it moves into the fullscreen element while
// one is shown, so it stays visible there.
let label = null, current = null;

function show(el) {
  if (!label) { label = document.createElement('div'); label.className = 'gv-htip'; label.hidden = true; }
  const parent = document.fullscreenElement || document.body;
  if (label.parentNode !== parent) parent.append(label);
  label.textContent = el.dataset.tip; label.hidden = false;
  const r = el.getBoundingClientRect(), w = label.offsetWidth, h = label.offsetHeight;
  label.style.left = Math.max(12, Math.min(innerWidth - w - 12, r.left + r.width / 2 - w / 2)) + 'px';
  label.style.top = (r.bottom + 8 + h > innerHeight ? r.top - h - 8 : r.bottom + 8) + 'px';
}
function hide() { if (label) label.hidden = true; current = null; }

export function hoverTips(node) {
  const over = ev => {
    const el = ev.target.closest?.('[data-tip]');
    if (el === current) return;
    current = el;
    if (el && node.contains(el)) show(el); else hide();
  };
  const out = ev => { if (!node.contains(ev.relatedTarget)) hide(); };
  node.addEventListener('pointerover', over);
  node.addEventListener('pointerout', out);
  node.addEventListener('pointerdown', hide);
  return {
    destroy() {
      node.removeEventListener('pointerover', over);
      node.removeEventListener('pointerout', out);
      node.removeEventListener('pointerdown', hide);
      if (current && node.contains(current)) hide();
    },
  };
}
