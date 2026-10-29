// Text shared by the graph view's canvas labels, hover cards and inspector.

export const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
export const kindClass = k => 's-' + k.toLowerCase();  // a source kind's colour (graph.css)
export const fmtK = v => v >= 1000 ? (v / 1000).toFixed(1) + 'k' : v.toFixed(0);
export const compact = v => new Intl.NumberFormat('en', {notation: 'compact', maximumFractionDigits: 1}).format(v);
export const pctText = x => !x ? '0%' : x < 1e-5 ? '<0.001%' : (x * 100).toFixed(x < 1e-3 ? 3 : 2) + '%';
const fmtAvg = v => String(Math.round(v * 100) / 100);

export const fillPct = q => q.c > 0 ? Math.min(100, q.avg / q.c * 100) : 0;
export const fillClass = q => q.st === 'drop' ? 'bad' : q.st === 'full' ? 'warn' : '';
export const framesText = q => `frames ${q.o} / ${fmtAvg(q.avg)} / ${q.mx} of ${q.c}`;
// Totals since start; a drop in the last sample is what turns a queue red.
export const dropText = q => (q.nb && q.dr ? `${q.dr} skipped (latest frame wins)` : `${q.dr} dropped`) + (q.st === 'drop' ? ` · ${q.drD} since the last sample` : '');
export const queueText = q => `${framesText(q)} · ${q.pps.toFixed(1)} pps · ${dropText(q)}`;

// Params for display: long lists cut to four items, long strings to 120 characters.
export function trimParams(v) {
  if (Array.isArray(v)) return [...v.slice(0, 4).map(trimParams), ...(v.length > 4 ? [`… ${v.length} items`] : [])];
  if (v && typeof v === 'object') return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, trimParams(x)]));
  if (typeof v === 'string' && v.length > 120) return v.slice(0, 117) + '…';
  return v;
}
