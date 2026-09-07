<script lang="ts">
  import type { Position } from 'rete-area-plugin';
  import { queueStatsByName } from './graphStores';
  import { summarizeQueueFlow } from './queueFlow.mjs';

  // Rete classic connection fields are spread into this component.
  // We use sourceOutput (queue name) to look up live stats in a store.
  // svelte-ignore unused-export-let
  export let sourceOutput: string | undefined;
  // svelte-ignore unused-export-let
  export let targetInput: string | undefined;
  // svelte-ignore unused-export-let
  export let start: Position;
  // svelte-ignore unused-export-let
  export let end: Position;
  export let path: string;
  export let id: string = '';
  export let __route: Position[][] = [];
  $: routedPath = __route.length
    ? __route.map(points => points.map((point, index) => `${index ? 'L' : 'M'} ${point.x} ${point.y}`).join(' ')).join(' ')
    : path;

  let hovered = false;

  $: queueNameRaw = sourceOutput ?? targetInput ?? '';
  $: queueName = String(queueNameRaw).trim();
  $: q =
    queueName
      ? $queueStatsByName.get(queueName) ||
        (queueName[0] === '@' ? $queueStatsByName.get(queueName.slice(1)) : $queueStatsByName.get(`@${queueName}`))
      : null;
  $: pct = q && q.capacity > 0 ? Math.max(0, Math.min(1, q.occupied / q.capacity)) : null;
  $: pps = q && typeof q.pps === 'number' && Number.isFinite(q.pps) ? Math.max(0, q.pps) : 0;
  $: fiq = q && q.frames_in_queue && typeof q.frames_in_queue === 'object' ? q.frames_in_queue : null;
  $: fiqAvg =
    fiq && typeof fiq.avg === 'number' && Number.isFinite(fiq.avg) ? fiq.avg.toFixed(2) : fiq && fiq.avg != null ? String(fiq.avg) : '';
  $: fiqLine =
    fiq && fiq.cur != null && fiq.min != null && fiq.max != null
      ? `frames_in_queue: cur=${fiq.cur}, min=${fiq.min}, avg=${fiqAvg}, max=${fiq.max}`
      : '';
  $: label =
    queueName && q && q.capacity > 0
      ? `${queueName}: ${q.occupied}/${q.capacity} (${Math.round((q.occupied / q.capacity) * 100)}%)${
          pps > 0 ? `, ${pps.toFixed(1)} items/s${q?.aggregate ? ' total' : ''}` : ''
        }${fiqLine ? `\n${fiqLine}` : ''}${q?.aggregate ? `\n${q.members.join('\n')}` : ''}`
      : queueName
        ? queueName
        : '';
  $: hoverText =
    pct !== null
      ? `${fiq && fiq.min != null && fiq.max != null ? `min=${fiq.min} avg=${fiqAvg} max=${fiq.max}` : ''}${
          pps > 0 ? ` · ${pps.toFixed(1)} items/s${q?.aggregate ? ' total' : ''}` : ''
        }`
      : '';

  // Hover badge layout (bar + one text line)
  const hoverBadgeW = 240;
  const hoverBadgeH = 40;
  const barW = 200;
  const barH = 8;
  $: barX = -barW / 2;
  $: barY = -14;
  $: barFillW = pct !== null ? Math.round(barW * pct) : 0;
  $: barFillColor = stroke;
  $: barText =
    q && q.capacity > 0
      ? `${q.occupied}/${q.capacity} (${Math.round((q.occupied / q.capacity) * 100)}%)`
      : q
        ? `${q.occupied ?? ''}/?`
        : '';
  $: hoverBadgeY = -hoverBadgeH / 2;

  $: flow = q?.flowSummary || summarizeQueueFlow(q ? [q] : []);
  $: stroke = {flowing: '#60a5fa', backlog: '#fbbf24', dropped: '#f87171', idle: '#64748b', unknown: '#475569'}[flow.state];
  $: flowText = `${flow.active}/${flow.total} active · ${flow.growing} accumulating · ${flow.held} not draining · ${flow.dropped} new drops · ${flow.total - flow.known} unknown`;
  $: detail = `${label}${q ? `\n${flowText}\nEnqueue ${Number(q.enq_pps || 0).toFixed(1)} / dequeue ${pps.toFixed(1)} items/s\nFullest queue ${Math.round(flow.maxFill * 100)}%` : '\nNo fresh queue samples'}`;
  // Reuse the destination segment; never measure paths during telemetry updates.
  $: lastRoute = __route[__route.length - 1] || [start, end];
  $: tip = lastRoute[lastRoute.length - 1] || end;
  $: approach = lastRoute[lastRoute.length - 2] || start;
  $: angle = Math.atan2(tip.y - approach.y, tip.x - approach.x) * 180 / Math.PI;
  $: width = hovered ? 5 : 2.5;
  $: mid = {
    x: (start.x + end.x) / 2,
    y: (start.y + end.y) / 2
  };
</script>

<svg data-testid="connection" data-flow-state={flow.state}>
  <defs><marker id={`arrow-${id}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" style={`fill: ${stroke}; pointer-events: none`} />
  </marker></defs>
  <!-- svelte-ignore a11y-no-noninteractive-element-interactions -->
  <!-- svelte-ignore a11y-mouse-events-have-key-events -->
  <!-- svelte-ignore a11y-no-noninteractive-element-to-interactive-role -->
  <!-- svelte-ignore a11y-no-static-element-interactions -->
  <path
    d={routedPath}
    stroke={stroke}
    stroke-width={width}
    stroke-linejoin="round"
    marker-end={`url(#arrow-${id})`}
    on:mouseenter={() => (hovered = true)}
    on:mouseleave={() => (hovered = false)}
  >
    {#if label}
      <title>{detail}</title>
    {/if}
  </path>

  {#if q}
    <g class="fill-marker" transform={`translate(${tip.x}, ${tip.y}) rotate(${angle})`} aria-hidden="true">
      <rect x="-26" y="-3" width="16" height="6" rx="2" fill="#0f172a" stroke={stroke} stroke-width="1" />
      <rect x="-25" y="-2" width={14 * flow.maxFill} height="4" rx="1" fill={stroke} />
    </g>
  {/if}

  {#if hovered && pct !== null}
    <g transform={`translate(${mid.x}, ${mid.y})`}>
      <rect x={-hoverBadgeW / 2} y={hoverBadgeY} width={hoverBadgeW} height={hoverBadgeH} rx="7" ry="7" class="badge-bg" />

      <!-- current fill as bar (first line) -->
      <rect x={barX} y={barY} width={barW} height={barH} rx="4" ry="4" class="badge-bar-bg" />
      <rect x={barX} y={barY} width={barFillW} height={barH} rx="4" ry="4" style={`fill: ${barFillColor};`} />
      <text text-anchor="middle" y={barY + barH + 10} class="badge-bar-text">{barText}</text>

      <!-- stats text (second line) -->
      <text text-anchor="middle" y="14" class="badge-text">{hoverText}</text>
    </g>
  {/if}
</svg>

<style>
  /*! keep same behavior as classic preset */
  svg {
    overflow: visible !important;
    position: absolute;
    pointer-events: none;
    width: 9999px;
    height: 9999px;
  }

  svg path {
    fill: none;
    pointer-events: auto;
    cursor: default;
  }

  .fill-marker { pointer-events: none; }

  .badge-bg {
    fill: rgba(2, 6, 23, 0.9);
    stroke: rgba(229, 231, 235, 0.25);
    stroke-width: 1px;
  }

  .badge-text {
    font-size: 14px;
    fill: #e5e7eb;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New',
      monospace;
    text-shadow: 0 0 2px #000;
  }

  .badge-bar-bg {
    fill: rgba(229, 231, 235, 0.14);
    stroke: rgba(229, 231, 235, 0.18);
    stroke-width: 1px;
  }

  .badge-bar-text {
    font-size: 12px;
    fill: #e5e7eb;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New',
      monospace;
    text-shadow: 0 0 2px #000;
  }
</style>

