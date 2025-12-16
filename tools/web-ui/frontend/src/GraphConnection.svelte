<script lang="ts">
  import type { Position } from 'rete-area-plugin';
  import { queueStatsByName } from './graphStores';

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

  let hovered = false;

  $: queueNameRaw = sourceOutput ?? targetInput ?? '';
  $: queueName = String(queueNameRaw).trim();
  $: firstKey = $queueStatsByName && $queueStatsByName.size ? Array.from($queueStatsByName.keys())[0] : '';
  $: q =
    queueName
      ? $queueStatsByName.get(queueName) ||
        (queueName[0] === '@' ? $queueStatsByName.get(queueName.slice(1)) : $queueStatsByName.get(`@${queueName}`))
      : null;
  $: pct = q && q.capacity > 0 ? Math.max(0, Math.min(1, q.occupied / q.capacity)) : null;
  $: pps = q && typeof q.pps === 'number' && Number.isFinite(q.pps) ? Math.max(0, q.pps) : 0;
  $: label =
    queueName && q && q.capacity > 0
      ? `${queueName}: ${q.occupied}/${q.capacity} (${Math.round((q.occupied / q.capacity) * 100)}%)${
          pps > 0 ? `, ${pps.toFixed(1)} pps` : ''
        }`
      : queueName
        ? `${queueName} (no stats; queues[0]=${JSON.stringify(firstKey)})`
        : '';
  $: flowOn = pps > 0;
  // Map packets/sec to animation duration (higher pps => faster movement).
  // Tuned so ~60 pps => ~1s period, clamped for usability.
  $: flowDuration = flowOn ? Math.max(0.25, Math.min(6, 60 / Math.max(pps, 0.1))) : 6;
  $: flowOpacity = flowOn ? Math.max(0.45, Math.min(0.95, 0.45 + pps / 200)) : 0;

  const HEX2 = Array.from({ length: 256 }, (_, i) => i.toString(16).padStart(2, '0'));
  const clamp255 = (v: number) => (v < 0 ? 0 : v > 255 ? 255 : v);
  const rgbToHex = (r: number, g: number, b: number) =>
    `#${HEX2[clamp255(Math.round(r))]}${HEX2[clamp255(Math.round(g))]}${HEX2[clamp255(Math.round(b))]}`;

  function colorFor(p: number) {
    // blue (empty) -> green -> orange -> red, with fuzzy/smooth transitions.
    // p is expected to be queue occupancy ratio in [0..1], but we clamp defensively.
    if (!Number.isFinite(p)) return '#60a5fa';
    const x = Math.max(0, Math.min(1, p));

    const clamp01 = (t: number) => Math.max(0, Math.min(1, t));
    const smoothstep = (edge0: number, edge1: number, v: number) => {
      const t = clamp01((v - edge0) / (edge1 - edge0));
      return t * t * (3 - 2 * t);
    };

    // Numeric RGB blending (avoid hex string parsing on every call).
    const blue = { r: 0x3b, g: 0x82, b: 0xf6 };
    const green = { r: 0x22, g: 0xc5, b: 0x5e };
    const orange = { r: 0xf9, g: 0x73, b: 0x16 };
    const red = { r: 0xdc, g: 0x26, b: 0x26 };

    const mixInto = (cur: { r: number; g: number; b: number }, to: { r: number; g: number; b: number }, t: number) => {
      const k = clamp01(t);
      cur.r = cur.r + (to.r - cur.r) * k;
      cur.g = cur.g + (to.g - cur.g) * k;
      cur.b = cur.b + (to.b - cur.b) * k;
    };

    // Transition bands (overlapping ranges => "fuzzy" thresholds).
    const tBG = smoothstep(0.0, 0.08, x); // empty/nearly-empty => blue
    const tGO = smoothstep(0.45, 0.62, x);
    const tOR = smoothstep(0.76, 0.9, x);

    const c = { r: blue.r, g: blue.g, b: blue.b };
    mixInto(c, green, tBG);
    mixInto(c, orange, tGO);
    mixInto(c, red, tOR);
    return rgbToHex(c.r, c.g, c.b);
  }

  $: stroke = pct === null ? (queueName ? '#a78bfa' : '#60a5fa') : colorFor(pct);
  $: width = hovered ? 7 : 5;
  $: mid = {
    x: (start.x + end.x) / 2,
    y: (start.y + end.y) / 2
  };
</script>

<svg data-testid="connection">
  <!-- svelte-ignore a11y-no-noninteractive-element-interactions -->
  <!-- svelte-ignore a11y-mouse-events-have-key-events -->
  <path
    d={path}
    stroke={stroke}
    stroke-width={width}
    on:mouseenter={() => (hovered = true)}
    on:mouseleave={() => (hovered = false)}
  >
    {#if label}
      <title>{label}</title>
    {/if}
  </path>

  {#if flowOn}
    <!-- Animated "dots" overlay: cheap dash animation along the same path -->
    <path
      d={path}
      class="flow"
      style={`--flow-duration: ${flowDuration}s; --flow-opacity: ${flowOpacity};`}
    >
      <!-- Use SVG SMIL animation (more reliable than CSS animations for SVG paths) -->
      <animate
        attributeName="stroke-dashoffset"
        from="0"
        to="-240"
        dur={`${flowDuration}s`}
        repeatCount="indefinite"
      />
    </path>
  {/if}

  {#if hovered && pct !== null}
    <g transform={`translate(${mid.x}, ${mid.y})`}>
      <rect x="-54" y="-10" width="108" height="20" rx="6" ry="6" class="badge-bg" />
      <text text-anchor="middle" dominant-baseline="middle" class="badge-text">
        {Math.round(pct * 100)}% {#if pps > 0}· {pps.toFixed(1)} pps{/if}
      </text>
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

  .flow {
    pointer-events: none;
    /*stroke: rgba(56, 189, 248, var(--flow-opacity));*/
    stroke: rgba(0, 0, 0, var(--flow-opacity));
    stroke-width: 4;
    stroke-linecap: round;
    stroke-dasharray: 6 18;
    stroke-dashoffset: 0;
    filter: drop-shadow(0 0 2px rgba(0, 0, 0, 0.8));
  }

  .badge-bg {
    fill: rgba(2, 6, 23, 0.9);
    stroke: rgba(229, 231, 235, 0.25);
    stroke-width: 1px;
  }

  .badge-text {
    font-size: 11px;
    fill: #e5e7eb;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New',
      monospace;
    text-shadow: 0 0 2px #000;
  }
</style>


