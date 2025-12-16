<script>
  import { onDestroy, onMount } from 'svelte';

  import { NodeEditor, ClassicPreset } from 'rete';
  import { AreaPlugin, AreaExtensions } from 'rete-area-plugin';
  import { AutoArrangePlugin, Presets as ArrangePresets } from 'rete-auto-arrange-plugin';
  import { SveltePlugin, Presets as SveltePresets } from 'rete-svelte-plugin/svelte';
  import GraphConnection from './GraphConnection.svelte';
  import { queueStatsByName } from './graphStores';

  export let nodes = [];
  export let queues = [];

  /** @type {HTMLElement | null} */
  let container = null;

  /** @type {any} */
  let editor;
  /** @type {any} */
  let area;
  /** @type {any} */
  let arrange;

  let error = '';
  let lastGraphKey = '';
  let allowProgrammaticTranslate = false;

  // Maps used for efficient queue-fill updates (no rebuild).
  const nodeByName = new Map(); // name -> ClassicPreset.Node
  const outputsByQueue = new Map(); // queueName -> ClassicPreset.Output
  const nodeIdByQueue = new Map(); // queueName -> nodeId (owner node to refresh)

  const socket = new ClassicPreset.Socket('queue');

  function nextFrame() {
    return new Promise((resolve) => requestAnimationFrame(() => resolve()));
  }

  function normalizeQueueRef(v) {
    if (!v) return [];
    if (typeof v === 'string') return [v];
    if (Array.isArray(v)) return v.filter((x) => typeof x === 'string');
    // Some nodes may provide object syntax; ignore for now.
    return [];
  }

  function getSrcQueues(params) {
    const p = params || {};
    return normalizeQueueRef(p.src);
  }

  function getDstQueues(params) {
    const p = params || {};
    const out = new Set(normalizeQueueRef(p.dst));

    // demux commonly uses `routing` instead of `dst`
    if (p.routing && typeof p.routing === 'object' && !Array.isArray(p.routing)) {
      for (const v of Object.values(p.routing)) {
        if (typeof v === 'string' && v) out.add(v);
      }
    }
    return Array.from(out);
  }

  function guessNodeSize(node) {
    // fallback if DOM measurement isn't ready yet
    const baseW = 260;
    const baseH = 56;
    const inCount = node?.inputs ? Object.keys(node.inputs).length : 0;
    const outCount = node?.outputs ? Object.keys(node.outputs).length : 0;
    const ports = Math.max(inCount, outCount);
    return { width: baseW, height: baseH + ports * 18 };
  }

  function queueFillLabel(queueName, q) {
    if (!q || !q.capacity || q.capacity <= 0) return queueName;
    const pct = Math.max(0, Math.min(100, (q.occupied / q.capacity) * 100));
    return `${queueName} (${q.occupied}/${q.capacity} ${Math.round(pct)}%)`;
  }

  function buildGraphKey(nodesArr) {
    if (!Array.isArray(nodesArr)) return '';
    // stable enough for our needs: node + src/dst topology
    return nodesArr
      .map((n) => {
        const p = (n && n.params) || {};
        const src = getSrcQueues(p).slice().sort().join(',');
        const dst = getDstQueues(p).slice().sort().join(',');
        return `${n.name}|${n.type}|${src}|${dst}`;
      })
      .sort()
      .join(';;');
  }

  function indexQueues(queuesArr) {
    const map = new Map();
    if (!Array.isArray(queuesArr)) return map;
    for (const q of queuesArr) {
      if (q && typeof q.name === 'string') {
        map.set(q.name, q);
      }
    }
    return map;
  }

  async function measureAndApplyNodeSizes() {
    if (!area || !editor) return;

    // Wait for render to mount node elements
    await nextFrame();
    await nextFrame();

    for (const node of editor.getNodes()) {
      const view = area.nodeViews && area.nodeViews.get ? area.nodeViews.get(node.id) : null;
      const el = view && view.element ? view.element : null;
      if (!el || typeof el.getBoundingClientRect !== 'function') {
        const fallback = guessNodeSize(node);
        // eslint-disable-next-line no-param-reassign
        node.width = fallback.width;
        // eslint-disable-next-line no-param-reassign
        node.height = fallback.height;
        continue;
      }
      const rect = el.getBoundingClientRect();
      const w = Math.max(80, Math.ceil(rect.width));
      const h = Math.max(40, Math.ceil(rect.height));
      // Auto-arrange plugin and our own layout both rely on these fields.
      // eslint-disable-next-line no-param-reassign
      node.width = w;
      // eslint-disable-next-line no-param-reassign
      node.height = h;

      // Keep area plugin in sync (helps when zooming/selection uses view bounds)
      area.resize(node.id, w, h).catch(() => {});
    }
  }

  async function layoutDependencyOrder() {
    if (!area || !editor) return;

    const nodesList = editor.getNodes();
    const conns = editor.getConnections();

    const byId = new Map(nodesList.map((n) => [n.id, n]));
    const adj = new Map(); // id -> Set<id>
    const indeg = new Map(); // id -> number

    for (const n of nodesList) {
      adj.set(n.id, new Set());
      indeg.set(n.id, 0);
    }

    for (const c of conns) {
      const s = c.source;
      const t = c.target;
      if (!byId.has(s) || !byId.has(t)) continue;
      const set = adj.get(s);
      if (!set.has(t)) {
        set.add(t);
        indeg.set(t, (indeg.get(t) || 0) + 1);
      }
    }

    // Kahn topological order + longest-path level
    const level = new Map(); // id -> int
    const topoIndex = new Map(); // id -> int
    const q = [];

    for (const n of nodesList) {
      level.set(n.id, 0);
      if ((indeg.get(n.id) || 0) === 0) q.push(n.id);
    }
    // stable ordering for equal indegree
    q.sort((a, b) => String(byId.get(a)?.label || a).localeCompare(String(byId.get(b)?.label || b)));

    let idx = 0;
    while (q.length) {
      const id = q.shift();
      topoIndex.set(id, idx++);
      const base = level.get(id) || 0;
      for (const nb of adj.get(id) || []) {
        level.set(nb, Math.max(level.get(nb) || 0, base + 1));
        indeg.set(nb, (indeg.get(nb) || 0) - 1);
        if ((indeg.get(nb) || 0) === 0) {
          q.push(nb);
          q.sort((a, b) => String(byId.get(a)?.label || a).localeCompare(String(byId.get(b)?.label || b)));
        }
      }
    }

    // If cycle (shouldn't happen), fall back to label order
    for (const n of nodesList) {
      if (!topoIndex.has(n.id)) {
        topoIndex.set(n.id, idx++);
      }
    }

    // Crossing-minimizing ordering within each level (barycenter sweeps)
    const groups = new Map(); // level -> node[]
    let maxLevel = 0;
    for (const n of nodesList) {
      const l = level.get(n.id) || 0;
      maxLevel = Math.max(maxLevel, l);
      if (!groups.has(l)) groups.set(l, []);
      groups.get(l).push(n);
    }

    // init order by topo index
    for (const arr of groups.values()) arr.sort((a, b) => (topoIndex.get(a.id) || 0) - (topoIndex.get(b.id) || 0));

    const preds = new Map(); // id -> Set<id>
    const succs = new Map(); // id -> Set<id>
    for (const n of nodesList) {
      preds.set(n.id, new Set());
      succs.set(n.id, new Set());
    }
    for (const c of conns) {
      const s = c.source;
      const t = c.target;
      if (preds.has(t)) preds.get(t).add(s);
      if (succs.has(s)) succs.get(s).add(t);
    }

    function indexMapForLevel(l) {
      const arr = groups.get(l) || [];
      const m = new Map();
      arr.forEach((n, i) => m.set(n.id, i));
      return m;
    }

    function barycenter(nodeId, neighborIndexMap, neighborIds) {
      let sum = 0;
      let cnt = 0;
      for (const nb of neighborIds) {
        const v = neighborIndexMap.get(nb);
        if (typeof v === 'number') {
          sum += v;
          cnt++;
        }
      }
      return cnt ? sum / cnt : null;
    }

    // A few sweeps are usually enough for small/medium graphs
    for (let sweep = 0; sweep < 4; sweep++) {
      // forward sweep (use predecessors in previous layer)
      for (let l = 1; l <= maxLevel; l++) {
        const prevIdx = indexMapForLevel(l - 1);
        const arr = groups.get(l) || [];
        arr.sort((a, b) => {
          const ab = barycenter(a.id, prevIdx, preds.get(a.id) || []);
          const bb = barycenter(b.id, prevIdx, preds.get(b.id) || []);
          if (ab === null && bb === null) return (topoIndex.get(a.id) || 0) - (topoIndex.get(b.id) || 0);
          if (ab === null) return 1;
          if (bb === null) return -1;
          if (ab !== bb) return ab - bb;
          return (topoIndex.get(a.id) || 0) - (topoIndex.get(b.id) || 0);
        });
      }
      // backward sweep (use successors in next layer)
      for (let l = maxLevel - 1; l >= 0; l--) {
        const nextIdx = indexMapForLevel(l + 1);
        const arr = groups.get(l) || [];
        arr.sort((a, b) => {
          const ab = barycenter(a.id, nextIdx, succs.get(a.id) || []);
          const bb = barycenter(b.id, nextIdx, succs.get(b.id) || []);
          if (ab === null && bb === null) return (topoIndex.get(a.id) || 0) - (topoIndex.get(b.id) || 0);
          if (ab === null) return 1;
          if (bb === null) return -1;
          if (ab !== bb) return ab - bb;
          return (topoIndex.get(a.id) || 0) - (topoIndex.get(b.id) || 0);
        });
      }
    }

    const xGap = 120;
    const yGap = 70;

    // Compute max width per level to avoid overlaps horizontally
    const levelWidth = new Map();
    for (let l = 0; l <= maxLevel; l++) {
      const arr = groups.get(l) || [];
      let mw = 0;
      for (const n of arr) mw = Math.max(mw, Number(n.width) || guessNodeSize(n).width);
      levelWidth.set(l, mw);
    }
    const xOffset = [];
    let accX = 0;
    for (let l = 0; l <= maxLevel; l++) {
      xOffset[l] = accX;
      accX += (levelWidth.get(l) || 260) + xGap;
    }

    // Apply positions
    for (let l = 0; l <= maxLevel; l++) {
      const arr = groups.get(l) || [];
      let y = 0;
      for (const n of arr) {
        const w = Number(n.width) || guessNodeSize(n).width;
        const h = Number(n.height) || guessNodeSize(n).height;
        allowProgrammaticTranslate = true;
        await area.translate(n.id, { x: xOffset[l], y });
        allowProgrammaticTranslate = false;
        y += h + yGap;
      }
    }

    try {
      await AreaExtensions.zoomAt(area, nodesList);
    } catch (_) {
      // ignore
    }
  }

  async function rebuildGraph() {
    if (!editor || !area) return;

    nodeByName.clear();
    outputsByQueue.clear();
    nodeIdByQueue.clear();

    await editor.clear();

    const queueStats = indexQueues(queues);

    // 1) Create nodes with ports (queue names are port keys)
    for (const n of nodes || []) {
      if (!n || typeof n.name !== 'string') continue;
      const p = n.params || {};
      const srcQs = getSrcQueues(p);
      const dstQs = getDstQueues(p);

      const label = `${n.name}\n${n.type || ''}${n.working ? '' : ' (OFF)'}`.trim();
      const node = new ClassicPreset.Node(label);

      for (const qName of srcQs) {
        node.addInput(qName, new ClassicPreset.Input(socket, qName, true));
      }
      for (const qName of dstQs) {
        const q = queueStats.get(qName);
        const out = new ClassicPreset.Output(socket, queueFillLabel(qName, q), true);
        node.addOutput(qName, out);
        outputsByQueue.set(qName, out);
        nodeIdByQueue.set(qName, node.id);
      }

      await editor.addNode(node);
      nodeByName.set(n.name, node);
    }

    // 2) Create connections based on shared queue names: producer(dst) -> consumer(src)
    const producers = new Map(); // queueName -> nodeName
    const consumers = new Map(); // queueName -> nodeName[]

    for (const n of nodes || []) {
      if (!n || typeof n.name !== 'string') continue;
      const p = n.params || {};
      for (const qName of getDstQueues(p)) {
        // If multiple producers exist, keep the first to avoid throwing.
        if (!producers.has(qName)) producers.set(qName, n.name);
      }
      for (const qName of getSrcQueues(p)) {
        if (!consumers.has(qName)) consumers.set(qName, []);
        consumers.get(qName).push(n.name);
      }
    }

    for (const [qName, srcNodeName] of producers.entries()) {
      const srcNode = nodeByName.get(srcNodeName);
      if (!srcNode || !srcNode.outputs[qName]) continue;
      const targets = consumers.get(qName) || [];
      for (const dstNodeName of targets) {
        const dstNode = nodeByName.get(dstNodeName);
        if (!dstNode || !dstNode.inputs[qName]) continue;
        try {
          const conn = new ClassicPreset.Connection(srcNode, qName, dstNode, qName);
          // attach queue metadata for rendering (read-only)
          // eslint-disable-next-line no-param-reassign
          conn.__queueName = qName;
          await editor.addConnection(conn);
        } catch (_) {
          // ignore duplicate/invalid connections
        }
      }
    }

    // 3) Measure and deterministic layout (dependency order)
    await measureAndApplyNodeSizes();
    await layoutDependencyOrder();
  }

  async function updateQueueFills() {
    if (!area) return;
    const queueStats = indexQueues(queues);

    for (const [qName, out] of outputsByQueue.entries()) {
      const q = queueStats.get(qName);
      out.label = queueFillLabel(qName, q);

      const nodeId = nodeIdByQueue.get(qName);
      if (nodeId) {
        // Re-render node to refresh port labels.
        // (Connection labels/colors could be added later via custom connection component.)
        area.update('node', nodeId).catch(() => {});
      }
    }

    // Update connection fill metadata and rerender connections (for hover highlight/labels)
    if (editor && area) {
      for (const c of editor.getConnections()) {
        const qName = c.__queueName || c.sourceOutput || c.targetInput;
        const q = qName ? queueStats.get(qName) : null;
        const pct = q && q.capacity > 0 ? q.occupied / q.capacity : null;
        const pps = q && typeof q.pps === 'number' && Number.isFinite(q.pps) ? q.pps : null;
        // eslint-disable-next-line no-param-reassign
        c.__queueName = qName;
        // eslint-disable-next-line no-param-reassign
        c.__fillPct = typeof pct === 'number' && Number.isFinite(pct) ? pct : null;
        // eslint-disable-next-line no-param-reassign
        c.__pps = pps;
        // eslint-disable-next-line no-param-reassign
        c.__fillText =
          qName && q && q.capacity > 0
            ? `${qName}: ${q.occupied}/${q.capacity} (${Math.round((q.occupied / q.capacity) * 100)}%)${
                pps !== null ? `, ${pps.toFixed(1)} pps` : ''
              }`
            : qName || '';
        area.update('connection', c.id).catch(() => {});
      }
    }
  }

  onMount(async () => {
    try {
      if (!container) return;

      // Core editor + plugins
      editor = new NodeEditor();
      area = new AreaPlugin(container);
      const render = new SveltePlugin();
      arrange = new AutoArrangePlugin();

      // Presets
      render.addPreset(
        SveltePresets.classic.setup({
          customize: {
            connection: () => GraphConnection
          }
        })
      );
      arrange.addPreset(ArrangePresets.classic.setup());

      // Attach plugins
      editor.use(area);
      area.use(render);
      area.use(arrange);

      // Make graph read-only: block node dragging/translating initiated by pointer interactions.
      // We still allow pan/zoom and programmatic translations used by our layout.
      area.addPipe((context) => {
        if (!context || typeof context !== 'object' || !('type' in context)) return context;
        if (context.type === 'nodetranslate' && !allowProgrammaticTranslate) {
          return undefined; // stop
        }
        return context;
      });

      await rebuildGraph();
    } catch (e) {
      error = String(e && e.message ? e.message : e);
    }
  });

  onDestroy(() => {
    try {
      if (area) area.destroy();
    } catch (_) {
      // ignore
    }
    editor = null;
    area = null;
    arrange = null;
  });

  // Rebuild when topology changes (nodes/src/dst changes)
  $: {
    const key = buildGraphKey(nodes);
    if (key !== lastGraphKey) {
      lastGraphKey = key;
      rebuildGraph();
    }
  }

  // Update labels when queue fill changes
  $: {
    // Trigger on any queues array assignment
    updateQueueFills();
  }

  // Always publish queue stats to the store (even before Rete `area` is ready),
  // so the custom connection component can render reactively.
  $: {
    queueStatsByName.set(indexQueues(queues));
  }
</script>

<div class="rete-wrap">
  {#if error}
    <div class="rete-error">Graph preview error: {error}</div>
  {/if}
  <div class="rete-container" bind:this={container} />
</div>

<style>
  .rete-wrap {
    position: relative;
    min-height: 280px;
    border: 1px solid #111827;
    border-radius: 0.25rem;
    overflow: hidden;
    background: #020617;
  }

  .rete-container {
    height: 420px;
    width: 100%;
  }

  .rete-error {
    position: absolute;
    z-index: 10;
    top: 0.25rem;
    left: 0.25rem;
    right: 0.25rem;
    padding: 0.25rem 0.4rem;
    background: rgba(127, 29, 29, 0.9);
    color: #fee2e2;
    font-size: 0.75rem;
    border-radius: 0.25rem;
  }
</style>


