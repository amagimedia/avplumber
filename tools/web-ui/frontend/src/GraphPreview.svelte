<script>
  import { createEventDispatcher, onDestroy, onMount } from 'svelte';

  import { NodeEditor, ClassicPreset } from 'rete';
  import { AreaPlugin, AreaExtensions } from 'rete-area-plugin';
  import { AutoArrangePlugin } from 'rete-auto-arrange-plugin';
  import { SveltePlugin, Presets as SveltePresets } from 'rete-svelte-plugin/svelte';
  import GraphConnection from './GraphConnection.svelte';
  import GraphNode from './GraphNode.svelte';
  import { sourceQueues as getSrcQueues, destinationQueues as getDstQueues } from './graphGroups.mjs';
  import { queueStatsByName } from './graphStores';

  export let nodes = [];
  export let queues = [];
  export let selectedNodeName = '';
  export let liveQueueStats = true;
  export let groupedLayout = false;
  export let focusedLayout = false;
  export let minZoom = 0;
  let rebuilding = false;
  let rebuildRequested = false;

  const dispatch = createEventDispatcher();

  /** @type {HTMLElement | null} */
  let container = null;

  /** @type {any} */
  let editor;
  /** @type {any} */
  let area;
  /** @type {any} */
  let arrange;

  let error = '';
  let resizeObserver;
  let compactFocus = false;
  let verticalFlow = false;
  let lastGraphKey = '';
  let lastPublishedQueueStatsMode = '';

  const nodeByName = new Map(); // name -> ClassicPreset.Node
  const nodeNameById = new Map(); // nodeId -> avplumber node name

  const socket = new ClassicPreset.Socket('queue');

  /** @type {Map<any, { el: HTMLElement, handler: any }>} */
  const nodeDomHandlers = new Map();

  function clearNodeDomHandlers() {
    for (const { el, handler } of nodeDomHandlers.values()) {
      try {
        el.removeEventListener('click', handler);
      } catch (_) {
        // ignore
      }
    }
    nodeDomHandlers.clear();
  }

  function syncSelectedNodeHighlight() {
    if (!editor || !area) return;
    for (const n of editor.getNodes()) {
      const view = area.nodeViews && area.nodeViews.get ? area.nodeViews.get(n.id) : null;
      const el = view && view.element ? view.element : null;
      if (!el) continue;
      const avName = nodeNameById.get(n.id) || '';
      const selected = avName && selectedNodeName && avName === selectedNodeName;
      if (selected) el.classList.add('avp-selected');
      else el.classList.remove('avp-selected');
    }
  }

  async function attachNodeClickHandlers() {
    if (!editor || !area) return;
    // Wait until DOM views exist
    await nextFrame();
    clearNodeDomHandlers();

    for (const n of editor.getNodes()) {
      const view = area.nodeViews && area.nodeViews.get ? area.nodeViews.get(n.id) : null;
      const el = view && view.element ? view.element : null;
      if (!el) continue;
      const handler = (ev) => {
        ev.stopPropagation();
        const avName = nodeNameById.get(n.id);
        if (avName) dispatch('selectNode', { name: avName });
      };
      el.addEventListener('click', handler);
      nodeDomHandlers.set(n.id, { el, handler });
    }
    syncSelectedNodeHighlight();
  }

  function nextFrame() {
    return new Promise((resolve) => requestAnimationFrame(() => resolve()));
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

  const socketPositions = new Map();

  async function measureAndApplyNodeSizes() {
    await nextFrame();
    await nextFrame();
    if (!area || !editor) return;
    socketPositions.clear();
    const zoom = area.area.transform.k || 1;
    for (const node of editor.getNodes()) {
      const el = area.nodeViews.get(node.id)?.element.querySelector('[data-testid="node"]');
      if (!el) throw new Error('Node did not render: ' + node.label);
      // Layout uses unscaled sizes; client rect dimensions include the old
      // view's zoom and caused overlapping nodes after opening a group.
      node.width = el.offsetWidth;
      node.height = el.offsetHeight;
      const bounds = el.getBoundingClientRect();
      for (const side of ['input', 'output']) {
        for (const key of Object.keys(node[side === 'input' ? 'inputs' : 'outputs'])) {
          const row = [...el.querySelectorAll('[data-testid]')]
            .find(item => item.getAttribute('data-testid') === `${side}-${key}`);
          const socket = row?.querySelector(`[data-testid="${side}-socket"]`);
          if (!socket) throw new Error('Socket did not render: ' + key);
          const rect = socket.getBoundingClientRect();
          socketPositions.set(`${node.id}:${side}:${key}`, {
            x: ((rect.left + rect.right) / 2 - bounds.left) / zoom,
            y: ((rect.top + rect.bottom) / 2 - bounds.top) / zoom,
            width: 0, height: 0, side: verticalFlow
              ? (side === 'input' ? 'NORTH' : 'SOUTH') : (side === 'input' ? 'WEST' : 'EAST')
          });
          row.title = key;
        }
      }
      await area.resize(node.id, node.width, node.height);
    }
  }

  async function layoutGraph() {
    const { result } = await arrange.layout({ options: {
      'elk.algorithm': 'layered', 'elk.direction': verticalFlow ? 'DOWN' : 'RIGHT',
      'elk.edgeRouting': 'ORTHOGONAL',
      'elk.layered.spacing.nodeNodeBetweenLayers': focusedLayout ? '44' : '90',
      'elk.spacing.nodeNode': '44',
      'elk.spacing.edgeNode': '24',
      'elk.layered.spacing.edgeNodeBetweenLayers': '24',
      'elk.spacing.edgeEdge': '12',
      'elk.layered.spacing.edgeEdgeBetweenLayers': '12',
      'elk.layered.considerModelOrder.strategy': 'NODES_AND_EDGES',
      'elk.padding': '[top=24,left=24,bottom=24,right=24]'
    } });
    if (!area || !editor) return;
    const connections = new Map(editor.getConnections().map(connection => [connection.id, connection]));
    for (const edge of result.edges || []) {
      const connection = connections.get(edge.id);
      if (!connection) continue;
      connection.__route = (edge.sections || []).map(section =>
        [section.startPoint, ...(section.bendPoints || []), section.endPoint]);
      await area.update('connection', connection.id);
    }
    await fitGraph();
  }

  async function fitGraph() {
    if (!area || !editor || !editor.getNodes().length) return;
    await AreaExtensions.zoomAt(area, editor.getNodes());
    if (!area) return;
    if (minZoom && area.area.transform.k < minZoom) {
      await area.area.zoom(minZoom, 0, 0);
      await area.area.translate(24, 24);
    }
  }

  async function rebuildGraph() {
    rebuildRequested = true;
    if (rebuilding || !editor || !area) return;
    rebuilding = true;
    try {
      while (rebuildRequested && editor && area) {
        rebuildRequested = false;
        await rebuildGraphOnce(nodes.slice());
      }
    } catch (e) {
      if (area) error = String(e?.message || e);
    } finally { rebuilding = false; }
  }

  async function rebuildGraphOnce(graphNodes) {
    error = '';
    compactFocus = focusedLayout && container.clientWidth >= 1400;
    verticalFlow = focusedLayout && !compactFocus;

    nodeByName.clear();
    nodeNameById.clear();
    clearNodeDomHandlers();

    await editor.clear();

    // 1) Create nodes with ports (queue names are port keys)
    for (const n of graphNodes) {
      if (!n || typeof n.name !== 'string') continue;
      const p = n.params || {};
      const srcQs = getSrcQueues(p);
      const dstQs = getDstQueues(p);

      const label = `${n.label || n.name}\n${n.type || ''}${n.working ? '' : ' (OFF)'}`.trim();
      const node = new ClassicPreset.Node(label);
      node.width = compactFocus ? 180 : 300;
      node.__vertical = verticalFlow;

      for (const qName of srcQs) {
        node.addInput(qName, new ClassicPreset.Input(socket, qName, true));
      }
      for (const qName of dstQs) {
        node.addOutput(qName, new ClassicPreset.Output(socket, qName, true));
      }

      await editor.addNode(node);
      nodeByName.set(n.name, node);
      nodeNameById.set(node.id, n.name);
    }

    // 2) Create connections based on shared queue names: producer(dst) -> consumer(src)
    const producers = new Map(); // queueName -> nodeName
    const consumers = new Map(); // queueName -> nodeName[]

    for (const n of graphNodes) {
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
    await layoutGraph();

    // 4) Enable node selection by clicking node DOM
    await attachNodeClickHandlers();
  }

  function publishQueueStats() {
    if (liveQueueStats) {
      lastPublishedQueueStatsMode = 'live';
      queueStatsByName.set(indexQueues(queues));
      return;
    }

    if (lastPublishedQueueStatsMode !== 'off') {
      lastPublishedQueueStatsMode = 'off';
      queueStatsByName.set(new Map());
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
            connection: () => GraphConnection,
            node: () => GraphNode
          }
        })
      );
      arrange.addPreset(() => ({
        port: ({ nodeId, side, key }) => socketPositions.get(`${nodeId}:${side}:${key}`)
      }));

      // Attach plugins
      editor.use(area);
      area.use(render);
      area.use(arrange);
      // Read-only graph: preserve the routed layout while allowing pan/zoom.
      area.addPipe(context => context.type === 'nodetranslate' && !rebuilding ? undefined : context);

      await rebuildGraph();
      if (!area || !container) return;
      resizeObserver = new ResizeObserver(() => {
        if (rebuilding) return;
        if ((focusedLayout && container.clientWidth >= 1400) !== compactFocus) rebuildGraph();
        else fitGraph().catch(() => {});
      });
      resizeObserver.observe(container);
    } catch (e) {
      error = String(e && e.message ? e.message : e);
    }
  });

  onDestroy(() => {
    resizeObserver?.disconnect();
    try {
      if (area) area.destroy();
    } catch (_) {
      // ignore
    }
    clearNodeDomHandlers();
    editor = null;
    area = null;
    arrange = null;
    queueStatsByName.set(new Map());
  });

  // Rebuild when topology changes (nodes/src/dst changes)
  $: {
    const key = `${groupedLayout}|${focusedLayout}|${minZoom}|${buildGraphKey(nodes)}`;
    if (key !== lastGraphKey) {
      lastGraphKey = key;
      rebuildGraph();
    }
  }

  // Keep highlight in sync when selection changes from outside (nodes list)
  $: {
    selectedNodeName;
    syncSelectedNodeHighlight();
  }

  // Always publish queue stats to the store (even before Rete `area` is ready),
  // so the custom connection component can render reactively.
  $: {
    queues;
    liveQueueStats;
    publishQueueStats();
  }
</script>

<div class="rete-wrap">
  {#if error}
    <div class="rete-error">Graph preview error: {error}</div>
  {/if}
  {#if rebuilding}<div class="graph-loading">Arranging graph…</div>{/if}
  <!-- svelte-ignore a11y-click-events-have-key-events -->
  <!-- svelte-ignore a11y-no-static-element-interactions -->
  <div class="rete-container" style:visibility={rebuilding ? "hidden" : "visible"} bind:this={container} on:click={() => dispatch('selectNode', { name: '' })} />
</div>

<style>
  .rete-wrap {
    position: relative;
    flex: 1;
    min-height: 280px;
    border: 1px solid #111827;
    border-radius: 0.25rem;
    overflow: hidden;
    background: #020617;
    display: flex;
    flex-direction: column;
  }

  .rete-container {
    flex: 1;
    min-height: 0;
    width: 100%;
  }

  .graph-loading { position:absolute; padding:16px; color:#94a3b8; }

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

  /* selected node highlight (set by GraphPreview via DOM class) */
  .rete-wrap :global(.avp-selected) {
    outline: 2px solid rgba(59, 130, 246, 0.95);
    outline-offset: 1px;
    box-shadow: 0 0 0 2px rgba(2, 6, 23, 0.75);
  }
</style>
