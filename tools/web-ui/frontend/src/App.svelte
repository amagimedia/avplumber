<script>
  import { onDestroy, onMount } from 'svelte';
  import { instanceRestarted } from './instanceRestart.mjs';
  import { hoverTips } from './graph/hoverTips.mjs';
  import DockHost from './DockHost.svelte';
  import GraphPanel from './panels/GraphPanel.svelte';
  import SelectedPanel from './panels/SelectedPanel.svelte';
  import InstancePanel from './panels/InstancePanel.svelte';
  import NodeObjectsPanel from './panels/NodeObjectsPanel.svelte';
  import StatsPanel from './panels/StatsPanel.svelte';
  import LogsPanel from './panels/LogsPanel.svelte';
  import ConsolePanel from './panels/ConsolePanel.svelte';
  import InstanceSharedObjectsPanel from './panels/InstanceSharedObjectsPanel.svelte';
  import QueuesPage from './panels/QueuesPage.svelte';
  import NodesPage from './panels/NodesPage.svelte';

  // Behind a reverse proxy the page can live under a path prefix (e.g. /graph/); the API and the
  // WebSocket are then under the same prefix.
  const basePath = window.location.pathname.replace(/[^/]*$/, '');

  let ws;
  let wsConnected = false;

  let instances = [];
  let currentInstanceId = null;
  let requestedInstanceId = new URLSearchParams(window.location.search).get('instance');
  let instanceGeneration = 0;
  let requestedGeneration = -1;
  // The reload after a restart is answered as soon as the engine is ready, which can precede the
  // start of some nodes (the mixer starts its aux buses after setReady()); one more nodes.json,
  // this long after that reply, refreshes their ON/OFF state.
  const restartNodesRecheckMs = 5000;
  let restartNodesRecheckGeneration = -1;

  let nodes = [];
  let queues = [];
  let queuesObservedAt = 0;
  let queuesFresh = false;
  let queuesInstance = null;
  $: queueStatsFresh = wsConnected && queuesFresh && queuesInstance === currentInstanceId;
  let statsByInstance = {};
  let currentStatsPrettyText = '';
  let hasReceivedStats = false;
  let hasReceivedLog = false;
  let logs = [];
  let consoleLines = [];
  const queuesRefreshMs = 1000;
  let autoRefreshISOs = true;
  let syncGroups = [];
  let correctionGroups = [];

  /** @type {any} */
  let dockHost;

  // Pages: the graph with its dock of widgets, and the full Queues and Nodes tables (#queues, #nodes).
  const pages = ['graph', 'queues', 'nodes'];
  const pageFromHash = () => (pages.includes(window.location.hash.slice(1)) ? window.location.hash.slice(1) : 'graph');
  let page = pageFromHash();
  function showPage(p) {
    page = p;
    const url = new URL(window.location.href);
    url.hash = p === 'graph' ? '' : p;
    window.history.replaceState(null, '', url);
  }

  // Selected node or group, shared by the graph, the inspectors and the table pages; `queue` names
  // the queue that was followed to it, which the graph outlines.
  /** @type {{kind: 'node' | 'group', id: string, queue?: string} | null} */
  let selected = null;
  function setSelected(s) {
    selected = s && s.id ? { kind: s.kind === 'group' ? 'group' : 'node', id: String(s.id),
                             ...(typeof s.queue === 'string' ? { queue: s.queue } : {}) } : null;
  }
  function openInGraph(s) {
    setSelected(s);
    showPage('graph');
  }

  async function copyToClipboard(text) {
    const t = text == null ? '' : String(text);
    try {
      if (navigator && navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(t);
        return true;
      }
    } catch (_) {
      // fall back
    }
    try {
      const ta = document.createElement('textarea');
      ta.value = t;
      ta.style.position = 'fixed';
      ta.style.left = '-9999px';
      ta.style.top = '-9999px';
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      const ok = document.execCommand('copy');
      document.body.removeChild(ta);
      return ok;
    } catch (_) {
      return false;
    }
  }

  // node.object.get panel state
  const objectNamesByType = {
    input: ['streams', 'programs'],
    input_rec: ['streams', 'programs', 'stream-limits'],
    speed_video: ['info'],
    speed_audio: ['info'],
    sentinel_video: ['stats'],
    sentinel_audio: ['stats']
  };
  let autoRefreshNodeObjects = true;
  let nodeObjectsRefreshMs = 2000;
  let nodeObjects = {}; // { [nodeName]: { [objectName]: { ts, ok, data?, error? } } }
  let nodeObjectsInFlight = false;

  let lastInstanceSeen = null;

  let nextId = 1;
  const pending = new Map();
  const maxConsoleLines = 500;
  const maxLogLines = 1000;
  const logFlushMs = 250;
  let pendingLogLines = [];
  let logFlushTimer = null;

  function appendConsole(line) {
    if (!line) return;
    consoleLines = [...consoleLines, line].slice(-maxConsoleLines);
  }

  function flushPendingLogs() {
    if (!pendingLogLines.length) {
      logFlushTimer = null;
      return;
    }
    hasReceivedLog = true;
    logs = [...logs, ...pendingLogLines].slice(-maxLogLines);
    pendingLogLines = [];
    logFlushTimer = null;
  }

  function appendLog(line) {
    if (!line) return;
    pendingLogLines = [...pendingLogLines, line].slice(-maxLogLines);
    if (!logFlushTimer) {
      logFlushTimer = setTimeout(flushPendingLogs, logFlushMs);
    }
  }

  // Keep a reactive derived string so the stats panel updates when stats arrive.
  $: currentStatsPrettyText =
    currentInstanceId && statsByInstance[currentInstanceId]
      ? JSON.stringify(statsByInstance[currentInstanceId], null, 2)
      : '';

  function updateInstances(list) {
    const previous = instances;
    instances = Array.isArray(list) ? list : [];
    const wanted = currentInstanceId || requestedInstanceId;
    currentInstanceId = wanted
      ? (instances.some(instance => instance.id === wanted) ? wanted : null)
      : (instances[0]?.id || null);
    // Same instance, new process: re-request its graph but keep the shown one and the view
    // state until the new nodes.json arrives.
    if (currentInstanceId && instanceRestarted(previous, instances, currentInstanceId)) {
      appendConsole(`# Instance ${currentInstanceId} may have restarted, reloading its graph`);
      invalidateInstanceRequests();
      restartNodesRecheckGeneration = instanceGeneration;
    }
  }

  // Drops in-flight replies and the queue samples of the previous instance or process; the
  // reactive statement below then re-requests everything for the current instance.
  function invalidateInstanceRequests() {
    instanceGeneration += 1;
    for (const request of pending.values()) request.reject(new Error('Instance changed'));
    pending.clear();
    queuesFresh = false;
    queuesObservedAt = 0;
    queuesInstance = null;
  }

  async function loadInstances() {
    try {
      const res = await fetch(`${basePath}api/instances`);
      const data = await res.json();
      updateInstances(data.instances);
    } catch (e) {
      appendConsole(`# Failed to load instances: ${e}`);
    }
  }

  function connectWs() {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${proto}://${window.location.host}${basePath}ws`);

    ws.onopen = () => {
      wsConnected = true;
    };
    ws.onclose = () => {
      wsConnected = false;
      setTimeout(connectWs, 2000);
    };
    ws.onerror = () => {
      wsConnected = false;
    };
    ws.onmessage = (ev) => {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (!msg || typeof msg !== 'object') return;

      if (msg.type === 'response') {
        const { id, statusLine, body, error } = msg;
        const request = pending.get(id);
        // Check ownership before settling a promise: a delayed reply must not
        // complete a different instance's request.
        if (!request || msg.instanceId !== request.instanceId ||
            request.instanceId !== currentInstanceId || request.generation !== instanceGeneration) return;
        pending.delete(id);
        const code = parseInt(statusLine, 10) || 0;
        const ok = code >= 200 && code < 300;
        const resp = { statusLine, body, error, ok };
        if (ok) request.resolve(resp);
        else request.reject(resp);

        if (ok) {
          if (request.kind === 'nodes.json') {
            try {
              const arr = JSON.parse(body || '[]');
              nodes = Array.isArray(arr) ? arr : [];
            } catch (e) {
              appendConsole(`nodes.json parse error: ${e}`);
            }
            if (request.generation === restartNodesRecheckGeneration) {
              restartNodesRecheckGeneration = -1;
              setTimeout(() => {
                if (wsConnected && request.generation === instanceGeneration) refreshNodes();
              }, restartNodesRecheckMs);
            }
          } else if (request.kind === 'queues.json') {
            try {
              const arr = JSON.parse(body || '[]');
              queues = Array.isArray(arr) ? arr : [];
              queuesObservedAt = Date.now();
              queuesFresh = true;
              queuesInstance = currentInstanceId;
            } catch (e) {
              appendConsole(`queues.json parse error: ${e}`);
            }
          } else if (request.kind === 'sync_groups.json') {
            try {
              const arr = JSON.parse(body || '[]');
              syncGroups = Array.isArray(arr) ? arr : [];
            } catch (e) {
              appendConsole(`sync_groups.json parse error: ${e}`);
            }
          } else if (request.kind === 'correction_groups.json') {
            try {
              const arr = JSON.parse(body || '[]');
              correctionGroups = Array.isArray(arr) ? arr : [];
            } catch (e) {
              appendConsole(`correction_groups.json parse error: ${e}`);
            }
          }
        }
      } else if (msg.type === 'stats') {
        const instId = msg.instanceId || 'default';
        hasReceivedStats = true;
        statsByInstance = {
          ...statsByInstance,
          [instId]: msg.payload
        };
      } else if (msg.type === 'log') {
        appendLog(msg.line);
      } else if (msg.type === 'instances') {
        updateInstances(msg.instances);
      }
    };
  }

  function sendCommand(command, idOverride) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      appendConsole('# WS not connected');
      return Promise.reject(new Error('WebSocket not connected'));
    }
    if (!currentInstanceId) {
      appendConsole('# No instance selected');
      return Promise.reject(new Error('No instance selected'));
    }
    const id = `${idOverride || 'cmd'}-${nextId++}`;
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject, kind: idOverride, instanceId: currentInstanceId,
                        generation: instanceGeneration });
      ws.send(JSON.stringify({ type: 'command', id, command, instanceId: currentInstanceId }));
    });
  }

  // The Objects panel follows the selection: the selected node, or the nodes of the selected group, that expose objects.
  $: nodesSupportingObjects =
    Array.isArray(nodes) && selected
      ? nodes.filter((n) => n && n.name && objectNamesByType[n.type] &&
          (selected.kind === 'node' ? n.name === selected.id : String(n.params?.group || '') === selected.id))
      : [];
  // A new selection shows its objects at once instead of at the next tick.
  let objectsFor = '';
  $: {
    const key = nodesSupportingObjects.map((n) => n.name).join(',');
    if (key !== objectsFor) {
      objectsFor = key;
      clearNodeObjects();
      refreshNodeObjects();
    }
  }

  function refreshNodes() {
    sendCommand('nodes.json', 'nodes.json').catch(() => {});
  }
  function refreshQueues() {
    sendCommand('queues.json', 'queues.json').catch(() => {});
  }
  function refreshSyncGroups() {
    sendCommand('sync_groups.json', 'sync_groups.json').catch(() => {});
  }
  function refreshCorrectionGroups() {
    sendCommand('correction_groups.json', 'correction_groups.json').catch(() => {});
  }
  function clearNodeObjects() {
    nodeObjects = {};
  }

  // One node.object.get reply as the Objects panel shows it; null when the instance changed meanwhile.
  async function fetchNodeObject(nodeName, objName) {
    const generation = instanceGeneration;
    const cmd = `node.object.get ${nodeName} ${objName}`;
    const id = `obj-${encodeURIComponent(nodeName)}-${objName}-${Date.now()}`;
    try {
      const resp = await sendCommand(cmd, id);
      if (generation !== instanceGeneration) return null;
      let parsed = null;
      let parseOk = false;
      try {
        parsed = JSON.parse(resp.body || 'null');
        parseOk = true;
      } catch {
        parsed = resp.body || '';
      }
      return { ts: Date.now(), ok: true, data: parsed, json: parseOk };
    } catch (err) {
      if (generation !== instanceGeneration) return null;
      const msg =
        err && typeof err === 'object' ? err.error || err.statusLine || String(err) : String(err);
      // A declared node the engine has not created (e.g. the wipe-clip loader before a clip loads) has no objects yet.
      return { ts: Date.now(), ok: false, error: msg, off: /Node not created/.test(msg) };
    }
  }

  async function refreshNodeObjects() {
    if (nodeObjectsInFlight) return;
    if (!wsConnected) return;
    if (!currentInstanceId) return;
    if (!nodesSupportingObjects || nodesSupportingObjects.length === 0) {
      // keep state empty if nothing supports it
      if (Object.keys(nodeObjects).length) clearNodeObjects();
      return;
    }

    const generation = instanceGeneration, forKey = objectsFor;
    nodeObjectsInFlight = true;
    try {
      // One request at a time, applied together: a mixer has hundreds of objects, and every change of
      // nodeObjects updates all panels.
      const next = {};
      for (const n of nodesSupportingObjects) {
        for (const objName of objectNamesByType[n.type] || []) {
          const entry = n.working === false ? { ts: Date.now(), ok: false, off: true } : await fetchNodeObject(n.name, objName);
          if (!entry || generation !== instanceGeneration) return;
          (next[n.name] ||= {})[objName] = entry;
        }
      }
      nodeObjects = next;
    } finally {
      nodeObjectsInFlight = false;
      if (forKey !== objectsFor) refreshNodeObjects();
    }
  }

  onMount(() => {
    loadInstances();
    connectWs();
    const onHash = () => { page = pageFromHash(); };
    window.addEventListener('hashchange', onHash);

    const t = setInterval(() => {
      if (queuesFresh && Date.now() - queuesObservedAt > queuesRefreshMs * 3) queuesFresh = false;
      if (!wsConnected) return;
      if (!currentInstanceId) return;
      refreshQueues();
    }, queuesRefreshMs);

    const tObjects = setInterval(() => {
      if (!autoRefreshNodeObjects) return;
      refreshNodeObjects();
    }, nodeObjectsRefreshMs);

    const tISOs = setInterval(() => {
      if (!autoRefreshISOs) return;
      if (!wsConnected) return;
      if (!currentInstanceId) return;
      refreshSyncGroups();
      refreshCorrectionGroups();
    }, queuesRefreshMs);

    return () => {
      clearInterval(t);
      clearInterval(tObjects);
      clearInterval(tISOs);
      window.removeEventListener('hashchange', onHash);
    };
  });

  onDestroy(() => {
    if (logFlushTimer) {
      clearTimeout(logFlushTimer);
      logFlushTimer = null;
    }
  });

  $: {
    if (currentInstanceId !== lastInstanceSeen) {
      lastInstanceSeen = currentInstanceId;
      invalidateInstanceRequests();
      nodes = [];
      queues = [];
      selected = null;
      clearNodeObjects();
      syncGroups = [];
      correctionGroups = [];
      if (currentInstanceId) {
        requestedInstanceId = currentInstanceId;
        const url = new URL(window.location.href);
        url.searchParams.set('instance', currentInstanceId);
        window.history.replaceState(null, '', url);
      }
    }
    if (!wsConnected) requestedGeneration = -1;
    if (wsConnected && currentInstanceId && requestedGeneration !== instanceGeneration) {
      requestedGeneration = instanceGeneration;
      const generation = instanceGeneration;
      // give WS a tick in case selection happens during reconnect
      setTimeout(() => {
        if (!wsConnected || generation !== instanceGeneration) return;
        refreshNodes();
        refreshQueues();
        clearNodeObjects();
        refreshNodeObjects();
        refreshSyncGroups();
        refreshCorrectionGroups();
      }, 50);
    }
  }

  function setAutoRefreshNodeObjects(v) {
    autoRefreshNodeObjects = !!v;
  }

  $: currentInstance = instances.find(instance => instance.id === currentInstanceId) || null;
  $: liveState = !wsConnected ? 'down' : queueStatsFresh ? 'live' : currentInstanceId ? 'stale' : 'waiting';
  $: stamp = !wsConnected ? 'Disconnected · reconnecting'
    : !currentInstanceId ? (requestedInstanceId ? `Waiting for ${requestedInstanceId}` : 'No instance registered')
    : queueStatsFresh ? `Live · ${new Date(queuesObservedAt).toLocaleTimeString()}`
    : 'Waiting for queue stats';

  const panelRegistry = {
    graph: GraphPanel,
    selected: SelectedPanel,
    instance: InstancePanel,
    nodeObjects: NodeObjectsPanel,
    stats: StatsPanel,
    instanceSharedObjects: InstanceSharedObjectsPanel,
    logs: LogsPanel,
    console: ConsolePanel
  };

  const panel = (componentType, title, componentState = {}) => ({ type: 'component', componentType, title, componentState });
  // The graph card and a rail of widget cards, as on the mixer page; panels can be dragged into
  // other stacks, and Reset layout restores this.
  const defaultDockLayoutConfig = {
    root: {
      type: 'row',
      content: [
        { type: 'stack', size: '76%', header: { show: false }, content: [panel('graph', 'Graph')] },
        {
          type: 'column',
          size: '24%',
          content: [
            { type: 'stack', size: '45%', content: [panel('selected', 'Selected')] },
            { type: 'stack', size: '31%', content: [panel('instance', 'Instance')] },
            {
              type: 'stack',
              size: '24%',
              content: [
                panel('nodeObjects', 'Objects'),
                panel('stats', 'Stats'),
                panel('instanceSharedObjects', 'Shared'),
                panel('logs', 'Logs', { wrap: false, autoScroll: true }),
                panel('console', 'Messages')
              ]
            }
          ]
        }
      ]
    },
    settings: {
      constrainDragToContainer: true,
      reorderEnabled: true,
      responsiveMode: 'none',
      tabOverlapAllowance: 0,
      reorderOnTabMenuClick: true,
      tabControlOffset: 10
    },
    dimensions: {
      borderWidth: 12,
      borderGrabWidth: 12,
      defaultMinItemHeight: '60px',
      defaultMinItemWidth: '200px',
      headerHeight: 34
    },
    header: {
      show: 'top',
      popout: false,
      dock: false,
      close: 'Close',
      maximise: 'Maximise',
      minimise: 'Restore',
      tabDropdown: 'More panels'
    }
  };

  $: dockCtx = {
    nodes,
    queues,
    selected,
    setSelected,

    objectNamesByType,
    nodeObjects,
    nodesSupportingObjects,
    nodeObjectsInFlight,
    refreshNodeObjects,
    autoRefreshNodeObjects,
    nodeObjectsRefreshMs,
    setAutoRefreshNodeObjects,

    copyToClipboard,

    hasReceivedStats,
    currentStatsPrettyText,

    hasReceivedLog,
    logs,

    consoleLines,

    syncGroups,
    correctionGroups,
    refreshSyncGroups,
    refreshCorrectionGroups,
    autoRefreshISOs,
    setAutoRefreshISOs: (v) => { autoRefreshISOs = !!v; }
  };
</script>

<!-- Controls carry data-tip for the dark hover labels of the mixer page, never a native title. -->
<div class="app-root" use:hoverTips>
  <header class="appbar">
    <div class="brand">avplumber · web UI</div>
    <label class="inst" data-tip={currentInstance ? `${currentInstance.host}:${currentInstance.port} · ${stamp}` : stamp}>
      <span class="dot {liveState}"></span>
      <select bind:value={currentInstanceId} aria-label="Instance">
        {#if !currentInstanceId && requestedInstanceId}
          <option value={null}>Waiting for {requestedInstanceId}</option>
        {:else if instances.length === 0}
          <option value={null}>No instances</option>
        {/if}
        {#each instances as inst (inst.id)}
          <option value={inst.id}>{inst.name || inst.id}</option>
        {/each}
      </select>
    </label>
    <nav class="tabs" aria-label="Pages">
      <button aria-pressed={page === 'graph'} on:click={() => showPage('graph')}>Graph</button>
      <button aria-pressed={page === 'queues'} on:click={() => showPage('queues')}>Queues</button>
      <button aria-pressed={page === 'nodes'} on:click={() => showPage('nodes')}>Nodes</button>
    </nav>
    <div class="spacer"></div>
    {#if page === 'graph'}
      <button class="icon-btn small" data-tip="Put the widgets back where they started"
        on:click={() => dockHost && dockHost.resetLayout && dockHost.resetLayout()}>Reset layout</button>
    {/if}
    <span class="stamp">{stamp}</span>
  </header>

  <main data-nodes={nodes.length} data-queues={queues.length} data-selected={selected ? `${selected.kind}:${selected.id}` : ''}>
    <!-- The graph page stays mounted while a table page is shown, so the view keeps its camera. -->
    <section class="page page-graph" hidden={page !== 'graph'}>
      <DockHost
        bind:this={dockHost}
        registry={panelRegistry}
        ctx={dockCtx}
        layoutConfig={defaultDockLayoutConfig}
        storageKey="avplumber.webui.dock.layout.v2"
      />
    </section>
    {#if page === 'queues'}
      <QueuesPage {nodes} {queues} on:select={(e) => openInGraph(e.detail)} />
    {:else if page === 'nodes'}
      <NodesPage {nodes} {queues} on:select={(e) => openInGraph(e.detail)} />
    {/if}
  </main>
</div>
