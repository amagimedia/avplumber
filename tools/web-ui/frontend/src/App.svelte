<script>
  import { onMount } from 'svelte';
  import DockHost from './DockHost.svelte';
  import GraphPanel from './panels/GraphPanel.svelte';
  import NodesPanel from './panels/NodesPanel.svelte';
  import SelectedNodePanel from './panels/SelectedNodePanel.svelte';
  import NodeObjectsPanel from './panels/NodeObjectsPanel.svelte';
  import QueuesPanel from './panels/QueuesPanel.svelte';
  import StatsPanel from './panels/StatsPanel.svelte';
  import LogsPanel from './panels/LogsPanel.svelte';
  import ConsolePanel from './panels/ConsolePanel.svelte';

  let ws;
  let wsConnected = false;

  let instances = [];
  let currentInstanceId = null;

  let nodes = [];
  let queues = [];
  let queuesText = '';
  let statsByInstance = {};
  let currentStatsPrettyText = '';
  let hasReceivedStats = false;
  let hasReceivedLog = false;
  let logs = [];
  let consoleLines = [];
  let autoRefreshQueues = true;
  let autoRefreshMs = 1000;

  /** @type {any} */
  let dockHost;

  // Selected node state (from graph click or nodes list click)
  let selectedNodeName = '';
  $: selectedNode = selectedNodeName ? (nodes || []).find((n) => n && n.name === selectedNodeName) : null;

  function setSelectedNodeName(name) {
    selectedNodeName = name ? String(name) : '';
  }
  function setAutoRefreshQueues(v) {
    autoRefreshQueues = !!v;
  }

  // Clipboard helpers (for params table)
  let lastCopiedParamKey = '';
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

  async function copyParam(key, value) {
    const text =
      value == null
        ? ''
        : typeof value === 'string'
          ? value
          : typeof value === 'number' || typeof value === 'boolean'
            ? String(value)
            : JSON.stringify(value, null, 2);
    const ok = await copyToClipboard(text);
    if (ok) {
      lastCopiedParamKey = String(key);
      setTimeout(() => {
        if (lastCopiedParamKey === String(key)) lastCopiedParamKey = '';
      }, 900);
    }
  }

  // node.object.get panel state
  const objectNamesByType = {
    input: ['streams', 'programs'],
    input_rec: ['streams', 'programs', 'stream-limits'],
    speed_video: ['info'],
    speed_audio: ['info']
  };
  let autoRefreshNodeObjects = true;
  let nodeObjectsRefreshMs = 2000;
  let nodeObjects = {}; // { [nodeName]: { [objectName]: { ts, ok, data?, error? } } }
  let nodeObjectsInFlight = false;
  let selectedNodeObjectsInFlight = false;

  let lastInstanceSeen = null;

  let nextId = 1;
  const pending = new Map();

  function appendConsole(line) {
    if (!line) return;
    consoleLines = [...consoleLines, line];
  }

  function appendLog(line) {
    if (!line) return;
    hasReceivedLog = true;
    logs = [...logs, line];
  }

  // Keep a reactive derived string so the stats panel updates when stats arrive.
  $: currentStatsPrettyText =
    currentInstanceId && statsByInstance[currentInstanceId]
      ? JSON.stringify(statsByInstance[currentInstanceId], null, 2)
      : '';

  async function loadInstances() {
    try {
      const res = await fetch('/api/instances');
      const data = await res.json();
      instances = Array.isArray(data.instances) ? data.instances : [];
      if (instances.length > 0) {
        if (!currentInstanceId) {
          currentInstanceId = instances[0].id;
        }
      } else {
        currentInstanceId = null;
      }
    } catch (e) {
      appendConsole(`# Failed to load instances: ${e}`);
    }
  }

  function connectWs() {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${proto}://${window.location.host}/ws`;
    ws = new WebSocket(url);

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
        const resolver = id ? pending.get(id) : null;
        const code = parseInt(statusLine, 10) || 0;
        const ok = code >= 200 && code < 300;
        const resp = { statusLine, body, error, ok };
        if (resolver) {
          pending.delete(id);
          if (ok) {
            resolver(resp);
          } else {
            // propagate non-2xx as rejection to allow fallbacks
            resolver(Promise.reject(resp));
          }
        }
        const header = statusLine || (error ? 'ERROR' : '');
        const text = [header, body || '', error || ''].filter(Boolean).join('\n');
        //if (text) appendConsole(`# [${id || '-'}]\n${text}`);

        if (ok) {
          if (id === 'nodes.json') {
            try {
              const arr = JSON.parse(body || '[]');
              nodes = Array.isArray(arr) ? arr : [];
            } catch (e) {
              appendConsole(`nodes.json parse error: ${e}`);
            }
          } else if (id === 'queues.json') {
            try {
              const arr = JSON.parse(body || '[]');
              queues = Array.isArray(arr) ? arr : [];
              queuesText = '';
            } catch (e) {
              appendConsole(`queues.json parse error: ${e}`);
            }
          } else if (id === 'queues.stats') {
            queuesText = body || '';
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
    const id = idOverride || `cmd-${nextId++}`;
    return new Promise((resolve) => {
      pending.set(id, resolve);
      ws.send(JSON.stringify({ type: 'command', id, command, instanceId: currentInstanceId }));
    });
  }

  $: nodesSupportingObjects =
    Array.isArray(nodes) && nodes.length
      ? nodes.filter((n) => n && n.type && objectNamesByType[n.type] && n.name)
      : [];

  function refreshNodes() {
    sendCommand('nodes.json', 'nodes.json').catch(() => {});
  }
  function refreshQueues() {
    // Prefer JSON stats; fall back to text if unsupported
    sendCommand('queues.json', 'queues.json').catch(() => {
      sendCommand('queues.stats', 'queues.stats').catch(() => {});
    });
  }
  function resetQueueStats() {
    // Reset queue occupancy stats in avplumber so we don't miss peaks between UI polls.
    sendCommand('queues.stats.reset', 'queues.stats.reset')
      .catch(() => {})
      .finally(() => {
        refreshQueues();
      });
  }

  function clearNodeObjects() {
    nodeObjects = {};
  }

  function objectEntryText(v) {
    if (v == null) return '';
    if (typeof v === 'string') return v;
    if (typeof v === 'number' || typeof v === 'boolean') return String(v);
    try {
      return JSON.stringify(v, null, 2);
    } catch {
      return String(v);
    }
  }

  async function refreshSingleNodeObject(nodeName, objName) {
    const now = Date.now();
    const cmd = `node.object.get ${nodeName} ${objName}`;
    const id = `obj-${encodeURIComponent(nodeName)}-${objName}-${now}`;
    try {
      const resp = await sendCommand(cmd, id);
      let parsed = null;
      let parseOk = false;
      try {
        parsed = JSON.parse(resp.body || 'null');
        parseOk = true;
      } catch {
        parsed = resp.body || '';
      }
      nodeObjects = {
        ...nodeObjects,
        [nodeName]: {
          ...(nodeObjects[nodeName] || {}),
          [objName]: {
            ts: Date.now(),
            ok: true,
            data: parsed,
            json: parseOk
          }
        }
      };
    } catch (err) {
      const msg =
        err && typeof err === 'object' ? err.error || err.statusLine || String(err) : String(err);
      nodeObjects = {
        ...nodeObjects,
        [nodeName]: {
          ...(nodeObjects[nodeName] || {}),
          [objName]: {
            ts: Date.now(),
            ok: false,
            error: msg
          }
        }
      };
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

    nodeObjectsInFlight = true;
    try {
      // Sequential refresh: few calls, avoids state update races.
      for (const n of nodesSupportingObjects) {
        const objNames = objectNamesByType[n.type] || [];
        for (const objName of objNames) {
          await refreshSingleNodeObject(n.name, objName);
        }
      }
    } finally {
      nodeObjectsInFlight = false;
    }
  }

  $: selectedNodeObjectNames =
    selectedNode && selectedNode.type && objectNamesByType[selectedNode.type]
      ? objectNamesByType[selectedNode.type]
      : [];

  async function refreshSelectedNodeObjects() {
    if (selectedNodeObjectsInFlight) return;
    if (!wsConnected) return;
    if (!currentInstanceId) return;
    if (!selectedNode || !selectedNode.name) return;
    if (!selectedNodeObjectNames || selectedNodeObjectNames.length === 0) return;
    selectedNodeObjectsInFlight = true;
    try {
      for (const objName of selectedNodeObjectNames) {
        await refreshSingleNodeObject(selectedNode.name, objName);
      }
    } finally {
      selectedNodeObjectsInFlight = false;
    }
  }

  let consoleInput = '';

  function submitConsole() {
    const cmd = consoleInput.trim();
    if (!cmd) return;
    consoleInput = '';
    sendCommand(cmd).catch(() => {});
  }

  function handleConsoleKeydown(e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      submitConsole();
    }
  }

  onMount(() => {
    loadInstances();
    connectWs();

    setTimeout(() => {
      refreshNodes();
      refreshQueues();
    }, 1000);

    const t = setInterval(() => {
      if (!autoRefreshQueues) return;
      if (!wsConnected) return;
      if (!currentInstanceId) return;
      refreshQueues();
    }, autoRefreshMs);

    const tObjects = setInterval(() => {
      if (!autoRefreshNodeObjects) return;
      refreshNodeObjects();
    }, nodeObjectsRefreshMs);

    const tNodes = setInterval(() => {
      if (!wsConnected) return;
      if (!currentInstanceId) return;
      // Nodes/topology change rarely, but refreshing periodically keeps preview accurate.
      refreshNodes();
    }, 10000);

    return () => {
      clearInterval(t);
      clearInterval(tObjects);
      clearInterval(tNodes);
    };
  });

  $: {
    if (wsConnected && currentInstanceId && currentInstanceId !== lastInstanceSeen) {
      lastInstanceSeen = currentInstanceId;
      // give WS a tick in case selection happens during reconnect
      setTimeout(() => {
        refreshNodes();
        refreshQueues();
        clearNodeObjects();
        refreshNodeObjects();
      }, 50);
    }
  }

  function setAutoRefreshNodeObjects(v) {
    autoRefreshNodeObjects = !!v;
  }

  function setConsoleInput(v) {
    consoleInput = v == null ? '' : String(v);
  }

  const panelRegistry = {
    graph: GraphPanel,
    nodes: NodesPanel,
    selected: SelectedNodePanel,
    nodeObjects: NodeObjectsPanel,
    queues: QueuesPanel,
    stats: StatsPanel,
    logs: LogsPanel,
    console: ConsolePanel
  };

  const defaultDockLayoutConfig = {
    settings: {
      constrainDragToContainer: true,
      reorderEnabled: true,
      // popouts are useful, but can be confusing in debug UIs; keep disabled by default
      showPopoutIcon: false
    },
    header: {
      popout: false,
      maximise: 'Maximise',
      close: false,
      minimise: 'Minimise',
      tabDropdown: 'Tabs'
    },
    root: {
      type: 'column',
      content: [
        {
          type: 'row',
          size: '72%',
          content: [
            {
              type: 'stack',
              size: '22%',
              header: { maximise: 'Maximise', popout: false, close: false, minimise: false, tabDropdown: 'Tabs' },
              content: [
                { type: 'component', componentType: 'nodes', title: 'Graph / nodes', isClosable: false },
                { type: 'component', componentType: 'queues', title: 'Queues', isClosable: false }
              ]
            },
            { type: 'component', componentType: 'graph', title: 'Graph preview', isClosable: false, size: '56%' },
            {
              type: 'stack',
              size: '22%',
              header: { maximise: 'Maximise', popout: false, close: false, minimise: false, tabDropdown: 'Tabs' },
              content: [
                { type: 'component', componentType: 'selected', title: 'Selected node', isClosable: false },
                { type: 'component', componentType: 'nodeObjects', title: 'Node objects', isClosable: false },
                { type: 'component', componentType: 'stats', title: 'Statistics', isClosable: false }
              ]
            }
          ]
        },
        {
          type: 'stack',
          size: '28%',
          header: { maximise: 'Maximise', popout: false, close: false, minimise: false, tabDropdown: 'Tabs' },
          content: [
            { type: 'component', componentType: 'logs', title: 'Logs', isClosable: false },
            { type: 'component', componentType: 'console', title: 'Raw console', isClosable: false }
          ]
        }
      ]
    }
  };

  $: dockCtx = {
    nodes,
    queues,
    queuesText,

    selectedNodeName,
    selectedNode,
    setSelectedNodeName,

    refreshNodes,
    refreshQueues,
    resetQueueStats,

    autoRefreshQueues,
    autoRefreshMs,
    setAutoRefreshQueues,

    objectNamesByType,
    nodeObjects,
    nodesSupportingObjects,
    nodeObjectsInFlight,
    refreshNodeObjects,
    autoRefreshNodeObjects,
    nodeObjectsRefreshMs,
    setAutoRefreshNodeObjects,

    selectedNodeObjectNames,
    refreshSelectedNodeObjects,
    selectedNodeObjectsInFlight,

    objectEntryText,
    copyParam,
    copyToClipboard,
    lastCopiedParamKey,

    hasReceivedStats,
    currentStatsPrettyText,

    hasReceivedLog,
    logs,

    consoleLines,
    consoleInput,
    setConsoleInput,
    handleConsoleKeydown,
    submitConsole
  };
</script>

<div class="app-root">
  <header class="topbar">
    <div class="title">avplumber web-ui</div>
    <div class="status">
      <span class="badge {wsConnected ? 'badge-connected' : 'badge-disconnected'}">
        WS: {wsConnected ? 'connected' : 'disconnected'}
      </span>
      <button on:click={() => dockHost && dockHost.resetLayout && dockHost.resetLayout()}>Reset layout</button>
      <div class="instance-select">
        <label>
          Instance:
          <select bind:value={currentInstanceId}>
            {#if instances.length === 0}
              <option value="">no instances</option>
            {:else}
              {#each instances as inst}
                <option value={inst.id}>
                  {inst.name || inst.id} ({inst.host}:{inst.port})
                </option>
              {/each}
            {/if}
          </select>
        </label>
      </div>
    </div>
  </header>

  <DockHost
    bind:this={dockHost}
    registry={panelRegistry}
    ctx={dockCtx}
    layoutConfig={defaultDockLayoutConfig}
    storageKey="avplumber.webui.dock.layout.v1"
  />
</div>


