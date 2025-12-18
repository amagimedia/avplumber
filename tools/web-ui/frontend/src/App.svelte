<script>
  import { onMount } from 'svelte';
  import GraphPreview from './GraphPreview.svelte';

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

  // Selected node state (from graph click or nodes list click)
  let selectedNodeName = '';
  $: selectedNode = selectedNodeName ? (nodes || []).find((n) => n && n.name === selectedNodeName) : null;

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
</script>

<header class="topbar">
  <div class="title">avplumber web-ui</div>
  <div class="status">
    <span class="badge {wsConnected ? 'badge-connected' : 'badge-disconnected'}">
      WS: {wsConnected ? 'connected' : 'disconnected'}
    </span>
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

<main class="layout">
  <section class="panel panel-wide">
    <h2>Graph preview (Rete)</h2>
    <div class="toolbar">
      <button on:click={refreshNodes}>Refresh graph</button>
      <button on:click={refreshQueues}>Refresh queues</button>
      <button on:click={resetQueueStats}>Reset queue stats</button>
      <label class="hint" style="margin-left: auto;">
        <input type="checkbox" bind:checked={autoRefreshQueues} />
        auto-refresh queue fill ({autoRefreshMs}ms)
      </label>
    </div>
    <GraphPreview
      {nodes}
      {queues}
      {selectedNodeName}
      on:selectNode={(e) => {
        selectedNodeName = (e && e.detail && typeof e.detail.name === 'string') ? e.detail.name : '';
      }}
    />
  </section>

  <section class="panel">
    <h2>Graph / nodes</h2>
    <div class="toolbar">
      <button on:click={refreshNodes}>Refresh graph</button>
    </div>
    <div class="scrollable small-text">
      {#if nodes.length === 0}
        <div>No nodes yet.</div>
      {:else}
        <table class="node-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Type</th>
              <th>Group</th>
              <th>Status</th>
              <th>Params</th>
            </tr>
          </thead>
          <tbody>
            {#each nodes as n}
              <tr
                class:selected={selectedNodeName && n.name === selectedNodeName}
                class="click-row"
                on:click={() => (selectedNodeName = n.name)}
              >
                <td>{n.name}</td>
                <td>{n.type}</td>
                <td>{n.params && n.params.group ? n.params.group : ''}</td>
                <td>
                  <span class="node-badge {n.working ? 'node-badge-on' : 'node-badge-off'}">
                    {n.working ? 'ON' : 'OFF'}
                  </span>
                </td>
                <td class="params-json">
                  {JSON.stringify(n.params || {})}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      {/if}
    </div>
  </section>

  <section class="panel">
    <h2>Selected node</h2>

    {#if !selectedNode}
      <div class="hint">Click a node in the graph or in the nodes list to inspect it.</div>
    {:else}
      <div class="hint">
        <strong>{selectedNode.name}</strong>
        <span class="hint">({selectedNode.type})</span>
      </div>

      <details open>
        <summary><strong>Parameters</strong> <span class="hint">(creation-time)</span></summary>
        <div class="scrollable small-text" style="margin-top: 8px;">
          {#if selectedNode.params && typeof selectedNode.params === 'object'}
            {@const keys = Object.keys(selectedNode.params || {}).sort()}
            {#if keys.length === 0}
              <div class="hint">No params.</div>
            {:else}
              <table class="kv-table">
                <thead>
                  <tr>
                    <th>Key</th>
                    <th>Value</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {#each keys as k (k)}
                    <tr>
                      <td class="kv-key">{k}</td>
                      <td class="kv-val">
                        <pre class="kv-pre">{objectEntryText(selectedNode.params[k])}</pre>
                      </td>
                      <td class="kv-act">
                        <button class="btn-small" on:click={() => copyParam(k, selectedNode.params[k])}>
                          {lastCopiedParamKey === k ? 'Copied' : 'Copy'}
                        </button>
                      </td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            {/if}
          {:else}
            <div class="hint">No params.</div>
          {/if}
        </div>
      </details>

      <details open style="margin-top: 8px;">
        <summary><strong>Objects</strong> <span class="hint">(node.object.get)</span></summary>
        <div class="toolbar" style="margin-top: 8px;">
          <button on:click={refreshSelectedNodeObjects} disabled={selectedNodeObjectsInFlight}>
            {selectedNodeObjectsInFlight ? 'Refreshing…' : 'Refresh objects'}
          </button>
        </div>

        <div class="scrollable small-text">
          {#if !selectedNodeObjectNames || selectedNodeObjectNames.length === 0}
            <div class="hint">
              This node type does not expose objects via <code>node.object.get</code> (known mapping).
            </div>
          {:else}
            {#each selectedNodeObjectNames as objName (objName)}
              <div style="margin: 8px 0;">
                <div class="hint"><code>{`node.object.get ${selectedNode.name} ${objName}`}</code></div>
                {#if nodeObjects[selectedNode.name] && nodeObjects[selectedNode.name][objName] && nodeObjects[selectedNode.name][objName].ok}
                  <pre class="small-text">
{JSON.stringify(nodeObjects[selectedNode.name][objName].data, null, 2)}</pre
                  >
                {:else if nodeObjects[selectedNode.name] && nodeObjects[selectedNode.name][objName] && nodeObjects[selectedNode.name][objName].error}
                  <div class="hint">Error: {nodeObjects[selectedNode.name][objName].error}</div>
                {:else}
                  <div class="hint">No data yet.</div>
                {/if}
              </div>
            {/each}
          {/if}
        </div>
      </details>
    {/if}
  </section>

  <section class="panel">
    <h2>Node objects (node.object.get)</h2>
    <div class="toolbar">
      <button on:click={refreshNodeObjects} disabled={nodeObjectsInFlight}>
        {nodeObjectsInFlight ? 'Refreshing…' : 'Refresh node objects'}
      </button>
      <label class="hint" style="margin-left: auto;">
        <input type="checkbox" bind:checked={autoRefreshNodeObjects} />
        auto-refresh ({nodeObjectsRefreshMs}ms)
      </label>
    </div>

    <div class="scrollable small-text">
      {#if !nodesSupportingObjects || nodesSupportingObjects.length === 0}
        <div class="hint">
          No nodes in this graph currently expose objects via <code>node.object.get</code>.
        </div>
      {:else}
        {#each nodesSupportingObjects as n (n.name)}
          <details open>
            <summary>
              <strong>{n.name}</strong>
              <span class="hint">({n.type})</span>
            </summary>
            <div class="small-text" style="margin: 8px 0 12px 0;">
              {#each objectNamesByType[n.type] as objName (objName)}
                <div style="margin: 8px 0;">
                  <div class="hint"><code>{`node.object.get ${n.name} ${objName}`}</code></div>
                  {#if nodeObjects[n.name] && nodeObjects[n.name][objName] && nodeObjects[n.name][objName].ok}
                    <pre class="small-text">{JSON.stringify(nodeObjects[n.name][objName].data, null, 2)}</pre>
                  {:else if nodeObjects[n.name] && nodeObjects[n.name][objName] && nodeObjects[n.name][objName].error}
                    <div class="hint">Error: {nodeObjects[n.name][objName].error}</div>
                  {:else}
                    <div class="hint">No data yet.</div>
                  {/if}
                </div>
              {/each}
            </div>
          </details>
        {/each}
      {/if}
    </div>
  </section>

  <section class="panel">
    <h2>Queues</h2>
    <div class="toolbar">
      <button on:click={refreshQueues}>Refresh queues</button>
    </div>
    <div class="scrollable small-text">
      {#if queues.length > 0}
        <table class="node-table">
          <thead>
            <tr>
              <th>Queue</th>
              <th>Type</th>
              <th>Fill</th>
              <th>Capacity</th>
              <th>Fill</th>
              <th>Frames in queue<br />(min/avg/max)</th>
              <th>PTS (s)</th>
              <th>Timebase</th>
              <th>pps</th>
              <th>enq/deq/drop</th>
            </tr>
          </thead>
          <tbody>
            {#each queues as q}
              <tr>
                <td>{q.name}</td>
                <td>{q.type || ''}</td>
                <td>{q.occupied}</td>
                <td>{q.capacity}</td>
                <td>
                  <div class="queue-bar">
                    <div
                      class="queue-bar-fill"
                      style={`width: ${
                        q.capacity > 0 ? Math.min(100, Math.max(0, (q.occupied / q.capacity) * 100)) : 0
                      }%;`}
                    />
                    <span class="queue-bar-text">
                      {#if q.capacity > 0}
                        {Math.round((q.occupied / q.capacity) * 100)}%
                      {:else}
                        -
                      {/if}
                    </span>
                  </div>
                </td>
                <td>
                  {#if q.frames_in_queue}
                    {q.frames_in_queue.min}/
                    {typeof q.frames_in_queue.avg === 'number' ? q.frames_in_queue.avg.toFixed(2) : q.frames_in_queue.avg}/{q.frames_in_queue.max}
                  {:else}
                    -
                  {/if}
                </td>
                <td>{q.last_ts_seconds}</td>
                <td>{q.timebase_num}/{q.timebase_den}</td>
                <td>{typeof q.pps === 'number' ? q.pps.toFixed(1) : ''}</td>
                <td>
                  {q.enqueued_total ?? ''}/{q.dequeued_total ?? ''}/{q.dropped_total ?? ''}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      {:else}
        <pre class="small-text">{queuesText}</pre>
      {/if}
    </div>
  </section>

  <section class="panel">
    <h2>Statistics (stats.subscribe)</h2>
    {#if !hasReceivedStats}
      <p class="hint">
        Configure <code>stats.subscribe</code> in avplumber with
        <code>"url":"http://WEBUI_HOST:WEBUI_PORT/api/stats?instance=INSTANCE_ID"</code>.
      </p>
    {/if}
    <pre class="scrollable small-text">{currentStatsPrettyText}</pre>
  </section>

  <section class="panel">
    <h2>Logs</h2>
    {#if !hasReceivedLog}
      <p class="hint">
        If avplumber is started with <code>-l logfile</code> and web-ui has
        <code>AVPLUMBER_LOGFILE</code> set, new lines will appear here.
      </p>
    {/if}
    <pre class="scrollable small-text">
{#each logs as line}
{line}
{/each}</pre>
  </section>

  <section class="panel panel-wide">
    <h2>Raw console</h2>
    <div class="console-input">
      <input
        type="text"
        bind:value={consoleInput}
        placeholder="Command, e.g. nodes.json"
        on:keydown={handleConsoleKeydown}
      />
      <button on:click={submitConsole}>Send</button>
    </div>
    <pre class="scrollable small-text">
{#each consoleLines as line}
{line}
{/each}</pre>
  </section>
</main>


