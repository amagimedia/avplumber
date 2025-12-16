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
  let logs = [];
  let consoleLines = [];
  let autoRefreshQueues = true;
  let autoRefreshMs = 1000;

  let lastInstanceSeen = null;

  let nextId = 1;
  const pending = new Map();

  function appendConsole(line) {
    if (!line) return;
    consoleLines = [...consoleLines, line];
  }

  function appendLog(line) {
    if (!line) return;
    logs = [...logs, line];
  }

  function currentStatsPretty() {
    if (currentInstanceId && statsByInstance[currentInstanceId]) {
      return JSON.stringify(statsByInstance[currentInstanceId], null, 2);
    }
    return '';
  }

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

  function refreshNodes() {
    sendCommand('nodes.json', 'nodes.json').catch(() => {});
  }
  function refreshQueues() {
    // Prefer JSON stats; fall back to text if unsupported
    sendCommand('queues.json', 'queues.json').catch(() => {
      sendCommand('queues.stats', 'queues.stats').catch(() => {});
    });
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

    const tNodes = setInterval(() => {
      if (!wsConnected) return;
      if (!currentInstanceId) return;
      // Nodes/topology change rarely, but refreshing periodically keeps preview accurate.
      refreshNodes();
    }, 10000);

    return () => {
      clearInterval(t);
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
      <label class="hint" style="margin-left: auto;">
        <input type="checkbox" bind:checked={autoRefreshQueues} />
        auto-refresh queue fill ({autoRefreshMs}ms)
      </label>
    </div>
    <GraphPreview {nodes} {queues} />
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
              <tr>
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
    <p class="hint">
      Configure <code>stats.subscribe</code> in avplumber with
      <code>"url":"http://WEBUI_HOST:WEBUI_PORT/api/stats?instance=INSTANCE_ID"</code>.
    </p>
    <pre class="scrollable small-text">{currentStatsPretty()}</pre>
  </section>

  <section class="panel">
    <h2>Logs</h2>
    <p class="hint">
      If avplumber is started with <code>-l logfile</code> and web-ui has
      <code>AVPLUMBER_LOGFILE</code> set, new lines will appear here.
    </p>
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


