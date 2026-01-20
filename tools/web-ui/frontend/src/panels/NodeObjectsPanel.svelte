<script>
  import TextPanelToolbar from './TextPanelToolbar.svelte';

  export let ctx; // store
  export let dock;
  export let state;

  $: c = $ctx;
  $: dock;

  let wrap = !!(state && state.wrap);
  $: if (state && typeof state.wrap === 'boolean') wrap = state.wrap;

  function persist() {
    try {
      if (dock?.container) dock.container.stateRequestEvent = () => ({ wrap });
      dock?.saveLayout?.();
    } catch (_) {
      // ignore
    }
  }

  function setWrap(v) {
    wrap = !!v;
    persist();
  }

  function copyAll() {
    const out = [];
    for (const n of c.nodesSupportingObjects || []) {
      const objNames = (c.objectNamesByType?.[n.type] || []).slice();
      for (const objName of objNames) {
        const entry = c.nodeObjects?.[n.name]?.[objName];
        if (entry && entry.ok) {
          out.push(`# node.object.get ${n.name} ${objName}\n${JSON.stringify(entry.data, null, 2)}`);
        } else if (entry && entry.error) {
          out.push(`# node.object.get ${n.name} ${objName}\nERROR: ${entry.error}`);
        }
      }
    }
    c.copyToClipboard?.(out.join('\n\n'));
  }
</script>

<div class="panel avp-panel">
  <TextPanelToolbar wrap={wrap} onWrapChange={setWrap} onCopy={copyAll} />

  <div class="toolbar">
    <button on:click={() => c.refreshNodeObjects?.()} disabled={c.nodeObjectsInFlight}>
      {c.nodeObjectsInFlight ? 'Refreshing…' : 'Refresh node objects'}
    </button>
    <label class="hint" style="margin-left: auto;">
      <input
        type="checkbox"
        checked={c.autoRefreshNodeObjects}
        on:change={(e) => c.setAutoRefreshNodeObjects?.(e.target.checked)}
      />
      auto-refresh ({c.nodeObjectsRefreshMs}ms)
    </label>
  </div>

  <div class="scrollable small-text">
    {#if !c.nodesSupportingObjects || c.nodesSupportingObjects.length === 0}
      <div class="hint">
        No nodes in this graph currently expose objects via <code>node.object.get</code>.
      </div>
    {:else}
      {#each c.nodesSupportingObjects as n (n.name)}
        <details open>
          <summary>
            <strong>{n.name}</strong>
            <span class="hint">({n.type})</span>
          </summary>
          <div class="small-text" style="margin: 8px 0 12px 0;">
            {#each (c.objectNamesByType?.[n.type] || []) as objName (objName)}
              <div style="margin: 8px 0;">
                <div class="hint"><code>{`node.object.get ${n.name} ${objName}`}</code></div>
                {#if c.nodeObjects?.[n.name]?.[objName]?.ok}
                  <pre class="small-text" class:avp-prewrap={wrap}>{JSON.stringify(c.nodeObjects[n.name][objName].data, null, 2)}</pre>
                {:else if c.nodeObjects?.[n.name]?.[objName]?.error}
                  <div class="hint">Error: {c.nodeObjects[n.name][objName].error}</div>
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
</div>


