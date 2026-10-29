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

<div class="panel">
  <TextPanelToolbar wrap={wrap} onWrapChange={setWrap} onCopy={copyAll}>
    <label class="check" data-tip="Ask every node below for its objects every {c.nodeObjectsRefreshMs / 1000} s">
      <input type="checkbox" checked={c.autoRefreshNodeObjects}
        on:change={(e) => c.setAutoRefreshNodeObjects?.(e.target.checked)} /> Auto-refresh
    </label>
    <button class="icon-btn small" on:click={() => c.refreshNodeObjects?.()} disabled={c.nodeObjectsInFlight}>
      Refresh
    </button>
  </TextPanelToolbar>

  <div class="body">
    {#if !c.nodesSupportingObjects || c.nodesSupportingObjects.length === 0}
      <p class="hint">No node in this graph exposes objects via <code>node.object.get</code>.</p>
    {:else}
      {#each c.nodesSupportingObjects as n (n.name)}
        <details open>
          <summary><b>{n.name}</b><small>{n.type}</small></summary>
          {#each (c.objectNamesByType?.[n.type] || []) as objName (objName)}
            {@const entry = c.nodeObjects?.[n.name]?.[objName]}
            <div class="obj">
              <div class="sub-h">{objName}</div>
              {#if entry?.ok}
                <pre class:wrap>{JSON.stringify(entry.data, null, 2)}</pre>
              {:else if entry?.error}
                <p class="hint">Error: {entry.error}</p>
              {:else}
                <p class="hint">No data yet.</p>
              {/if}
            </div>
          {/each}
        </details>
      {/each}
    {/if}
  </div>
</div>
