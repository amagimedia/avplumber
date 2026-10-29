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
    <label class="check" data-tip="Ask the selected node for its objects every {c.nodeObjectsRefreshMs / 1000} s">
      <input type="checkbox" checked={c.autoRefreshNodeObjects}
        on:change={(e) => c.setAutoRefreshNodeObjects?.(e.target.checked)} /> Auto-refresh
    </label>
    <button class="icon-btn small" on:click={() => c.refreshNodeObjects?.()} disabled={c.nodeObjectsInFlight}>
      Refresh
    </button>
  </TextPanelToolbar>

  <div class="body">
    {#if !c.selected}
      <p class="hint">Select an input node or its group to see its streams and programs.</p>
    {:else if !c.nodesSupportingObjects || c.nodesSupportingObjects.length === 0}
      <p class="hint">The selected {c.selected.kind} has no streams or programs to show.</p>
    {:else}
      {#each c.nodesSupportingObjects as n (n.name)}
        {@const names = c.objectNamesByType?.[n.type] || []}
        {@const off = n.working === false || names.some((o) => c.nodeObjects?.[n.name]?.[o]?.off)}
        <details open>
          <summary><b>{n.name}</b><small>{n.type}</small></summary>
          {#if off}
            <p class="hint">Not running right now, so it has no streams or programs.</p>
          {:else}
            {#each names as objName (objName)}
              {@const entry = c.nodeObjects?.[n.name]?.[objName]}
              <div class="obj">
                <div class="sub-h">{objName}</div>
                {#if entry?.ok}
                  <pre class:wrap>{JSON.stringify(entry.data, null, 2)}</pre>
                {:else if entry?.error}
                  <p class="hint">Unavailable: {entry.error.replace(/^\d{3} ERROR:\s*/, '')}</p>
                {:else}
                  <p class="hint">No data yet.</p>
                {/if}
              </div>
            {/each}
          {/if}
        </details>
      {/each}
    {/if}
  </div>
</div>
