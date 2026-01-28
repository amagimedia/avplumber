<script>
  import TextPanelToolbar from './TextPanelToolbar.svelte';

  export let ctx; // store
  export let dock;
  export let state;

  $: c = $ctx;
  $: dock;
  $: selectedNode = c.selectedNode;

  let wrapObjects = !!(state && state.wrapObjects);
  $: if (state && typeof state.wrapObjects === 'boolean') wrapObjects = state.wrapObjects;

  function persist() {
    try {
      if (dock?.container) dock.container.stateRequestEvent = () => ({ wrapObjects });
      dock?.saveLayout?.();
    } catch (_) {
      // ignore
    }
  }

  function setWrapObjects(v) {
    wrapObjects = !!v;
    persist();
  }
</script>

<div class="panel avp-panel">
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
                      <pre class="kv-pre">{c.objectEntryText ? c.objectEntryText(selectedNode.params[k]) : String(selectedNode.params[k] ?? '')}</pre>
                    </td>
                    <td class="kv-act">
                      <button class="btn-small" on:click={() => c.copyParam?.(k, selectedNode.params[k])}>
                        {c.lastCopiedParamKey === k ? 'Copied' : 'Copy'}
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
      <TextPanelToolbar wrap={wrapObjects} onWrapChange={setWrapObjects} />
      <div class="toolbar" style="margin-top: 8px;">
        <button on:click={() => c.refreshSelectedNodeObjects?.()} disabled={c.selectedNodeObjectsInFlight}>
          {c.selectedNodeObjectsInFlight ? 'Refreshing…' : 'Refresh objects'}
        </button>
      </div>

      <div class="scrollable small-text">
        {#if !c.selectedNodeObjectNames || c.selectedNodeObjectNames.length === 0}
          <div class="hint">
            This node type does not expose objects via <code>node.object.get</code> (known mapping).
          </div>
        {:else}
          {#each c.selectedNodeObjectNames as objName (objName)}
            <div style="margin: 8px 0;">
              <div class="hint"><code>{`node.object.get ${selectedNode.name} ${objName}`}</code></div>
              {#if c.nodeObjects?.[selectedNode.name]?.[objName]?.ok}
                <pre class="small-text" class:avp-prewrap={wrapObjects}>{JSON.stringify(c.nodeObjects[selectedNode.name][objName].data, null, 2)}</pre>
              {:else if c.nodeObjects?.[selectedNode.name]?.[objName]?.error}
                <div class="hint">Error: {c.nodeObjects[selectedNode.name][objName].error}</div>
              {:else}
                <div class="hint">No data yet.</div>
              {/if}
            </div>
          {/each}
        {/if}
      </div>
    </details>
  {/if}
</div>


