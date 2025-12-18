<script>
  import GraphPreview from '../GraphPreview.svelte';

  export let ctx; // store
  export let dock; // unused for now
  export let state; // unused for now

  $: c = $ctx;
  $: dock;
  $: state;
</script>

<div class="panel avp-panel">
  <div class="toolbar">
    <button on:click={() => c.refreshNodes?.()}>Refresh graph</button>
    <button on:click={() => c.refreshQueues?.()}>Refresh queues</button>
    <button on:click={() => c.resetQueueStats?.()}>Reset queue stats</button>
    <label class="hint" style="margin-left: auto;">
      <input type="checkbox" checked={c.autoRefreshQueues} on:change={(e) => c.setAutoRefreshQueues?.(e.target.checked)} />
      auto-refresh queue fill ({c.autoRefreshMs}ms)
    </label>
  </div>

  <GraphPreview
    nodes={c.nodes}
    queues={c.queues}
    selectedNodeName={c.selectedNodeName}
    on:selectNode={(e) => {
      const name = e && e.detail && typeof e.detail.name === 'string' ? e.detail.name : '';
      c.setSelectedNodeName?.(name);
    }}
  />
</div>


