<script>
  import GraphPreview from '../GraphPreview.svelte';

  export let ctx; // store
  export let dock; // unused for now
  export let state; // unused for now

  $: c = $ctx;
  $: dock;
  $: state;

  let liveQueueStats = true;
  let liveQueueStatsUserSet = false;

  $: graphNodeCount = Array.isArray(c?.nodes) ? c.nodes.length : 0;
  $: graphQueueCount = Array.isArray(c?.queues) ? c.queues.length : 0;
  $: largeGraph = graphNodeCount >= 150 || graphQueueCount >= 150 || graphNodeCount + graphQueueCount >= 300;
  $: if (!liveQueueStatsUserSet) liveQueueStats = !largeGraph;

  function setLiveQueueStats(value) {
    liveQueueStatsUserSet = true;
    liveQueueStats = value;
  }
</script>

<div class="panel avp-panel">
  <div class="toolbar">
    <button on:click={() => c.refreshNodes?.()}>Refresh graph</button>
    <button on:click={() => c.refreshQueues?.()}>Refresh queues</button>
    <button on:click={() => c.resetQueueStats?.()}>Reset queue stats</button>
    <label class="hint">
      <input type="checkbox" checked={liveQueueStats} on:change={(e) => setLiveQueueStats(e.target.checked)} />
      live graph queue stats{largeGraph && !liveQueueStats ? ' off for large graph' : ''}
    </label>
    <label class="hint" style="margin-left: auto;">
      <input type="checkbox" checked={c.autoRefreshQueues} on:change={(e) => c.setAutoRefreshQueues?.(e.target.checked)} />
      auto-refresh queue fill ({c.autoRefreshMs}ms)
    </label>
  </div>

  <GraphPreview
    nodes={c.nodes}
    queues={c.queues}
    selectedNodeName={c.selectedNodeName}
    liveQueueStats={liveQueueStats}
    on:selectNode={(e) => {
      const name = e && e.detail && typeof e.detail.name === 'string' ? e.detail.name : '';
      c.setSelectedNodeName?.(name);
    }}
  />
</div>
