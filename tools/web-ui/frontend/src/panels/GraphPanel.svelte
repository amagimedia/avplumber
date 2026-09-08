<script>
  import GraphPreview from '../GraphPreview.svelte';
  import { groupGraph, focusGroup } from '../graphGroups.mjs';

  export let ctx; // store
  export let dock;
  export let state; // unused for now

  $: c = $ctx;
  $: dock;
  $: state;

  let grouped = true;
  let family = '';
  let focused = '';
  $: overview = groupGraph(c?.nodes || [], c?.queues || []);
  $: familyMembers = family
    ? [...new Set((overview.membership.get(family)?.nodes || []).map(n => n.params.group))].sort((a,b) => a.localeCompare(b, undefined, {numeric:true}))
    : [];
  $: projection = focused ? focusGroup(c?.nodes || [], c?.queues || [], focused) : overview;
  function selectGraphNode(name) {
    // Projection IDs are navigation targets, never native control identities.
    if (!grouped || focused) {
      if ((c?.nodes || []).some(node => node.name === name)) c.setSelectedNodeName?.(name);
      return;
    }
    const group = grouped && overview.membership.get(name);
    if (group?.key.startsWith('family:')) family = name;
    else if (group) focused = group.label;
    else c.setSelectedNodeName?.(name);
  }
  function showOverview() { family = ''; focused = ''; }
  function goBack() {
    if (focused && family) focused = '';
    else showOverview();
  }
  function setGrouped(value) { grouped = value; showOverview(); }
  $: if (family && !overview.membership.has(family)) showOverview();
  $: if (focused && !(c?.nodes || []).some(node => node.params?.group === focused)) showOverview();
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

<div class="panel avp-panel" data-graph-view={!grouped ? 'full' : focused ? 'group' : family ? 'family' : 'overview'} data-graph-focus={focused}>
  <div class="toolbar">
    {#if grouped && (family || focused)}
      <button class="graph-back" title={focused && family ? `Back to ${overview.membership.get(family)?.label}` : 'Back to overview'} on:click={goBack}>← Back</button>
    {/if}
    <label class="hint"><input class="graph-grouped-toggle" type="checkbox" checked={grouped} on:change={(e) => setGrouped(e.target.checked)} /> Grouped overview</label>
    {#if grouped}<button class="graph-overview" on:click={showOverview}>Overview</button>{/if}
    <button class="graph-expand" on:click={() => dock?.container?.parent?.parent?.toggleMaximise?.()}>Expand / restore graph</button>
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

  <div class="hint breadcrumb">
    <span class="flow-legend" title="Sampled queue activity, not frame uniqueness or timing. Amber can be intentional buffering. Fill marker shows the fullest constituent queue.">{liveQueueStats && c.queueStatsFresh ? 'Flow: blue active · gray idle/unknown · amber accumulating/not draining · red new drops' : 'Flow: paused or awaiting fresh queue samples'}</span>
  </div>
  <div class="hint breadcrumb">
    <span class="graph-native-counts" data-nodes={overview.nodeCount} data-queues={overview.queueCount}>{overview.nodeCount} native nodes · {overview.queueCount} defined queues</span>
    {#if grouped}
      {#if family || focused}
        <button class="graph-overview" on:click={showOverview}>Overview</button>
        {#if family}<span>›</span><button class="graph-family-back" on:click={() => focused = ''}>{overview.membership.get(family)?.label}</button>{/if}
        {#if focused}<span>› {focused}</span>{/if}
      {:else} · {overview.nodes.length} blocks. Open a group to inspect its internals.{/if}
    {/if}
  </div>
  {#if grouped && family && !focused}
    <div class="group-browser">
      {#each familyMembers as name}
        <button on:click={() => focused = name}>
          <strong>{name}</strong>
          <span>{(c.nodes || []).filter(n => n.params?.group === name).length} nodes</span>
        </button>
      {/each}
    </div>
  {:else}
    <GraphPreview
      groupedLayout={grouped}
      focusedLayout={grouped && !!focused}
      nodes={grouped ? projection.nodes : c.nodes}
      queues={grouped ? projection.queues : c.queues}
      selectedNodeName={c.selectedNodeName}
      liveQueueStats={liveQueueStats && c.queueStatsFresh}
      on:selectNode={(e) => selectGraphNode(e?.detail?.name || '')}
    />
    {#if grouped && focused}<div class="hint breadcrumb">Drag to pan · Scroll to zoom · Click a node to inspect · Boundary rates are totals</div>{/if}
  {/if}
</div>

<style>
  .toolbar { flex-wrap: wrap; }
  .toolbar .graph-back { background:#2563eb; border-color:#60a5fa; color:#fff; font-weight:700; padding:6px 14px; }
  .toolbar .graph-back:hover { background:#1d4ed8; }
  .breadcrumb { padding: 6px 10px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .group-browser { display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:12px; padding:16px; overflow:auto; }
  .group-browser button { display:flex; flex-direction:column; align-items:flex-start; padding:16px; gap:8px; }
  .group-browser strong { font-size:16px; }
  .group-browser span { color:#94a3b8; }
  .toolbar button, .toolbar label { white-space: nowrap; flex-shrink: 0; }
</style>
