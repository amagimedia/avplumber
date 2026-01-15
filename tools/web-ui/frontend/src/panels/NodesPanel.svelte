<script>
  export let ctx; // store
  export let dock;
  export let state;

  $: c = $ctx;
  $: dock;
  $: state;
</script>

<div class="panel avp-panel">
  <div class="toolbar">
    <button on:click={() => c.refreshNodes?.()}>Refresh graph</button>
  </div>

  <div class="scrollable small-text">
    {#if !c.nodes || c.nodes.length === 0}
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
          {#each c.nodes as n}
            <tr
              class:selected={c.selectedNodeName && n.name === c.selectedNodeName}
              class="click-row"
              on:click={() => c.setSelectedNodeName?.(n.name)}
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
</div>


