<script>
  import { createEventDispatcher } from 'svelte';
  import { graphModel } from '../graph/model.mjs';
  import TablePager from './TablePager.svelte';

  export let nodes = [];
  export let queues = [];

  const dispatch = createEventDispatcher();
  const maxRows = 400;
  let text = '';
  let page = 0, previousFilter = '';

  $: model = graphModel(nodes, queues);
  $: all = model ? [...model.meta.values()] : [];
  $: needle = text.trim().toLowerCase();
  $: matching = needle
    ? all.filter((m) => m.n.toLowerCase().includes(needle) || m.t.toLowerCase().includes(needle) || m.g.toLowerCase().includes(needle))
    : all;
  $: if (needle !== previousFilter) { previousFilter = needle; page = 0; }
  $: rows = matching.slice(page * maxRows, (page + 1) * maxRows);
</script>

<section class="page sheet tbl-page">
  <div class="bar">
    <h3>Nodes</h3>
    <div class="tools">
      <input class="filter" type="search" placeholder="Filter nodes" aria-label="Filter nodes" bind:value={text} />
    </div>
  </div>
  <div class="tbl-wrap">
    <table>
      <thead><tr><th>Name</th><th>Type</th><th>Group</th><th>In / out</th><th>Status</th></tr></thead>
      <tbody>
        {#each rows as m (m.n)}
          <tr data-node={m.n} on:click={() => dispatch('select', { kind: 'node', id: m.n })}>
            <td class="m">{m.n}</td>
            <td class="m">{m.t}</td>
            <td>{m.g}</td>
            <td class="n">{model.inQ.get(m.n)?.length || 0} / {model.outQ.get(m.n)?.length || 0}</td>
            <td>{#if m.w}<span class="pill p-on">ON</span>{:else}<span class="pill p-off">OFF</span>{/if}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
  <TablePager total={matching.length} pageSize={maxRows} bind:page label="nodes" />
</section>
