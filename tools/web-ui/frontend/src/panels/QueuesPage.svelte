<script>
  import { createEventDispatcher } from 'svelte';
  import { graphModel, isProblem } from '../graph/model.mjs';
  import { fillClass, fillPct } from '../graph/format.mjs';

  export let nodes = [];
  export let queues = [];

  const dispatch = createEventDispatcher();
  const maxRows = 400;
  const filters = [['all', 'All'], ['active', 'Flowing'], ['idle', 'Idle'], ['prob', 'Problems']];
  let text = '';
  let filter = 'all';

  // The graph's model: the same queue states (drops since the previous sample, by-design fan-out skips) as the graph.
  $: model = graphModel(nodes, queues);
  // In the engine's order (queues.json), then the queues only named in node params.
  $: all = [...new Set([...(queues || []).map((s) => model.queues.get(s?.name)).filter(Boolean), ...model.queues.values()])];
  $: needle = text.trim().toLowerCase();
  $: matching = all.filter((q) =>
    !(filter === 'active' && q.st === 'idle' || filter === 'idle' && q.st !== 'idle' || filter === 'prob' && !isProblem(q.st)) &&
    (!needle || q.n.toLowerCase().includes(needle) || (q.a || '').toLowerCase().includes(needle) || (q.b || '').toLowerCase().includes(needle)));
  $: rows = matching.slice(0, maxRows);
  // Opens the consumer (or, for a queue nothing takes, its producer) with the queue outlined.
  const open = (q) => { if (q.b || q.a) dispatch('select', { kind: 'node', id: q.b || q.a, queue: q.n }); };
</script>

<section class="page sheet tbl-page">
  <div class="bar">
    <h3>Queues</h3>
    <div class="tools">
      <input class="filter" type="search" placeholder="Filter queues" aria-label="Filter queues" bind:value={text} />
      <div class="seg" role="group" aria-label="Queue state">
        {#each filters as [id, label] (id)}
          <button aria-pressed={filter === id} on:click={() => (filter = id)}>{label}</button>
        {/each}
      </div>
    </div>
  </div>
  <div class="tbl-wrap">
    <table>
      <thead>
        <tr><th>Queue</th><th>From → to</th><th>Fill</th><th>Frames in queue<br />now / avg / max</th><th>Capacity</th><th>pps</th><th>enq / deq / drop</th><th>State</th></tr>
      </thead>
      <tbody>
        {#each rows as q (q.n)}
          <tr data-queue={q.n} on:click={() => open(q)}>
            <td class="m">{q.n}</td>
            <td class="m">{q.a || '–'} → {q.b || '–'}</td>
            <td style="width: 120px"><div class="fill {fillClass(q)}"><i style="width: {fillPct(q)}%"></i></div></td>
            <td class="n">{q.o} / {q.avg} / {q.mx}</td>
            <td class="n">{q.c}</td>
            <td class="n">{q.pps.toFixed(1)}</td>
            <td class="n">{q.enq.toLocaleString()} / {q.deq.toLocaleString()} / {q.dr.toLocaleString()}</td>
            <td>
              {#if q.st === 'drop'}<span class="pill p-bad" data-tip="Frames lost since the previous sample">dropped {q.drD}</span>
              {:else if q.st === 'full'}<span class="pill p-warn" data-tip="Average fill at 90% of capacity or more">near full</span>
              {:else if q.st === 'idle'}<span class="pill p-off" data-tip="Nothing flows, e.g. nothing subscribed">idle</span>
              {:else}<span class="pill p-on">flowing</span>{/if}
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
  <div class="cap">
    Showing {rows.length.toLocaleString()} of {matching.length.toLocaleString()} matching queues ({all.length.toLocaleString()} total).
    Click a row to open it in the graph.
  </div>
</section>
