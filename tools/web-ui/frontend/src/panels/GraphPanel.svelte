<script>
  import GraphView from '../graph/GraphView.svelte';
  import Inspector from '../graph/Inspector.svelte';

  export let ctx; // store
  export let dock;
  export let state;

  $: c = $ctx;
  $: dock;
  $: state;
  const select = (e) => c.setSelected?.(e.detail);
</script>

<!-- The graph card; in fullscreen the inspector floats over it. -->
<div class="panel">
  <GraphView nodes={c.nodes} queues={c.queues} selected={c.selected} compose={c.compose} on:select={select}>
    <svelte:fragment slot="floating">
      <Inspector nodes={c.nodes} queues={c.queues} selected={c.selected} on:select={select} />
    </svelte:fragment>
  </GraphView>
</div>
