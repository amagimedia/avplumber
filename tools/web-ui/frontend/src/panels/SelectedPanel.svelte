<script>
  import Inspector from '../graph/Inspector.svelte';

  export let ctx; // store
  export let dock;
  export let state;

  $: c = $ctx;
  $: dock;
  $: state;
</script>

<div class="panel">
  {#if c.selected}
    <div class="toolbar"><button class="icon-btn small push" on:click={() => c.setSelected?.(null)}>Clear</button></div>
  {/if}
  <div class="body">
    <Inspector nodes={c.nodes} queues={c.queues} selected={c.selected} on:select={(e) => c.setSelected?.(e.detail)} />
  </div>
</div>
