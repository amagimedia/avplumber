<script>
  import InstanceTiles from '../graph/InstanceTiles.svelte';

  export let ctx; // store
  export let dock;
  export let state;

  $: c = $ctx;
  $: dock;
  $: state;
</script>

<div class="panel">
  <div class="body"><InstanceTiles nodes={c.nodes} queues={c.queues} /></div>
</div>
