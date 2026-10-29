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
  <div class="body">
    {#if c.queueStatsFresh}<InstanceTiles nodes={c.nodes} queues={c.queues} />
    {:else}<p class="hint">Waiting for live queue telemetry.</p>{/if}
  </div>
</div>
