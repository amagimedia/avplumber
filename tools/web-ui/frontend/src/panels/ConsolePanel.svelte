<script>
  export let ctx; // store
  export let dock;
  export let state;

  $: c = $ctx;
  $: dock;
  $: state;
</script>

<!-- Display only: the UI's own messages (parse errors, restart notices). -->
<div class="panel avp-panel">
  <pre class="scrollable small-text">
{#each c.consoleLines as line}
{line}
{/each}</pre>
</div>


