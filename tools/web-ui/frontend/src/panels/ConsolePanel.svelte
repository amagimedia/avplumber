<script>
  export let ctx; // store
  export let dock;
  export let state;

  $: c = $ctx;
  $: dock;
  $: state;
</script>

<!-- Display only: the UI's own messages (parse errors, restart notices). -->
<div class="panel">
  {#if c.consoleLines.length}
    <pre class="body">{c.consoleLines.join('\n')}</pre>
  {:else}
    <div class="body"><p class="hint">No messages.</p></div>
  {/if}
</div>
