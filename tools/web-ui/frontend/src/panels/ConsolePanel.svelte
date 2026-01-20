<script>
  export let ctx; // store
  export let dock;
  export let state;

  $: c = $ctx;
  $: dock;
  $: state;
</script>

<div class="panel avp-panel">
  <div class="console-input">
    <input
      type="text"
      value={c.consoleInput}
      placeholder="Command, e.g. nodes.json"
      on:input={(e) => c.setConsoleInput?.(e.target.value)}
      on:keydown={(e) => c.handleConsoleKeydown?.(e)}
    />
    <button on:click={() => c.submitConsole?.()}>Send</button>
  </div>

  <pre class="scrollable small-text">
{#each c.consoleLines as line}
{line}
{/each}</pre>
</div>


