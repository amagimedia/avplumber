<script>
  export let wrap = false;
  export let onWrapChange = () => {};
  export let onCopy = null;
</script>

<div class="toolbar">
  <label class="hint">
    <input type="checkbox" checked={wrap} on:change={(e) => onWrapChange(!!e.target.checked)} />
    wrap
  </label>

  {#if onCopy}
    <button on:click={onCopy} style="margin-left: auto;">Copy</button>
  {/if}
</div>


