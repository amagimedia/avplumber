<script>
  export let wrap = false;
  export let onWrapChange = () => {};
  export let onCopy = null;
</script>

<div class="toolbar">
  <label class="check"><input type="checkbox" checked={wrap} on:change={(e) => onWrapChange(!!e.target.checked)} /> Wrap</label>
  <slot />
  {#if onCopy}
    <button class="icon-btn small push" on:click={onCopy}>Copy</button>
  {/if}
</div>
