<script>
  import TextPanelToolbar from './TextPanelToolbar.svelte';

  export let ctx; // store
  export let dock;
  export let state;

  $: c = $ctx;
  $: dock;
  let wrap = !!(state && state.wrap);
  $: if (state && typeof state.wrap === 'boolean') wrap = state.wrap;

  function persist() {
    try {
      if (dock?.container) {
        dock.container.stateRequestEvent = () => ({ wrap });
      }
      dock?.saveLayout?.();
    } catch (_) {
      // ignore
    }
  }

  function setWrap(v) {
    wrap = !!v;
    persist();
  }

  function copy() {
    const t = c.currentStatsPrettyText || '';
    c.copyToClipboard?.(t);
  }
</script>

<div class="panel">
  <TextPanelToolbar wrap={wrap} onWrapChange={setWrap} onCopy={copy} />
  {#if c.hasReceivedStats}
    <pre class="body" class:wrap>{c.currentStatsPrettyText}</pre>
  {:else}
    <div class="body">
      <p class="hint">
        Stream analysis statistics appear here once avplumber sends them: configure <code>stats.subscribe</code> with
        <code>"url":"http://WEBUI_HOST:WEBUI_PORT/api/stats?instance=INSTANCE_ID"</code>.
      </p>
    </div>
  {/if}
</div>
