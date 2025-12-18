<script>
  import { onMount } from 'svelte';
  import TextPanelToolbar from './TextPanelToolbar.svelte';

  export let ctx; // store
  export let dock;
  export let state;

  $: c = $ctx;
  $: dock;

  let wrap = !!(state && state.wrap);
  let autoScroll = state && typeof state.autoScroll === 'boolean' ? state.autoScroll : true;
  $: if (state && typeof state.wrap === 'boolean') wrap = state.wrap;
  $: if (state && typeof state.autoScroll === 'boolean') autoScroll = state.autoScroll;

  /** @type {HTMLElement | null} */
  let preEl = null;

  function persist() {
    try {
      if (dock?.container) {
        dock.container.stateRequestEvent = () => ({ wrap, autoScroll });
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

  function setAutoScroll(v) {
    autoScroll = !!v;
    persist();
    if (autoScroll) scrollToBottom();
  }

  function scrollToBottom() {
    if (!preEl) return;
    preEl.scrollTop = preEl.scrollHeight;
  }

  function handleScroll() {
    if (!preEl) return;
    const atBottom = preEl.scrollTop + preEl.clientHeight >= preEl.scrollHeight - 8;
    // If user scrolls away, pause following. If they return to bottom, resume.
    if (!atBottom && autoScroll) {
      autoScroll = false;
      persist();
    } else if (atBottom && !autoScroll) {
      autoScroll = true;
      persist();
    }
  }

  function copy() {
    const t = (c.logs || []).join('\n');
    c.copyToClipboard?.(t);
  }

  // Follow new logs when enabled.
  $: if (autoScroll) {
    // Trigger after DOM updates
    if (preEl) setTimeout(scrollToBottom, 0);
  }

  onMount(() => {
    persist();
    if (autoScroll) scrollToBottom();
  });
</script>

<div class="panel avp-panel">
  <div class="toolbar">
    <label class="hint">
      <input type="checkbox" checked={wrap} on:change={(e) => setWrap(e.target.checked)} />
      wrap
    </label>
    <label class="hint" style="margin-left: 0.6rem;">
      <input type="checkbox" checked={autoScroll} on:change={(e) => setAutoScroll(e.target.checked)} />
      follow
    </label>
    <button on:click={copy} style="margin-left: auto;">Copy</button>
  </div>

  {#if !c.hasReceivedLog}
    <p class="hint">
      If avplumber is started with <code>-l logfile</code> and web-ui has
      <code>AVPLUMBER_LOGFILE</code> set, new lines will appear here.
    </p>
  {/if}
  <pre bind:this={preEl} class="scrollable small-text" class:avp-prewrap={wrap} on:scroll={handleScroll}>
{#each c.logs as line}
{line}
{/each}</pre>
</div>


