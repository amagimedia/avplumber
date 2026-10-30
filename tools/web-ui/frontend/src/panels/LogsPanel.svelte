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

  // Follow new lines when enabled; only a new logs array has them (the panel context changes far more often).
  let followedLogs = null;
  $: if (autoScroll && preEl && c.logs !== followedLogs) {
    followedLogs = c.logs;
    setTimeout(scrollToBottom, 0);  // after the DOM update
  }

  onMount(() => {
    persist();
    if (autoScroll) scrollToBottom();
  });
</script>

<div class="panel">
  <TextPanelToolbar wrap={wrap} onWrapChange={setWrap} onCopy={copy}>
    <label class="check" data-tip="Keep the newest line in view; scrolling up pauses it">
      <input type="checkbox" checked={autoScroll} on:change={(e) => setAutoScroll(e.target.checked)} /> Follow
    </label>
  </TextPanelToolbar>
  {#if !c.hasReceivedLog}
    <div class="body" style="flex: none">
      <p class="hint">
        If avplumber is started with <code>-l logfile</code> and the web UI has <code>AVPLUMBER_LOGFILE</code> set, new lines
        appear here.
      </p>
    </div>
  {/if}
  <pre bind:this={preEl} class="body" class:wrap on:scroll={handleScroll}>{(c.logs || []).join('\n')}</pre>
</div>
