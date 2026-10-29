<script>
  // The graph widget: toolbar (search, legend, fullscreen), the zoomable stage, HUD and zoom controls. In fullscreen
  // the "floating" slot shows the selection beside the stage.
  import { createEventDispatcher, onDestroy, onMount } from 'svelte';
  import './graph.css';
  import { graphModel, searchEntries, search } from './model.mjs';
  import { createGraphStage } from './stage.mjs';

  export let nodes = [];
  export let queues = [];
  export let selected = null;  // {kind: 'node' | 'group', id, queue?} or null

  const dispatch = createEventDispatcher();
  const LEVEL_NAMES = ['Overview', 'Groups', 'Nodes', 'Details'];
  let root, stageEl, sigmaEl, edgesEl, zonesEl, cardsEl, tipEl, msEl, searchEl;
  let stage = null, shownModel = null, shownVersion = -1, noGl = false;
  let level = -1, max = false, msText = '';

  $: model = graphModel(nodes, queues);
  $: if (stage) sync(model);
  $: if (stage && shownModel) stage.setSelection(selected);

  function sync(m) {
    if (m !== shownModel) { stage.setModel(m); shownModel = m; }
    else if (m.version !== shownVersion) stage.statsChanged();
    shownVersion = m.version;
  }

  onMount(() => {
    try {
      stage = createGraphStage({stage: stageEl, sigma: sigmaEl, edges: edgesEl, zones: zonesEl, cards: cardsEl, tip: tipEl}, {
        onSelect: s => dispatch('select', s),
        onLevel: l => { level = l; },
        onFrame: ms => { const t = ms.toFixed(1); if (t !== msText) { msText = t; msEl.textContent = t; } },
      });
    } catch (e) {
      // sigma throws without WebGL (GPU off, locked-down or remote-desktop browsers); its first canvas is already in
      // place. The stage stays empty with a notice, search still selects into the inspector.
      console.warn('Graph stage unavailable:', e);
      sigmaEl.replaceChildren();
      noGl = true;
    }
    document.addEventListener('keydown', onKey);
    document.addEventListener('fullscreenchange', onFullscreenChange);
  });
  onDestroy(() => {
    stage?.destroy();
    document.removeEventListener('keydown', onKey);
    document.removeEventListener('fullscreenchange', onFullscreenChange);
  });

  // ---------- fullscreen ----------
  function setMax(on) { max = on; stage?.resize(); }
  async function toggleMax() {
    const on = !max;
    try {
      if (on && root.requestFullscreen) await root.requestFullscreen();
      else if (!on && document.fullscreenElement) await document.exitFullscreen();
    } catch (_) { /* the fixed overlay still fills the window */ }
    setMax(on);
  }
  function onFullscreenChange() { if (!document.fullscreenElement && max) setMax(false); }

  // ---------- search ----------
  let query = '', hits = [], hi = 0, suggOpen = false, entries = [], entriesModel = null;
  function showSugg() {
    if (entriesModel !== model) { entries = searchEntries(model); entriesModel = model; }
    hits = search(entries, query); hi = 0; suggOpen = !!query.trim();
  }
  function pick(x) {
    const s = {kind: x.kind, id: x.id};
    if (stage) stage.select(s, true); else dispatch('select', s);
    suggOpen = false; searchEl.blur();
  }
  function searchKey(ev) {
    if ((ev.key === 'ArrowDown' || ev.key === 'ArrowUp') && hits.length) { hi = (hi + (ev.key === 'ArrowDown' ? 1 : -1) + hits.length) % hits.length; ev.preventDefault(); }
    if (ev.key === 'Enter' && hits[hi]) pick(hits[hi]);
    if (ev.key === 'Escape') { suggOpen = false; searchEl.blur(); }
  }

  // ---------- keys, while the graph is on screen ----------
  function onKey(ev) {
    if (!root.getClientRects().length || ev.metaKey || ev.ctrlKey || ev.altKey) return;
    if (ev.target.closest?.('input,select,textarea,[contenteditable]')) return;
    if (ev.key === '/') { ev.preventDefault(); searchEl.focus(); }
    else if (!stage) return;  // without WebGL only the search works
    else if (ev.key === 'F' && ev.shiftKey) toggleMax();
    else if (ev.key >= '1' && ev.key <= '4') stage.jumpLevel(+ev.key - 1);
    else if (ev.key === '0') stage.fit();
    else if (ev.key === '+' || ev.key === '=') stage.zoomIn();
    else if (ev.key === '-') stage.zoomOut();
    else if (ev.key === 'Escape') { if (max && !document.fullscreenElement) setMax(false); else stage.select(null); }
  }
</script>

<div class="gv-gw" class:max bind:this={root}>
  <div class="bar">
    <div class="tools">
      <div class="gv-search">
        <input bind:this={searchEl} bind:value={query} on:input={showSugg} on:keydown={searchKey} on:blur={() => setTimeout(() => { suggOpen = false; }, 100)}
          type="search" placeholder="Find node or group  ( / )" autocomplete="off" aria-label="Find node or group" />
        {#if suggOpen}
          <div class="gv-sugg">
            {#each hits as x, k (x.kind + x.id)}
              <button class:on={k === hi} on:mousedown|preventDefault={() => pick(x)}>{x.id}<small>{x.sub}</small></button>
            {:else}
              <div class="gv-empty" style="padding:6px 8px">No match</div>
            {/each}
          </div>
        {/if}
      </div>
    </div>
    <div class="tools">
      <div class="gv-legend">
        <span><i style="border-color:var(--pvw)"></i>frames</span><span><i style="border-color:var(--wipe)"></i>packets</span><span><i style="border-color:var(--accent)"></i>near full</span><span><i style="border-color:var(--pgm)"></i>dropped</span><span><b class="tally"></b>drawn now</span>
      </div>
      {#if !noGl}<button class="icon-btn" data-tip="Fullscreen graph (Shift+F)" on:click={toggleMax}>{max ? '⤡ Exit fullscreen' : '⤢ Fullscreen'}</button>{/if}
    </div>
  </div>
  <div class="gv-stage" bind:this={stageEl}>
    <div class="gv-zones" bind:this={zonesEl}></div>
    <canvas class="gv-edges" bind:this={edgesEl}></canvas>
    <div class="gv-sigma" bind:this={sigmaEl}></div>
    <div class="gv-cards" bind:this={cardsEl}></div>
    <div class="gv-tip" bind:this={tipEl} hidden></div>
    {#if noGl}<p class="gv-empty gv-nogl">The graph needs WebGL, which this browser does not provide; search and the Queues and Nodes pages still work.</p>{/if}
    <div class="gv-hud">
      {#if !noGl}
        <div class="seg" role="group" aria-label="Zoom level">
          {#each LEVEL_NAMES as name, l}<button aria-pressed={level === l} on:click={() => stage?.jumpLevel(l)}>{name}</button>{/each}
        </div>
      {/if}
      <span><b>{model.nodeCount.toLocaleString()}</b> nodes</span><span><b>{model.edges.length.toLocaleString()}</b> queues</span>
      {#if !noGl}<span>frame <b bind:this={msEl}>–</b> ms</span>{/if}
    </div>
    {#if !noGl}
      <div class="gv-zoomctl">
        <button aria-label="Zoom in" on:click={() => stage?.zoomIn()}>+</button>
        <button aria-label="Zoom out" on:click={() => stage?.zoomOut()}>−</button>
        <button aria-label="Fit whole graph" data-tip="Fit (0)" on:click={() => stage?.fit()}>⌂</button>
      </div>
    {/if}
    <div class="sheet gv-float" class:has={!!selected}>
      <div class="bar"><h3>Selected</h3></div>
      <div class="gv-wbody"><slot name="floating" /></div>
    </div>
  </div>
</div>
