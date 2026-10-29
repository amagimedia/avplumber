<script>
  // The selected node or group: state, wiring, queues with their fill, what compositors draw, params.
  import { createEventDispatcher } from 'svelte';
  import './graph.css';
  import { graphModel, selectionPath, nodeInputs, nodeOutputs, groupInputs } from './model.mjs';
  import { fillPct, fillClass, framesText, dropText, trimParams, kindClass, subscriptionText } from './format.mjs';

  export let nodes = [];
  export let queues = [];
  export let selected = null;  // {kind: 'node' | 'group', id, queue?} or null
  export let copyToClipboard = async () => false;
  let copied = false;
  $: selected, copied = false;

  const dispatch = createEventDispatcher();
  const MAX_ROWS = 8;

  $: model = graphModel(nodes, queues);
  $: sel = selected && (selected.kind === 'node' ? model.meta.has(selected.id) : model.groups.has(selected.id)) ? selected : null;
  $: path = selectionPath(model, sel);
  $: node = sel?.kind === 'node' ? model.meta.get(sel.id) : null;
  $: group = sel?.kind === 'group' ? model.groups.get(sel.id) : null;
  $: ins = node ? nodeInputs(model, node.n) : group ? groupInputs(model, group.id) : [];
  $: outs = node ? nodeOutputs(model, node.n) : [];
  $: gid = node?.g || group?.id;
  $: src = gid && model.groups.get(gid).src;
  $: drawnBy = gid && (model.onAir.get(gid) || []);
  $: showDrawnBy = gid && (model.onAir.has(gid) || model.groups.get(gid).member && model.groups.get(gid).d === 'inputs');
  $: drawing = node && model.drawing.get(node.n);
  $: params = node ? JSON.stringify(trimParams(node.p), null, 1) : '';

  const pickNode = id => dispatch('select', {kind: 'node', id});
  const pickGroup = id => dispatch('select', {kind: 'group', id});
  // Following a queue opens its other end and keeps the queue highlighted.
  const follow = q => dispatch('select', {kind: 'node', id: q.a === sel?.id ? q.b : q.a, queue: q.n});
</script>

<div class="gv-inspector">
  {#if !sel}
    <p class="gv-empty">Pinch or scroll-zoom from the overview down to a single node; it never re-lays out. Click a group, a connection or a queue to follow it, or press <code>/</code> to search. Keys <code>1</code>–<code>4</code> jump between Overview, Groups, Nodes and Details.</p>
  {:else}
    <p class="gv-nm">{sel.id}</p>
    <div class="gv-pills">
      {#if node}
        {#if node.w}<span class="pill p-on">ON</span>{:else}<span class="pill p-off">OFF</span>{/if}
        {#if ins.some(q => q.st === 'drop')}<span class="pill p-warn">dropped frames</span>{/if}
      {:else}
        <span class="pill p-tap">{group.d}</span>
        {#if group.st === 'drop'}<span class="pill p-warn">dropped frames</span>{/if}
        {#if group.off}<span class="pill p-off">has OFF nodes</span>{/if}
      {/if}
    </div>
    {#if node}
      <dl class="gv-kv">
        <dt>Type</dt><dd><code>{node.t}</code></dd>
        <dt>Group</dt><dd><button class="icon-btn" style="padding:2px 8px" on:click={() => pickGroup(node.g)}>{node.g}</button></dd>
        {#if src}<dt>Source</dt><dd class="gv-src"><i class={kindClass(src)}></i>{src}</dd>{/if}
        <dt>Path</dt><dd>{path.nodes.size} nodes · {path.edges.size} flowing queues</dd>
      </dl>
    {:else if src}
      <dl class="gv-kv"><dt>Source</dt><dd class="gv-src"><i class={kindClass(src)}></i>{src}</dd></dl>
    {/if}
    {#if showDrawnBy}
      <div class="sub-h">Drawn by · {drawnBy.length}</div>
      {#if drawnBy.length}
        <div class="gv-glist">{#each drawnBy as id}<button on:click={() => pickNode(id)}>{id}</button>{/each}</div>
      {:else}<span class="gv-empty">not drawn by any compositor</span>{/if}
    {/if}
    {#if drawing}
      <div class="sub-h">Drawing · {drawing.length} sources</div>
      <div class="gv-glist">{#each drawing as id}<button on:click={() => pickGroup(id)}>{id}</button>{/each}</div>
    {/if}
    {#if group}
      <div class="sub-h">Nodes · {group.n}</div>
      <div class="gv-glist">{#each group.nodes as id}<button data-tip={model.meta.get(id).t} on:click={() => pickNode(id)}>{id}</button>{/each}</div>
    {/if}
    {#each node ? [['Inputs', ins, 'a'], ['Outputs', outs, 'b']] : [['Inputs from other groups', ins, 'a']] as [title, qs, end]}
      {@const flowing = qs.filter(q => q.st !== 'idle')}
      <div class="sub-h">{title} · {qs.length}</div>
      <div class="gv-qlist">
        {#each flowing.slice(0, MAX_ROWS) as q (q.n)}
          <button class="gv-q" data-tip={q.n} on:click={() => follow(q)}>
            <span>{q[end]}</span>
            <div class="fill {fillClass(q)}"><i style="width:{fillPct(q)}%"></i></div>
            <em>{q.pps.toFixed(0)}</em>
            <span class="gv-qmeta">{framesText(q)} · {dropText(q)}{q.subscriptionActive === null ? '' : ` · ${subscriptionText(q)}`}</span>
          </button>
        {:else}<span class="gv-empty">none flowing</span>{/each}
        {#if flowing.length > MAX_ROWS}<span class="gv-empty">+{flowing.length - MAX_ROWS} more on the Queues page</span>{/if}
        {#if qs.length > flowing.length}<span class="gv-empty">{qs.length - flowing.length} idle (nothing subscribed)</span>{/if}
      </div>
    {/each}
    {#if node}
      <div class="sub-h">Params</div>
      <button class="icon-btn small" aria-label="Copy node parameters" on:click={async () => { copied = await copyToClipboard(JSON.stringify(node.p, null, 2)); }}>{copied ? 'Copied' : 'Copy JSON'}</button>
      <pre class="gv-params">{params}</pre>
      <details class="gv-full-params">
        <summary>All parameters</summary>
        <pre class="gv-params">{JSON.stringify(node.p, null, 2)}</pre>
      </details>
    {/if}
  {/if}
</div>
