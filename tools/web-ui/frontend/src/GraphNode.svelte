<script>
  import { Ref } from 'rete-svelte-plugin/svelte';
  import GraphQueueLabel from './GraphQueueLabel.svelte';
  export let data;
  export let emit;

  $: inputs = Object.entries(data.inputs);
  $: outputs = Object.entries(data.outputs);
  $: rows = Array.from({ length: Math.max(inputs.length, outputs.length) }, (_, index) => [inputs[index], outputs[index]]);

  function renderSocket(element, side, key, port) {
    emit({ type: 'render', data: { type: 'socket', side, key,
      nodeId: data.id, element, payload: port.socket } });
  }
</script>

<div class:vertical={data.__vertical} class="node" data-testid="node" style:width={`${data.width || 300}px`}>
  <div class="title" data-testid="title">{data.label}</div>
  {#each rows as row, rowIndex}
    <div class="ports">
      {#each row as port, index}
        {@const side = index ? 'output' : 'input'}
        <div class={`port ${side}`} data-testid={port ? `${side}-${port[0]}` : undefined} title={port ? port[0] : ''}>
          {#if port}
            <span class="port-label" data-testid={`${side}-title`}><GraphQueueLabel name={port[0]} showFill={index === 1} /></span>
            <Ref class={`socket-ref ${side}-socket`} data-testid={`${side}-socket`}
              style={data.__vertical ? `left:calc(${100 * (rowIndex + 1) / ((index ? outputs : inputs).length + 1)}% - 6px);right:auto;top:${index ? 'auto' : '-7px'};bottom:${index ? '-7px' : 'auto'}` : ''}
              init={element => renderSocket(element, side, port[0], port[1])}
              unmount={element => emit({ type: 'unmount', data: { element } })} />
          {/if}
        </div>
      {/each}
    </div>
  {/each}
</div>

<style>
  .node { position:relative; box-sizing:border-box; background:#132035; border:1px solid #475569; border-radius:6px; padding-bottom:8px; color:#e2e8f0; cursor:pointer; font-family:system-ui,sans-serif; }
  .node:hover { border-color:#94a3b8; }
  .title { padding:10px 12px; white-space:pre-line; overflow-wrap:anywhere; font-size:15px; line-height:1.35; }
  .ports { display:flex; min-height:28px; }
  .port { width:50%; min-width:0; position:relative; display:flex; align-items:center; box-sizing:border-box; }
  .vertical .port { position:static; }
  .input { padding:0 10px 0 12px; }
  .output { padding:0 12px 0 10px; justify-content:flex-end; }
  .port-label { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:11px; line-height:28px; }
  .port :global(.socket-ref) { position:absolute; top:8px; width:12px; height:12px; }
  .port :global(.input-socket) { left:-7px; }
  .port :global(.output-socket) { right:-7px; }
  .port :global(.socket) { width:12px; height:12px; margin:0; border:1px solid #a5b4fc; background:#64748b; vertical-align:top; }
</style>
