<script>
  import { tick } from 'svelte';
  import { graphHoveredQueueName, selectedQueueName } from '../graphStores';

  export let ctx; // store
  export let dock;
  export let state;

  $: c = $ctx;
  $: dock;
  $: state;

  let lastScrolledSel = '';

  function parsePts(v) {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }

  function formatDelta(seconds) {
    if (!Number.isFinite(seconds)) return '';
    const s = seconds.toFixed(6).replace(/\.?0+$/, '');
    const sign = seconds > 0 ? '+' : '';
    return `${sign}${s}`;
  }

  $: refPts =
    $selectedQueueName && c.queues
      ? parsePts(c.queues.find((x) => x && x.name === $selectedQueueName)?.last_ts_seconds)
      : null;

  $: {
    const sel = $selectedQueueName;
    if (sel && sel !== lastScrolledSel && typeof document !== 'undefined') {
      lastScrolledSel = sel;
      tick().then(() => {
        const el = document.getElementById(`queue-row-${encodeURIComponent(sel)}`);
        el?.scrollIntoView?.({ block: 'nearest', behavior: 'smooth' });
      });
    }
    if (!sel) lastScrolledSel = '';
  }
</script>

<div class="panel avp-panel">
  <div class="toolbar">
    <button on:click={() => c.refreshQueues?.()}>Refresh queues</button>
    <button type="button" class="btn-small" on:click={() => selectedQueueName.set('')} disabled={!$selectedQueueName}>
      Clear queue selection
    </button>
  </div>

  <div class="scrollable small-text">
    {#if c.queues && c.queues.length > 0}
      <table class="node-table">
        <thead>
          <tr>
            <th>Queue</th>
            <th>Type</th>
            <th>Fill</th>
            <th>Capacity</th>
            <th>Fill</th>
            <th>Frames in queue<br />(min/avg/max)</th>
            <th>PTS (s)</th>
            <th>Timebase</th>
            <th>pps</th>
            <th>enq/deq/drop</th>
          </tr>
        </thead>
        <tbody>
          {#each c.queues as q (q.name)}
            <tr
              id={`queue-row-${encodeURIComponent(q.name)}`}
              class="click-row"
              class:selected={$selectedQueueName === q.name}
              class:queue-graph-hover={$graphHoveredQueueName === q.name}
              on:click={() => selectedQueueName.update((s) => (s === q.name ? '' : q.name))}
            >
              <td>{q.name}</td>
              <td>{q.type || ''}</td>
              <td>{q.occupied}</td>
              <td>{q.capacity}</td>
              <td>
                <div class="queue-bar">
                  <div
                    class="queue-bar-fill"
                    style={`width: ${
                      q.capacity > 0 ? Math.min(100, Math.max(0, (q.occupied / q.capacity) * 100)) : 0
                    }%;`}
                  />
                  <span class="queue-bar-text">
                    {#if q.capacity > 0}
                      {Math.round((q.occupied / q.capacity) * 100)}%
                    {:else}
                      -
                    {/if}
                  </span>
                </div>
              </td>
              <td>
                {#if q.frames_in_queue}
                  {q.frames_in_queue.min}/
                  {typeof q.frames_in_queue.avg === 'number' ? q.frames_in_queue.avg.toFixed(2) : q.frames_in_queue.avg}/{q.frames_in_queue.max}
                {:else}
                  -
                {/if}
              </td>
              <td
                class:pts-delta-cell={$selectedQueueName &&
                  q.name !== $selectedQueueName &&
                  refPts != null &&
                  parsePts(q.last_ts_seconds) != null}
              >
                {#if $selectedQueueName && q.name !== $selectedQueueName && refPts != null}
                  {@const dq = parsePts(q.last_ts_seconds)}
                  {#if dq != null}
                    <span class="pts-delta">Δ = {formatDelta(dq - refPts)}</span>
                  {:else}
                    {q.last_ts_seconds}
                  {/if}
                {:else}
                  {q.last_ts_seconds}
                {/if}
              </td>
              <td>{q.timebase_num}/{q.timebase_den}</td>
              <td>{typeof q.pps === 'number' ? q.pps.toFixed(1) : ''}</td>
              <td>
                {q.enqueued_total ?? ''}/{q.dequeued_total ?? ''}/{q.dropped_total ?? ''}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    {:else}
      <pre class="small-text">{c.queuesText}</pre>
    {/if}
  </div>
</div>
