<script>
  export let ctx; // store
  export let dock;
  export let state;

  $: c = $ctx;
  $: dock;
  $: state;
</script>

<div class="panel avp-panel">
  <div class="toolbar">
    <button on:click={() => c.refreshQueues?.()}>Refresh queues</button>
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
          {#each c.queues as q}
            <tr>
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
              <td>{q.last_ts_seconds}</td>
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


