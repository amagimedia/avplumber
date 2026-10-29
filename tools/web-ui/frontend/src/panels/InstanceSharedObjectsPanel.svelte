<script>
  export let ctx; // store
  export let dock;
  export let state;

  $: c = $ctx;
  $: dock;
  $: state;
  const yes = (v) => (v ? 'Yes' : 'No');
  const seconds = (v) => (typeof v === 'number' ? v.toFixed(6) : '–');
</script>

<div class="panel">
  <div class="toolbar">
    <label class="check" data-tip="Ask for both lists every second">
      <input type="checkbox" checked={c.autoRefreshISOs} on:change={(e) => c.setAutoRefreshISOs?.(e.target.checked)} /> Auto-refresh
    </label>
    <button class="icon-btn small push" on:click={() => { c.refreshSyncGroups?.(); c.refreshCorrectionGroups?.(); }}>Refresh</button>
  </div>

  <div class="body flush">
    <div class="sub-h" style="margin: 10px 12px 6px">Sync groups · realtime teams</div>
    {#if c.syncGroups && c.syncGroups.length > 0}
      <div style="overflow-x: auto">
        <table class="mini-tbl">
          <thead><tr><th>Name</th><th>Offset</th><th>Timebase</th><th>Flushing</th><th>First</th><th>Seek targets</th><th>Linked teams</th></tr></thead>
          <tbody>
            {#each c.syncGroups as sg (sg.name)}
              <tr>
                <td><code>{sg.name}</code></td>
                <td>{sg.offset != null ? sg.offset : 'NOPTS'}</td>
                <td>{sg.timebase_num}/{sg.timebase_den}</td>
                <td>{yes(sg.flushing)}</td>
                <td>{yes(sg.first)}</td>
                <td>{sg.seek_targets_count}</td>
                <td>{sg.linked_teams_count}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {:else}
      <p class="hint" style="margin: 0 12px">None. <code>realtime</code> nodes with a <code>team</code> parameter create sync groups.</p>
    {/if}

    <div class="sub-h" style="margin: 16px 12px 6px">Sentinel correction groups</div>
    {#if c.correctionGroups && c.correctionGroups.length > 0}
      <div style="overflow-x: auto">
        <table class="mini-tbl">
          <thead><tr><th>Name</th><th>Timeshift (s)</th><th>Locked</th><th>Timebase</th><th>Clock (s)</th><th>Start TS (s)</th><th>Reporting</th></tr></thead>
          <tbody>
            {#each c.correctionGroups as cg (cg.name)}
              <tr>
                <td><code>{cg.name}</code></td>
                <td>{cg.has_timeshift ? seconds(cg.timeshift_seconds) : '–'}</td>
                <td>{yes(cg.timeshift_locked)}</td>
                <td>{cg.timebase_num}/{cg.timebase_den}</td>
                <td>{cg.has_clock ? seconds(cg.clock_seconds) : '–'}</td>
                <td>{seconds(cg.start_ts_seconds)}</td>
                <td>{yes(cg.reporting)}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
      <details style="margin: 10px 12px 12px">
        <summary><b>All fields</b></summary>
        {#each c.correctionGroups as cg (cg.name)}
          <div class="obj"><div class="sub-h">{cg.name}</div><pre class="wrap">{JSON.stringify(cg, null, 2)}</pre></div>
        {/each}
      </details>
    {:else}
      <p class="hint" style="margin: 0 12px 12px"><code>sentinel</code> nodes with a <code>correction_group</code> parameter create correction groups.</p>
    {/if}
  </div>
</div>
