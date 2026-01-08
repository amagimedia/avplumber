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
    <button on:click={() => c.refreshSyncGroups?.()}>Refresh Sync Groups</button>
    <button on:click={() => c.refreshCorrectionGroups?.()}>Refresh Correction Groups</button>
    <button on:click={() => {
      c.refreshSyncGroups?.();
      c.refreshCorrectionGroups?.();
    }}>Refresh All</button>
    <label style="margin-left: auto; display: flex; align-items: center; gap: 0.25rem; font-size: 0.85rem;">
      <input type="checkbox" checked={c.autoRefreshISOs} on:change={(e) => c.setAutoRefreshISOs?.(e.target.checked)} />
      Auto-refresh
    </label>
  </div>

  <div class="scrollable small-text iso-scrollable">
    <h3>Sync Groups (RealTimeTeam)</h3>
    {#if c.syncGroups && c.syncGroups.length > 0}
      <table class="node-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Offset</th>
            <th>Timebase</th>
            <th>Flushing</th>
            <th>First</th>
            <th>Seek Targets</th>
            <th>Linked Teams</th>
          </tr>
        </thead>
        <tbody>
          {#each c.syncGroups as sg}
            <tr>
              <td><code>{sg.name}</code></td>
              <td>{sg.offset != null ? sg.offset : 'NOPTS'}</td>
              <td>{sg.timebase_num}/{sg.timebase_den}</td>
              <td>{sg.flushing ? 'Yes' : 'No'}</td>
              <td>{sg.first ? 'Yes' : 'No'}</td>
              <td>{sg.seek_targets_count}</td>
              <td>{sg.linked_teams_count}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    {:else}
      <p class="hint">No sync groups found. Use <code>realtime</code> nodes with a <code>team</code> parameter to create sync groups.</p>
    {/if}

    <h3 style="margin-top: 2em;">Sentinel Correction Groups (PTSCorrectorCommon)</h3>
    {#if c.correctionGroups && c.correctionGroups.length > 0}
      <table class="node-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Has Timeshift</th>
            <th>Timeshift (s)</th>
            <th>Timeshift Locked</th>
            <th>Timebase</th>
            <th>Has Clock</th>
            <th>Clock (s)</th>
            <th>Start TS (s)</th>
            <th>Reporting</th>
          </tr>
        </thead>
        <tbody>
          {#each c.correctionGroups as cg}
            <tr>
              <td><code>{cg.name}</code></td>
              <td>{cg.has_timeshift ? 'Yes' : 'No'}</td>
              <td>
                {#if cg.has_timeshift && typeof cg.timeshift_seconds === 'number'}
                  {cg.timeshift_seconds.toFixed(6)}
                {:else}
                  -
                {/if}
              </td>
              <td>{cg.timeshift_locked ? 'Yes' : 'No'}</td>
              <td>{cg.timebase_num}/{cg.timebase_den}</td>
              <td>{cg.has_clock ? 'Yes' : 'No'}</td>
              <td>
                {#if cg.has_clock && typeof cg.clock_seconds === 'number'}
                  {cg.clock_seconds.toFixed(6)}
                {:else}
                  -
                {/if}
              </td>
              <td>
                {#if typeof cg.start_ts_seconds === 'number'}
                  {cg.start_ts_seconds.toFixed(6)}
                {:else}
                  -
                {/if}
              </td>
              <td>{cg.reporting ? 'Yes' : 'No'}</td>
            </tr>
          {/each}
        </tbody>
      </table>

      <details style="margin-top: 1em;">
        <summary>Detailed Statistics</summary>
        <div style="margin-top: 0.5em;">
          {#each c.correctionGroups as cg}
            <div style="margin-bottom: 1em; padding: 0.5em; background: #f5f5f5; border-radius: 4px;">
              <h4 style="margin: 0 0 0.5em 0;"><code>{cg.name}</code></h4>
              <pre class="small-text" style="margin: 0; white-space: pre-wrap;">{JSON.stringify(cg, null, 2)}</pre>
            </div>
          {/each}
        </div>
      </details>
    {:else}
      <p class="hint">No correction groups found. Use <code>sentinel</code> nodes with a <code>correction_group</code> parameter to create correction groups.</p>
    {/if}
  </div>
</div>

<style>
  h3 {
    margin-top: 1em;
    margin-bottom: 0.5em;
    font-size: 1.1em;
    font-weight: bold;
  }
  h4 {
    font-size: 1em;
    font-weight: bold;
  }
  .hint {
    color: #9ca3af;
    font-style: italic;
    padding: 1em;
  }
  code {
    background: #1f2937;
    color: #e5e7eb;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: 0.9em;
    border: 1px solid #374151;
  }
  .iso-scrollable {
    min-height: 300px;
    max-height: none !important;
    flex: 1 1 auto;
  }
  .toolbar label {
    color: #9ca3af;
  }
  .toolbar input[type="checkbox"] {
    cursor: pointer;
  }
</style>

