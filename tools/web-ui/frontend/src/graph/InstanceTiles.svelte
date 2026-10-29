<script>
  // Instance health at a glance: nodes, flowing queues, frames lost, queues near full.
  import './graph.css';
  import { graphModel, summarizeInstance } from './model.mjs';
  import { compact, pctText } from './format.mjs';

  export let nodes = [];
  export let queues = [];

  $: s = summarizeInstance(graphModel(nodes, queues));
</script>

<div class="gv-tiles">
  <div class="gv-tile hero"><b>{s.nodes.toLocaleString()}</b><span>nodes{s.off ? `, ${s.off} OFF` : ''}</span></div>
  <div class="gv-tile"><b>{s.flowing.toLocaleString()}</b><span>flowing of {s.queues.toLocaleString()} queues</span></div>
  <div class="gv-tile {s.level}" data-tip="Green under 0.5% of frames dropped, orange from 0.5%, red over 1%: since start, or by the queue losing the most since the previous sample ({s.droppingNow} queues dropping now). Compositor inputs skipping to the newest frame ({s.skipped.toLocaleString()} skips) is by design and not counted.">
    <b>{pctText(s.share)}</b><span>frames dropped since start</span>
    <small>{s.dropped.toLocaleString()} of {compact(s.moved)} passed between nodes{s.dropQueues ? `, in ${s.dropQueues} queues` : ''}</small>
  </div>
  <div class="gv-tile"><b>{s.full}</b><span>queues near full</span><small>avg ≥ 90% of capacity</small></div>
</div>
