<script>
  // Instance health at a glance: nodes, flowing queues, frames lost, queues near full.
  import './graph.css';
  import { graphModel, summarizeInstance } from './model.mjs';

  export let nodes = [];
  export let queues = [];

  $: s = summarizeInstance(graphModel(nodes, queues));
</script>

<div class="gv-tiles">
  <div class="gv-tile hero"><b>{s.nodes.toLocaleString()}</b><span>nodes{s.off ? `, ${s.off} OFF` : ''}</span></div>
  <div class="gv-tile"><b>{s.flowing.toLocaleString()}</b><span>flowing of {s.queues.toLocaleString()} queues</span></div>
  <div class="gv-tile" class:bad={s.droppingNow} data-tip="Red while frames are dropped now: in {s.droppingNow} queues since the previous sample">
    <b>{s.dropped.toLocaleString()}</b><span>frames dropped since start</span>
    <small>in {s.dropQueues} queues; {s.skipped.toLocaleString()} more skipped at compositor inputs, as designed</small>
  </div>
  <div class="gv-tile"><b>{s.full}</b><span>queues near full</span><small>avg ≥ 90% of capacity</small></div>
</div>
