<script>
  import { queueStatsByName } from './graphStores';
  export let name;
  export let showFill = false;
  $: queue = showFill ? $queueStatsByName.get(name) : null;
  $: fill = queue?.capacity > 0
    ? ` ${Math.round(Math.max(0, Math.min(100, queue.occupied / queue.capacity * 100)))}%` : '';
</script>

<!-- Statistics update only this text, never Rete nodes or their sockets. -->
{name}{fill}
