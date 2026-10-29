<script>
  export let total = 0;
  export let page = 0;
  export let pageSize = 400;
  export let label = 'rows';

  $: pages = Math.max(1, Math.ceil(total / pageSize));
  $: if (page >= pages) page = pages - 1;
</script>

<nav class="cap" aria-label={`${label} pagination`}>
  <span>Showing {total ? (page * pageSize + 1).toLocaleString() : 0}–{Math.min((page + 1) * pageSize, total).toLocaleString()}
    of {total.toLocaleString()} matching {label}.</span>
  <button class="icon-btn small" disabled={page === 0} on:click={() => page--}>Previous</button>
  <span>Page {page + 1} of {pages.toLocaleString()}</span>
  <button class="icon-btn small" disabled={page + 1 >= pages} on:click={() => page++}>Next</button>
</nav>
