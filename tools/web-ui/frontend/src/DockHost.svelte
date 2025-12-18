<script>
  import { onDestroy, onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import { GoldenLayout } from 'golden-layout';

  /**
   * Generic Golden Layout host for Svelte panels.
   *
   * Usage:
   * - Pass `registry`: { [componentType: string]: SvelteComponentConstructor }
   * - Pass `layoutConfig` (default layout) and optional `storageKey` to persist.
   * - Pass `ctx` (any) to forward shared props/state to every panel.
   */
  export let registry = {};
  export let ctx = {};
  export let layoutConfig = null;
  export let storageKey = 'avplumber.webui.dock.layout.v1';

  // Panels are mounted once by Golden Layout; to keep them reactive to App state,
  // we pass a store and update it whenever `ctx` changes.
  const ctxStore = writable(ctx);
  $: ctxStore.set(ctx);

  /** @type {HTMLElement | null} */
  let hostEl = null;

  /** @type {any} */
  let layout = null;

  /** @type {Map<any, any>} */
  const componentByContainer = new Map();

  let saveTimer = /** @type {any} */ (null);

  function safeJsonParse(text) {
    try {
      return JSON.parse(text);
    } catch {
      return null;
    }
  }

  function getSavedLayout() {
    if (!storageKey) return null;
    try {
      const raw = localStorage.getItem(storageKey);
      if (!raw) return null;
      const parsed = safeJsonParse(raw);
      return parsed && typeof parsed === 'object' ? parsed : null;
    } catch {
      return null;
    }
  }

  function saveLayoutDebounced() {
    if (!layout || !storageKey) return;
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      try {
        const cfg = layout.saveLayout();
        localStorage.setItem(storageKey, JSON.stringify(cfg));
      } catch (_) {
        // ignore persistence failures (private mode, quota, etc.)
      }
    }, 150);
  }

  function applyLayout(cfg) {
    if (!layout) return;
    const c = cfg || layoutConfig;
    if (!c) return;
    layout.loadLayout(c);
  }

  export function resetLayout() {
    try {
      if (storageKey) localStorage.removeItem(storageKey);
    } catch (_) {
      // ignore
    }
    applyLayout(layoutConfig);
  }

  function setSizeFromHost() {
    if (!layout || !hostEl) return;
    const rect = hostEl.getBoundingClientRect();
    const w = Math.max(0, Math.floor(rect.width));
    const h = Math.max(0, Math.floor(rect.height));
    layout.setSize(w, h);
  }

  onMount(() => {
    if (!hostEl) return;

    // Bind/unbind mounting of Svelte components into Golden Layout containers.
    const bind = (container, itemConfig) => {
      const type = itemConfig && itemConfig.componentType;
      const Ctor = type && typeof type === 'string' ? registry[type] : null;
      if (!Ctor) {
        const el = document.createElement('pre');
        el.style.margin = '0';
        el.textContent = `Unknown panel type: ${String(type)}`;
        container.element.appendChild(el);
        return { component: { destroy: () => el.remove() }, virtual: false };
      }

      // Ensure container is empty and ready.
      container.element.innerHTML = '';

      const inst = new Ctor({
        target: container.element,
        props: {
          ctx: ctxStore,
          dock: {
            container,
            layout,
            saveLayout: saveLayoutDebounced
          },
          state: itemConfig && itemConfig.componentState ? itemConfig.componentState : {}
        }
      });

      componentByContainer.set(container, inst);
      return { component: inst, virtual: false };
    };

    const unbind = (container) => {
      const inst = componentByContainer.get(container);
      componentByContainer.delete(container);
      try {
        if (inst && typeof inst.$destroy === 'function') inst.$destroy();
      } catch (_) {
        // ignore
      }
      try {
        container.element.innerHTML = '';
      } catch (_) {
        // ignore
      }
    };

    layout = new GoldenLayout(hostEl, bind, unbind);

    // Keep sizing correct (initial + resizes).
    const ro = new ResizeObserver(() => setSizeFromHost());
    ro.observe(hostEl);
    setSizeFromHost();

    // Persist on layout changes.
    layout.on('stateChanged', saveLayoutDebounced);

    // Load saved layout if present, else default.
    const saved = getSavedLayout();
    applyLayout(saved || layoutConfig);

    return () => {
      try {
        ro.disconnect();
      } catch (_) {
        // ignore
      }
    };
  });

  onDestroy(() => {
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = null;
    try {
      if (layout) layout.destroy();
    } catch (_) {
      // ignore
    }
    layout = null;
    componentByContainer.clear();
  });
</script>

<div class="dock-host" bind:this={hostEl} />


