<script>
  import { onDestroy, onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import { GoldenLayout, LayoutConfig } from 'golden-layout';

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
  let isDragging = false;
  let isApplyingLayout = false;
  let stopAuxClickCloseHandler = null;

  function safeJsonParse(text) {
    try {
      return JSON.parse(text);
    } catch {
      return null;
    }
  }

  function validateLayout(cfg) {
    if (!cfg || typeof cfg !== 'object') return false;
    // Check if layout has a root structure
    if (!cfg.root || typeof cfg.root !== 'object') return false;
    // Basic validation - ensure it's a valid layout structure
    return true;
  }

  function coerceToLayoutConfig(cfg) {
    if (!cfg || typeof cfg !== 'object') return cfg;
    // Back-compat: older versions of this UI persisted ResolvedLayoutConfig (resolved:true)
    // but loadLayout expects LayoutConfig.
    if (cfg.resolved === true) {
      try {
        return LayoutConfig.fromResolved(cfg);
      } catch (_) {
        return null;
      }
    }
    return cfg;
  }

  function normalizeLayoutSize(item) {
    if (!item || typeof item !== 'object' || Array.isArray(item)) return item;
    
    // Normalize size property for row/column items. Golden Layout supports `%` and `fr`.
    if ('size' in item) {
      const sizeValue = item.size;
      if (sizeValue == null) {
        // null or undefined - remove it
        delete item.size;
      } else if (typeof sizeValue === 'number') {
        // Should always be a string in LayoutConfig. Remove invalid values.
        delete item.size;
      } else if (typeof sizeValue === 'string') {
        const sizeStr = sizeValue.trim();
        if (!sizeStr) {
          delete item.size;
        } else if (/%$/.test(sizeStr) || /fr$/.test(sizeStr)) {
          // Keep supported units
          item.size = sizeStr;
        } else if (sizeStr.includes('px')) {
          // Not supported for ItemConfig.size
          delete item.size;
        } else {
          // Unknown unit - drop it
          delete item.size;
        }
      } else {
        // Any other type - remove it
        delete item.size;
      }
    }
    
    // Recursively normalize content items
    if (Array.isArray(item.content)) {
      item.content = item.content.map(normalizeLayoutSize);
    }
    
    return item;
  }

  function normalizeClosable(item) {
    if (!item || typeof item !== 'object' || Array.isArray(item)) return item;

    // Golden Layout cancels dragging a tab out of a stack when the stack is not "closable"
    // and there is only a single tab. To keep dragging working, we force internal closability
    // while separately preventing actual close UI/actions in our app.
    if ('isClosable' in item) {
      item.isClosable = true;
    }

    if (item.header && typeof item.header === 'object') {
      // Ensure close is enabled so Header considers the stack closable.
      if (item.header.close === false || item.header.close == null) {
        item.header.close = 'Close';
      }
    }

    if (Array.isArray(item.content)) {
      item.content = item.content.map(normalizeClosable);
    }

    return item;
  }

  function normalizeLayout(cfg) {
    if (!cfg || typeof cfg !== 'object') return cfg;
    
    // Deep clone to avoid mutating the original
    const normalized = JSON.parse(JSON.stringify(cfg));
    
    // Ensure settings.hasHeaders is true to allow dragging single tabs
    if (normalized.settings) {
      if (normalized.settings.hasHeaders === false) {
        normalized.settings.hasHeaders = true;
      }
    } else {
      normalized.settings = { hasHeaders: true };
    }

    // Ensure header close is enabled (needed for single-tab drag in Golden Layout)
    if (normalized.header && typeof normalized.header === 'object') {
      if (normalized.header.close === false || normalized.header.close == null) {
        normalized.header.close = 'Close';
      }
    } else {
      normalized.header = { close: 'Close' };
    }
    
    // Normalize root and all nested items
    if (normalized.root) {
      normalized.root = normalizeLayoutSize(normalized.root);
      normalized.root = normalizeClosable(normalized.root);
    }
    
    return normalized;
  }

  function getSavedLayout() {
    if (!storageKey) return null;
    try {
      const raw = localStorage.getItem(storageKey);
      if (!raw) return null;
      let parsed = safeJsonParse(raw);
      parsed = coerceToLayoutConfig(parsed);
      if (!validateLayout(parsed)) return null;
      return parsed;
    } catch {
      return null;
    }
  }

  function saveLayoutDebounced() {
    if (!layout || !storageKey || isDragging || isApplyingLayout) return;
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      if (isDragging || isApplyingLayout) return;
      try {
        // saveLayout() returns ResolvedLayoutConfig. Convert it to LayoutConfig before persisting,
        // otherwise loadLayout() will crash trying to parse numbers as strings.
        const resolved = layout.saveLayout();
        const cfg = LayoutConfig.fromResolved(resolved);
        const normalized = normalizeLayout(cfg);
        if (normalized && validateLayout(normalized)) {
          localStorage.setItem(storageKey, JSON.stringify(normalized));
        }
      } catch (_) {
        // ignore persistence failures (private mode, quota, etc.)
      }
    }, 150);
  }

  function applyLayout(cfg) {
    if (!layout) return;
    const c0 = cfg || layoutConfig;
    const c = coerceToLayoutConfig(c0) || layoutConfig;
    if (!c) return;
    isApplyingLayout = true;
    try {
      // Normalize the layout config to ensure size values are strings
      const normalized = normalizeLayout(c);
      layout.loadLayout(normalized);
      // Ensure layout size is updated after loading
      setTimeout(() => {
        setSizeFromHost();
        isApplyingLayout = false;
      }, 50);
    } catch (e) {
      isApplyingLayout = false;
      console.warn('Failed to apply layout:', e);
      // If saved layout fails, try default
      if (cfg !== layoutConfig && layoutConfig) {
        try {
          const normalizedDefault = normalizeLayout(layoutConfig);
          layout.loadLayout(normalizedDefault);
          setTimeout(() => setSizeFromHost(), 50);
        } catch (_) {
          // ignore
        }
      }
    }
  }

  export function resetLayout() {
    try {
      if (storageKey) localStorage.removeItem(storageKey);
    } catch (_) {
      // ignore
    }
    isDragging = false;
    if (saveTimer) {
      clearTimeout(saveTimer);
      saveTimer = null;
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

    // Track drag operations using Golden Layout's official events.
    layout.on('dragStart', () => {
      isDragging = true;
    });
    layout.on('dragStop', () => {
      isDragging = false;
      setTimeout(() => {
        setSizeFromHost();
        saveLayoutDebounced();
      }, 10);
    });

    // Prevent middle-click closing tabs once we force isClosable=true for drag support.
    // Golden Layout's Tab listens to 'click' and checks event.button (browser-dependent),
    // and browsers also emit 'auxclick' for middle button.
    stopAuxClickCloseHandler = (e) => {
      const target = e && e.target;
      if (!target) return;
      if (e.button !== 1) return; // only middle
      if (target.closest && target.closest('.lm_tab')) {
        e.preventDefault();
        e.stopImmediatePropagation();
        e.stopPropagation();
      }
    };
    hostEl.addEventListener('auxclick', stopAuxClickCloseHandler, true);
    hostEl.addEventListener('click', stopAuxClickCloseHandler, true);

    // Persist on layout changes, but not during drags
    layout.on('stateChanged', () => {
      if (!isDragging) {
        saveLayoutDebounced();
      }
    });

    // Load saved layout if present, else default.
    const saved = getSavedLayout();
    if (saved) {
      applyLayout(saved);
    } else {
      applyLayout(layoutConfig);
    }

    return () => {
      try {
        ro.disconnect();
      } catch (_) {
        // ignore
      }
      try {
        if (stopAuxClickCloseHandler) {
          hostEl.removeEventListener('auxclick', stopAuxClickCloseHandler, true);
          hostEl.removeEventListener('click', stopAuxClickCloseHandler, true);
        }
      } catch (_) {
        // ignore
      }
    };
  });

  onDestroy(() => {
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = null;
    isDragging = false;
    try {
      if (hostEl && stopAuxClickCloseHandler) {
        hostEl.removeEventListener('auxclick', stopAuxClickCloseHandler, true);
        hostEl.removeEventListener('click', stopAuxClickCloseHandler, true);
      }
    } catch (_) {
      // ignore
    }
    stopAuxClickCloseHandler = null;
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


