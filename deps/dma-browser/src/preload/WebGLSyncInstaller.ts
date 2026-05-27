/**
 * WebGL offscreen / sharedTexture flicker mitigation.
 *
 * Some pages render WebGL into a canvas that isn't full-screen, and Chromium
 * sometimes puts the canvas on a separate overlay/compositor path. Offscreen
 * capture can then intermittently miss it, showing as a black flicker over
 * the WebGL region.
 *
 * Mitigations (renderer-side, GPU stays on):
 * - Force the canvas to be "non-overlayable" via CSS that requires compositing.
 * - Insert a GPU sync after each animation frame (WebGL2 fenceSync if
 *   available, else finish()).
 *
 * Port of `preload.js:1-122` from the source repo.
 */
type AnyGL = WebGLRenderingContext | WebGL2RenderingContext;

export class WebGLSyncInstaller {
  private readonly glContexts = new Set<AnyGL>();
  private rafWrapped = false;
  private styleInstalled = false;

  public install(): void {
    const proto = HTMLCanvasElement.prototype;
    const origGetContext = proto.getContext.bind(proto) as (
      this: HTMLCanvasElement,
      type: string,
      attrs?: unknown,
    ) => unknown;
    if (typeof origGetContext !== 'function') return;

    const trackContext = (canvas: HTMLCanvasElement, type: string, ctx: unknown): void => {
      if (!ctx || !this.isGl(ctx)) return;
      const t = type.toLowerCase();
      if (t !== 'webgl' && t !== 'experimental-webgl' && t !== 'webgl2') return;
      this.glContexts.add(ctx);
      this.markCanvas(canvas);
      this.ensureRafWrapped();
    };

    proto.getContext = function (this: HTMLCanvasElement, type: string, attrs?: unknown): unknown {
      const ctx = origGetContext.call(this, type, attrs);
      try {
        trackContext(this, type, ctx);
      } catch {
        // ignore
      }
      return ctx;
    } as typeof proto.getContext;
  }

  private isGl(ctx: unknown): ctx is AnyGL {
    return typeof ctx === 'object' && ctx !== null && 'drawArrays' in ctx;
  }

  private ensureStyleInstalled(): void {
    if (this.styleInstalled) return;
    this.styleInstalled = true;
    try {
      const st = document.createElement('style');
      st.textContent =
        'canvas.__dma_webgl_canvas{' +
        'filter:opacity(0.999999)!important;' +
        'transform:translateZ(0)!important;' +
        'will-change:transform!important;' +
        'backface-visibility:hidden!important;' +
        '}';
      const parent = document.head ?? document.documentElement ?? document;
      parent.appendChild(st);
    } catch {
      // ignore
    }
  }

  private markCanvas(canvas: HTMLCanvasElement): void {
    try {
      this.ensureStyleInstalled();
      canvas.classList.add('__dma_webgl_canvas');
    } catch {
      // ignore
    }
  }

  private ensureRafWrapped(): void {
    if (this.rafWrapped) return;
    this.rafWrapped = true;
    const origRaf = window.requestAnimationFrame.bind(window);
    const contexts = this.glContexts;
    const syncAll = (): void => {
      for (const gl of contexts) {
        try {
          this.syncOne(gl);
        } catch {
          // ignore
        }
      }
    };
    window.requestAnimationFrame = (cb: FrameRequestCallback): number => {
      return origRaf((ts) => {
        try {
          cb(ts);
        } catch {
          // user callback errors are not ours
        }
        syncAll();
      });
    };
  }

  private syncOne(gl: AnyGL): void {
    const gl2 = gl as WebGL2RenderingContext;
    if (typeof gl2.fenceSync === 'function' && typeof gl2.clientWaitSync === 'function') {
      let sync: WebGLSync | null = null;
      try {
        sync = gl2.fenceSync(gl2.SYNC_GPU_COMMANDS_COMPLETE, 0);
      } catch {
        // fall through to finish()
      }
      if (!sync) {
        try {
          gl.finish();
        } catch {
          // ignore
        }
        return;
      }
      try {
        gl2.flush();
      } catch {
        // ignore
      }
      try {
        const t0 = performance.now();
        let status = gl2.clientWaitSync(sync, 0, 0);
        while (status === gl2.TIMEOUT_EXPIRED) {
          if (performance.now() - t0 > 10) break;
          status = gl2.clientWaitSync(sync, 0, 0);
        }
        if (status === gl2.TIMEOUT_EXPIRED) gl.finish();
      } catch {
        try {
          gl.finish();
        } catch {
          // ignore
        }
      } finally {
        try {
          gl2.deleteSync(sync);
        } catch {
          // ignore
        }
      }
      return;
    }
    try {
      gl.finish();
    } catch {
      // ignore
    }
  }
}
