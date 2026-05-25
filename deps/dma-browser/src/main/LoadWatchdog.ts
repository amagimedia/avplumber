export interface LoadWatchdogOptions {
  readonly timeoutMs: number;
  readonly onTimeout: () => void;
}

/**
 * Triggers `onTimeout()` once if `clear()` is not called within `timeoutMs`.
 * Used by ManagedWindow to detect stalled page loads and hard-reload.
 */
export class LoadWatchdog {
  private readonly opts: LoadWatchdogOptions;
  private timer: NodeJS.Timeout | null = null;
  private fired = false;

  constructor(opts: LoadWatchdogOptions) {
    this.opts = opts;
  }

  public start(): void {
    this.clear();
    this.fired = false;
    this.timer = setTimeout(() => {
      this.timer = null;
      this.fired = true;
      try {
        this.opts.onTimeout();
      } catch {
        // ignore
      }
    }, this.opts.timeoutMs);
  }

  public clear(): void {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  public hasFired(): boolean {
    return this.fired;
  }
}
