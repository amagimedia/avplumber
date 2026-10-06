/** Chromium's net::ERR_ABORTED: the navigation was cancelled or superseded, not failed. */
const ERR_ABORTED = -3;

/** True when a `did-fail-load` leaves the page unusable; sub-frame failures do not. */
export function isPageLoadFailure(errorCode: number, isMainFrame: boolean): boolean {
  return isMainFrame && errorCode !== ERR_ABORTED;
}

export interface PageReloaderOptions {
  readonly minDelayMs: number;
  readonly maxDelayMs: number;
  readonly reload: () => void;
}

/**
 * Reloads a page after load failures and renderer crashes, doubling the delay
 * from `minDelayMs` up to `maxDelayMs`. Failures while a reload is pending
 * coalesce into it. Chromium finishes loading its error page after
 * `did-fail-load`, so `loaded()` resets the backoff only when no failure was
 * reported since the last reload.
 */
export class PageReloader {
  private readonly opts: PageReloaderOptions;
  private timer: NodeJS.Timeout | null = null;
  private attempts = 0;
  private failedSinceReload = false;

  constructor(opts: PageReloaderOptions) {
    this.opts = opts;
  }

  /** Schedules a reload; returns its delay, or null when one is already pending. */
  public failed(): number | null {
    this.failedSinceReload = true;
    if (this.timer) return null;
    const delayMs = Math.min(this.opts.maxDelayMs, this.opts.minDelayMs * 2 ** this.attempts);
    if (delayMs < this.opts.maxDelayMs) this.attempts++;
    this.timer = setTimeout(() => {
      this.timer = null;
      this.failedSinceReload = false;
      this.opts.reload();
    }, delayMs);
    return delayMs;
  }

  /** A clean load (not an error page) restarts the backoff and returns true. */
  public loaded(): boolean {
    if (this.failedSinceReload) return false;
    this.attempts = 0;
    return true;
  }

  /** Cancels a pending reload and restarts the backoff, e.g. on an explicit navigation. */
  public reset(): void {
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    this.attempts = 0;
    this.failedSinceReload = false;
  }
}
