import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { isPageLoadFailure, PageReloader } from '../../src/main/PageReloader';

describe('isPageLoadFailure', () => {
  it('reloads only for main-frame failures that were not aborted', () => {
    expect(isPageLoadFailure(-105, true)).toBe(true);
    expect(isPageLoadFailure(-105, false)).toBe(false);
    expect(isPageLoadFailure(-3, true)).toBe(false);
  });
});

describe('PageReloader', () => {
  let reload: ReturnType<typeof vi.fn>;
  let reloader: PageReloader;
  beforeEach(() => {
    vi.useFakeTimers();
    reload = vi.fn();
    reloader = new PageReloader({ minDelayMs: 1000, maxDelayMs: 4000, reload });
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('doubles the delay up to the maximum while reloads keep failing', () => {
    const delays: (number | null)[] = [];
    for (let i = 0; i < 4; i++) {
      const delay = reloader.failed();
      delays.push(delay);
      vi.advanceTimersByTime(delay ?? 0);
    }
    expect(delays).toEqual([1000, 2000, 4000, 4000]);
    expect(reload).toHaveBeenCalledTimes(4);
  });

  it('coalesces failures while a reload is pending', () => {
    expect(reloader.failed()).toBe(1000);
    expect(reloader.failed()).toBeNull();
    vi.advanceTimersByTime(10_000);
    expect(reload).toHaveBeenCalledTimes(1);
  });

  it('keeps backing off when only the error page finishes loading', () => {
    reloader.failed();
    reloader.loaded();
    vi.advanceTimersByTime(1000);
    expect(reloader.failed()).toBe(2000);
  });

  it('restarts the backoff after a reload loads cleanly', () => {
    reloader.failed();
    vi.advanceTimersByTime(1000);
    reloader.failed();
    vi.advanceTimersByTime(2000);
    reloader.loaded();
    expect(reloader.failed()).toBe(1000);
  });

  it('reset() cancels a pending reload', () => {
    reloader.failed();
    reloader.failed();
    reloader.reset();
    vi.advanceTimersByTime(10_000);
    expect(reload).not.toHaveBeenCalled();
    expect(reloader.failed()).toBe(1000);
  });
});
