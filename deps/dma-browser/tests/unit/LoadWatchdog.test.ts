import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { LoadWatchdog } from '../../src/main/LoadWatchdog';

describe('LoadWatchdog', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('fires after the timeout when not cleared', () => {
    const onTimeout = vi.fn();
    const wd = new LoadWatchdog({ timeoutMs: 1000, onTimeout });
    wd.start();
    vi.advanceTimersByTime(999);
    expect(onTimeout).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(onTimeout).toHaveBeenCalledTimes(1);
    expect(wd.hasFired()).toBe(true);
  });

  it('does not fire after clear()', () => {
    const onTimeout = vi.fn();
    const wd = new LoadWatchdog({ timeoutMs: 1000, onTimeout });
    wd.start();
    vi.advanceTimersByTime(500);
    wd.clear();
    vi.advanceTimersByTime(10_000);
    expect(onTimeout).not.toHaveBeenCalled();
    expect(wd.hasFired()).toBe(false);
  });

  it('restarts the timer on re-start()', () => {
    const onTimeout = vi.fn();
    const wd = new LoadWatchdog({ timeoutMs: 1000, onTimeout });
    wd.start();
    vi.advanceTimersByTime(900);
    wd.start();
    vi.advanceTimersByTime(900);
    expect(onTimeout).not.toHaveBeenCalled();
    vi.advanceTimersByTime(100);
    expect(onTimeout).toHaveBeenCalledTimes(1);
  });

  it('swallows exceptions from onTimeout', () => {
    const wd = new LoadWatchdog({
      timeoutMs: 100,
      onTimeout: () => {
        throw new Error('boom');
      },
    });
    wd.start();
    expect(() => vi.advanceTimersByTime(100)).not.toThrow();
  });
});
