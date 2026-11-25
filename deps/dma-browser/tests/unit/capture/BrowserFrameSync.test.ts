import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  BrowserFrameSync,
  type BrowserSyncClock,
} from '../../../src/main/capture/BrowserFrameSync';
import { alignFrameGrid, frameIntervalNs } from '../../../src/main/capture/FramePhase';

class HostClock implements BrowserSyncClock {
  public mono: bigint;
  public wallOffset: bigint;
  constructor(boot: bigint, wall: bigint) {
    this.mono = boot;
    this.wallOffset = wall - boot;
  }
  public nowNs(): bigint {
    this.mono += 1000n;
    return this.mono;
  }
  public unixMs(): number {
    return Number((this.mono + this.wallOffset) / 1_000_000n);
  }
  public async sleep(ms: number): Promise<void> {
    this.mono += BigInt(ms) * 1_000_000n;
  }
}

const EPOCH = 1_800_000_000_000;
const config = { epochMs: EPOCH, phase: 0.25, loopAnimations: false };

function host(boot: bigint, fps: number) {
  const clock = new HostClock(boot, BigInt(EPOCH + 105) * 1_000_000n);
  const calls: { fps: number; wall: bigint }[] = [];
  const target = {
    setFrameRate: (rate: number): void => {
      calls.push({ fps: rate, wall: clock.mono + clock.wallOffset });
    },
  };
  const report = vi.fn();
  return {
    clock,
    calls,
    target,
    report,
    sync: new BrowserFrameSync(target, fps, config, report, clock),
  };
}

afterEach(() => vi.useRealTimers());

describe('BrowserFrameSync', () => {
  it.each([1, 25, 30, 50, 60, 240])(
    'aligns different boot clocks to one wall grid at %i fps',
    async (fps) => {
      vi.useFakeTimers();
      const a = host(103_000_000n, fps);
      const b = host(900_000_000_000n, fps);
      a.sync.start();
      b.sync.start();
      await vi.advanceTimersByTimeAsync(0);
      const period = frameIntervalNs(fps);
      for (const h of [a, b]) {
        expect(h.sync.snapshot().applications).toBe(1);
        const last = h.calls.at(-1)!;
        const phase = (last.wall - BigInt(EPOCH) * 1_000_000n) % period;
        expect(Math.abs(Number(phase) - Number(period) * config.phase)).toBeLessThan(1_100_000);
        expect(h.calls.map((call) => call.fps)).toEqual([fps === 1 ? 2 : fps - 1, fps]);
        h.sync.stop();
      }
      expect(Math.abs(Number(a.calls.at(-1)!.wall - b.calls.at(-1)!.wall))).toBeLessThan(1_100_000);
    },
  );

  it('leaves a stable cadence alone, corrects clock steps and realigns after navigation', async () => {
    vi.useFakeTimers();
    const h = host(103_000_000n, 30);
    h.sync.start();
    await vi.advanceTimersByTimeAsync(10_000);
    expect(h.sync.snapshot().applications).toBe(1);
    h.clock.wallOffset += 20_000_000n;
    await vi.advanceTimersByTimeAsync(5000);
    expect(h.sync.snapshot().applications).toBe(2);
    h.sync.realign();
    await vi.advanceTimersByTimeAsync(0);
    expect(h.sync.snapshot().applications).toBe(3);
    h.sync.stop();
    await vi.advanceTimersByTimeAsync(10_000);
    expect(h.sync.snapshot().applications).toBe(3);
  });

  it('cancels an alignment waiting for a frame boundary on close', async () => {
    vi.useFakeTimers();
    const h = host(103_000_000n, 1);
    let wake: (() => void) | undefined;
    h.clock.sleep = () =>
      new Promise<void>((resolve) => {
        wake = resolve;
      });
    h.sync.start();
    expect(wake).toBeDefined();
    h.sync.stop();
    wake!();
    await vi.advanceTimersByTimeAsync(0);
    expect(h.calls).toEqual([]);
    expect(h.sync.snapshot().failures).toBe(0);
  });

  it('reports errors and retries instead of leaving an unhandled rejection', async () => {
    vi.useFakeTimers();
    const h = host(103_000_000n, 25);
    h.target.setFrameRate = () => {
      throw new Error('renderer gone');
    };
    h.sync.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(h.sync.snapshot().failures).toBe(1);
    expect(h.report).toHaveBeenCalledWith(expect.stringContaining('renderer gone'));
    await vi.advanceTimersByTimeAsync(5000);
    expect(h.sync.snapshot().failures).toBe(2);
    h.sync.stop();
  });

  it('retries a failed navigation alignment even after an earlier success', async () => {
    vi.useFakeTimers();
    const h = host(103_000_000n, 25);
    h.sync.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(h.sync.snapshot().applications).toBe(1);
    const original = h.target.setFrameRate;
    h.target.setFrameRate = () => {
      throw new Error('compositor not ready');
    };
    h.sync.realign();
    await vi.advanceTimersByTimeAsync(0);
    expect(h.sync.snapshot().failures).toBe(1);
    h.target.setFrameRate = original;
    await vi.advanceTimersByTimeAsync(5000);
    expect(h.sync.snapshot().applications).toBe(2);
    h.sync.stop();
  });
});

it('aligns on the same grid before a future epoch', async () => {
  const clock = new HostClock(100_000_000n, 0n);
  const calls: bigint[] = [];
  await alignFrameGrid(
    { setFrameRate: () => calls.push(clock.mono) },
    25,
    0,
    1_000_000_000n,
    clock,
  );
  expect(calls.at(-1)! % 40_000_000n).toBeLessThan(50_000n);
});
