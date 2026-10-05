import { describe, expect, it } from 'vitest';
import {
  alignFramePhase,
  frameIntervalNs,
  MONOTONIC_CLOCK,
  workerFramePhase,
  type FramePhaseClock,
} from '../../../src/main/capture/FramePhase';

/** A clock that only moves when read or slept on; `lateNs` is how late each sleep wakes, in order. */
class FakeClock implements FramePhaseClock {
  public t: bigint;
  private readonly lateNs: bigint[];
  private readonly alwaysLateNs: bigint;

  constructor(start: bigint, lateNs: bigint[] = [], alwaysLateNs = 0n) {
    this.t = start;
    this.lateNs = [...lateNs];
    this.alwaysLateNs = alwaysLateNs;
  }

  public nowNs(): bigint {
    this.t += 1_000n; // reading the clock takes a microsecond
    return this.t;
  }

  public sleep(ms: number): Promise<void> {
    this.t += BigInt(Math.round(ms * 1e6)) + (this.lateNs.shift() ?? this.alwaysLateNs);
    return Promise.resolve();
  }
}

function recorder(clock: FakeClock): {
  target: { setFrameRate(fps: number): void };
  calls: { fps: number; at: bigint }[];
} {
  const calls: { fps: number; at: bigint }[] = [];
  return { calls, target: { setFrameRate: (fps) => calls.push({ fps, at: clock.t }) } };
}

/** Where in the 60 fps frame period the last setFrameRate call was made. */
function lastCallPhase(calls: { fps: number; at: bigint }[]): bigint {
  const last = calls[calls.length - 1];
  if (!last) throw new Error('setFrameRate was not called');
  return last.at % 16_666_000n;
}

const START = 277_470_376_279_043n;
const TOLERANCE_NS = 50_000n;

describe('frameIntervalNs', () => {
  it('is one second over the rate in whole microseconds, as Electron computes it', () => {
    expect(frameIntervalNs(60)).toBe(16_666_000n);
    expect(frameIntervalNs(50)).toBe(20_000_000n);
    expect(frameIntervalNs(30)).toBe(33_333_000n);
    expect(frameIntervalNs(25)).toBe(40_000_000n);
  });
});

describe('alignFramePhase', () => {
  it('sets the rate when the clock is at the requested phase of the frame period', async () => {
    const clock = new FakeClock(START);
    const { target, calls } = recorder(clock);

    expect(await alignFramePhase(target, 60, 0.4, clock)).toBe(true);

    const phase = lastCallPhase(calls);
    expect(phase).toBeGreaterThanOrEqual(6_666_400n);
    expect(phase).toBeLessThan(6_666_400n + TOLERANCE_NS);
  });

  it('at 60 fps sets another rate first, because Chromium only forwards a timebase after the interval changed', async () => {
    const clock = new FakeClock(START);
    const { target, calls } = recorder(clock);

    await alignFramePhase(target, 60, 0.2, clock);

    expect(calls.map((c) => c.fps)).toEqual([59, 60]);
  });

  it.each([25, 30, 50])('at %i fps sets only the requested rate', async (fps) => {
    const clock = new FakeClock(START);
    const { target, calls } = recorder(clock);

    await alignFramePhase(target, fps, 0.2, clock);

    expect(calls.map((c) => c.fps)).toEqual([fps]);
  });

  it('takes the next period when a timer wakes after the instant has passed', async () => {
    const clock = new FakeClock(START, [5_000_000n]);
    const { target, calls } = recorder(clock);

    expect(await alignFramePhase(target, 60, 0.4, clock)).toBe(true);

    const phase = lastCallPhase(calls);
    expect(phase).toBeGreaterThanOrEqual(6_666_400n);
    expect(phase).toBeLessThan(6_666_400n + TOLERANCE_NS);
  });

  it('leaves the rate alone when timers never wake in time', async () => {
    const clock = new FakeClock(START, [], 5_000_000n);
    const { target, calls } = recorder(clock);

    expect(await alignFramePhase(target, 60, 0.4, clock)).toBe(false);

    expect(calls).toEqual([]);
  });
});

describe('MONOTONIC_CLOCK', () => {
  it('moves forward across a sleep', async () => {
    const before = MONOTONIC_CLOCK.nowNs();

    await MONOTONIC_CLOCK.sleep(2);

    expect(MONOTONIC_CLOCK.nowNs() - before).toBeGreaterThanOrEqual(1_000_000n);
  });
});

describe('workerFramePhase', () => {
  const worker = (index: number): Record<string, string> => ({
    DMA_BROWSER_STAGGER_FRAMES: '1',
    DMA_BROWSER_WORKER_INDEX: String(index),
    DMA_BROWSER_PROCESS_COUNT: '5',
  });

  it('is off unless DMA_BROWSER_STAGGER_FRAMES is set', () => {
    expect(
      workerFramePhase({ DMA_BROWSER_WORKER_INDEX: '2', DMA_BROWSER_PROCESS_COUNT: '5' }),
    ).toBeNull();
  });

  it('spreads the workers evenly over the frame period', () => {
    expect([0, 1, 2, 3, 4].map((i) => workerFramePhase(worker(i)))).toEqual([
      0, 0.2, 0.4, 0.6, 0.8,
    ]);
  });

  it('is off without workers to spread', () => {
    expect(workerFramePhase({ DMA_BROWSER_STAGGER_FRAMES: '1' })).toBeNull();
    expect(workerFramePhase({ ...worker(0), DMA_BROWSER_PROCESS_COUNT: '1' })).toBeNull();
  });
});
