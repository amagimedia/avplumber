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

  it.each([25, 30, 50])(
    'leaves a %i fps window alone: Chromium already gives it its own phase',
    async (fps) => {
      const clock = new FakeClock(START);
      const { target, calls } = recorder(clock);

      expect(await alignFramePhase(target, fps, 0.2, clock)).toBe(true);

      expect(calls).toEqual([]);
    },
  );

  it('reaches a phase later in the current period without waiting for the next one', async () => {
    const clock = new FakeClock(START - (START % 16_666_000n) + 1_000_000n);
    const { target, calls } = recorder(clock);
    const begun = clock.t;

    await alignFramePhase(target, 60, 0.4, clock);

    expect(lastCallPhase(calls)).toBeGreaterThanOrEqual(6_666_400n);
    expect(lastCallPhase(calls)).toBeLessThan(6_666_400n + TOLERANCE_NS);
    expect(clock.t - begun).toBeLessThan(16_666_000n);
  });

  it('puts phase 0 on the start of a period', async () => {
    const clock = new FakeClock(START);
    const { target, calls } = recorder(clock);

    await alignFramePhase(target, 60, 0, clock);

    expect(lastCallPhase(calls)).toBeLessThan(TOLERANCE_NS);
  });

  it('gives every window of a worker its turn when twenty align at once', async () => {
    const clock = new FakeClock(START);
    const { target, calls } = recorder(clock);

    const aligned = await Promise.all(
      Array.from({ length: 20 }, () => alignFramePhase(target, 60, 0.4, clock)),
    );

    expect(aligned).toEqual(Array.from({ length: 20 }, () => true));
    expect(calls.filter((c) => c.fps === 60)).toHaveLength(20);
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
  });
  const phases = (count: number): number[] =>
    Array.from({ length: count }, (_, i) => workerFramePhase(worker(i)) ?? Number.NaN);

  it('is off unless DMA_BROWSER_STAGGER_FRAMES is set', () => {
    expect(workerFramePhase({ DMA_BROWSER_WORKER_INDEX: '2' })).toBeNull();
  });

  it('is off outside a worker process', () => {
    expect(workerFramePhase({ DMA_BROWSER_STAGGER_FRAMES: '1' })).toBeNull();
  });

  it('starts the first worker at the start of the period', () => {
    expect(workerFramePhase(worker(0))).toBe(0);
  });

  // The supervisor runs only as many workers as the show needs, so any first n must be spread.
  it.each([2, 3, 4, 5, 8, 16])('keeps the first %i workers apart', (count) => {
    const sorted = phases(count).sort((a, b) => a - b);
    const gaps = sorted.map((p, i) =>
      i === 0 ? p + 1 - (sorted[count - 1] ?? 0) : p - (sorted[i - 1] ?? 0),
    );

    expect(Math.min(...gaps) * count).toBeGreaterThan(0.6);
  });
});
