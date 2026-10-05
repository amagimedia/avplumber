import { envFlag, envInt, type Env } from '../support/env';

/**
 * Where in its frame period an offscreen window starts each frame.
 *
 * Chromium ticks an offscreen window on `timebase + n * interval` of CLOCK_MONOTONIC. Electron's
 * `setFrameRate` passes "now" as the timebase, but `ui::Compositor` forwards a timebase only after
 * the interval has differed from its 60 Hz default (16,666 us). At 60 fps it never differs, so every
 * window of every process keeps timebase zero and they all tick at the same instant: one burst of
 * runnable threads per frame. At other rates each window already ticks from the moment its rate
 * was set.
 *
 * `alignFramePhase` sets the rate again at a chosen instant, which puts the tick there.
 */

/** Chromium's default interval, `base::Seconds(1) / 60` in whole microseconds. */
const DEFAULT_INTERVAL_NS = 16_666_000n;

/** Timers are millisecond-accurate at best; the last stretch to the instant is a busy wait. */
const SPIN_NS = 2_000_000n;
const TIMER_GRAIN_NS = 1_000_000n;
const MAX_SLEEPS = 20;

export interface FramePhaseTarget {
  setFrameRate(fps: number): void;
}

export interface FramePhaseClock {
  /** CLOCK_MONOTONIC, the clock Chromium's frame timer runs on. */
  nowNs(): bigint;
  sleep(ms: number): Promise<void>;
}

export const MONOTONIC_CLOCK: FramePhaseClock = {
  nowNs: () => process.hrtime.bigint(),
  sleep: (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
};

/** The interval Electron gives Chromium for a rate: one second over it in whole microseconds. */
export function frameIntervalNs(fps: number): bigint {
  return BigInt(Math.trunc(1_000_000 / fps)) * 1000n;
}

/**
 * Sets `fps` on the target when the clock is at `phase` (0 to 1) of the frame period, so the
 * window's frames start there. Returns false, leaving the target alone, when timers wake too late
 * to reach the instant.
 */
export async function alignFramePhase(
  target: FramePhaseTarget,
  fps: number,
  phase: number,
  clock: FramePhaseClock,
): Promise<boolean> {
  const period = frameIntervalNs(fps);
  const offset = BigInt(Math.round(Number(period) * phase)) % period;
  const next = (now: bigint): bigint => now + ((offset - (now % period) + period) % period);

  let due = next(clock.nowNs());
  for (let sleeps = 0; ; sleeps++) {
    const wait = due - clock.nowNs();
    if (wait < 0n) {
      due = next(clock.nowNs());
    } else if (wait <= SPIN_NS + TIMER_GRAIN_NS) {
      break;
    } else if (sleeps === MAX_SLEEPS) {
      return false;
    } else {
      await clock.sleep(Number((wait - SPIN_NS) / TIMER_GRAIN_NS));
    }
  }
  while (clock.nowNs() < due);

  // A rate with another interval first: without it Chromium drops the timebase at 60 fps.
  if (period === DEFAULT_INTERVAL_NS) target.setFrameRate(fps - 1);
  target.setFrameRate(fps);
  return true;
}

/**
 * The phase for this worker's windows when DMA_BROWSER_STAGGER_FRAMES is on: worker i of n starts
 * its frames i/n into the period. All windows of one worker share a phase, so its compositor and
 * GPU threads still serve them in one wake-up. Null leaves Chromium's own timing.
 */
export function workerFramePhase(env: Env): number | null {
  if (!envFlag(env, 'DMA_BROWSER_STAGGER_FRAMES', false)) return null;
  const count = envInt(env, 'DMA_BROWSER_PROCESS_COUNT', 1, 1, Number.MAX_SAFE_INTEGER);
  const index = envInt(env, 'DMA_BROWSER_WORKER_INDEX', -1, -1, Number.MAX_SAFE_INTEGER);
  if (count < 2 || index < 0) return null;
  return (index % count) / count;
}
