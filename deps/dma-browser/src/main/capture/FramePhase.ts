import { envFlag, envInt, type Env } from '../support/env';

/**
 * Where in its frame period an offscreen window starts each frame.
 *
 * Chromium ticks an offscreen window on `timebase + n * interval` of CLOCK_MONOTONIC. Electron's
 * `setFrameRate` passes "now" as the timebase. `ui::Compositor` sends it to a display that already
 * exists, but hands it to a display created later (the first one, and each one after a GPU-process
 * restart) only once the interval has differed from the 60 Hz default (16,666 us). A 60 fps window
 * sets its rate before its display exists and never differs, so every such window of every process
 * keeps timebase zero and they all tick at the same instant: one burst of runnable threads per
 * frame. At other rates each window ticks from the moment its rate was set.
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
 * Sets `fps` on a 60 fps target when the clock is at `phase` (0 to 1) of the frame period, so the
 * window's frames start there; other rates are left alone. Returns false, leaving the target
 * alone, when timers wake too late to reach the instant.
 */
export async function alignFramePhase(
  target: FramePhaseTarget,
  fps: number,
  phase: number,
  clock: FramePhaseClock,
): Promise<boolean> {
  const period = frameIntervalNs(fps);
  if (period !== DEFAULT_INTERVAL_NS) return true; // already on its own timebase
  const offset = BigInt(Math.round(Number(period) * phase)) % period;
  const next = (now: bigint): bigint => now + ((offset - (now % period) + period) % period);

  let due = next(clock.nowNs());
  let sleeps = 0;
  for (;;) {
    const wait = due - clock.nowNs();
    if (wait < 0n) {
      due = next(clock.nowNs());
    } else if (wait <= SPIN_NS + TIMER_GRAIN_NS) {
      break;
    } else if (sleeps++ === MAX_SLEEPS) {
      return false;
    } else {
      await clock.sleep(Number((wait - SPIN_NS) / TIMER_GRAIN_NS));
    }
  }
  while (clock.nowNs() < due);

  // Another interval first: it makes Chromium keep the timebase for displays it creates later.
  target.setFrameRate(fps - 1);
  target.setFrameRate(fps);
  return true;
}

/**
 * The phase for this worker's 60 fps windows when DMA_BROWSER_STAGGER_FRAMES is on; null leaves
 * Chromium's timing. The index is read in reversed binary (0, 1/2, 1/4, 3/4, 1/8, ...): the
 * supervisor runs only as many workers as a show needs, and any first n of these stay at least
 * half an even spacing apart (exactly even for 2, 4, 8, 16). Windows of one worker share the
 * phase, so its compositor and GPU threads still serve them in one wake-up.
 */
export function workerFramePhase(env: Env): number | null {
  if (!envFlag(env, 'DMA_BROWSER_STAGGER_FRAMES', false)) return null;
  const index = envInt(env, 'DMA_BROWSER_WORKER_INDEX', -1, -1, Number.MAX_SAFE_INTEGER);
  if (index < 0) return null;
  let phase = 0;
  for (let rest = index, place = 0.5; rest > 0; rest = Math.floor(rest / 2), place /= 2) {
    if (rest % 2 === 1) phase += place;
  }
  return phase;
}
