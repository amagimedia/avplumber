import type { BrowserSyncConfig, BrowserSyncStatus } from '../../shared/browser-sync';
import { alignFrameGrid, frameIntervalNs, MONOTONIC_CLOCK } from './FramePhase';
import type { FramePhaseClock, FramePhaseTarget } from './FramePhase';

export interface BrowserSyncClock extends FramePhaseClock {
  unixMs(): number;
}

const SYSTEM_CLOCK: BrowserSyncClock = { ...MONOTONIC_CLOCK, unixMs: () => Date.now() };

/** Requests a common rendering phase; this does not measure actual paint or content agreement. */
export class BrowserFrameSync {
  private timer: ReturnType<typeof setInterval> | null = null;
  private stopped = true;
  private running = false;
  private requested = false;
  private generation = 0;
  private appliedOffsetNs: bigint | null = null;
  private applications = 0;
  private failures = 0;
  private lastAppliedUnixMs: number | null = null;
  private clockDriftMs = 0;

  constructor(
    private readonly target: FramePhaseTarget,
    private readonly fps: number,
    private readonly config: BrowserSyncConfig,
    private readonly report: (message: string) => void,
    private readonly clock: BrowserSyncClock = SYSTEM_CLOCK,
  ) {}

  public start(): void {
    if (!this.stopped) return;
    this.stopped = false;
    this.realign();
    this.timer = setInterval(() => this.checkClock(), 5000);
  }

  public stop(): void {
    this.generation++;
    this.stopped = true;
    this.requested = false;
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }

  /** Navigation or a recreated renderer can replace Chromium's compositor clock. */
  public realign(): void {
    if (this.stopped) return;
    this.requested = true;
    if (this.running) return;
    this.apply().catch((err: unknown) => {
      this.failures++;
      this.report(`browser frame sync failed: ${String(err)}`);
    });
  }

  public snapshot(): BrowserSyncStatus {
    return {
      config: this.config,
      applications: this.applications,
      failures: this.failures,
      lastAppliedUnixMs: this.lastAppliedUnixMs,
      clockDriftMs: this.clockDriftMs,
    };
  }

  private offsetNs(): bigint {
    const before = this.clock.nowNs();
    const unixNs = BigInt(this.clock.unixMs()) * 1_000_000n;
    const after = this.clock.nowNs();
    return unixNs - (before + after) / 2n;
  }

  private checkClock(): void {
    if (this.stopped) return;
    if (this.appliedOffsetNs === null) {
      this.realign();
      return;
    }
    const drift = this.offsetNs() - this.appliedOffsetNs;
    this.clockDriftMs = Number(drift) / 1e6;
    // Reapplying a rate can perturb cadence. Avoid doing so for Date.now's millisecond noise.
    if (Math.abs(this.clockDriftMs) > Math.max(2, Number(frameIntervalNs(this.fps)) / 4e6)) {
      this.realign();
    }
  }

  private async apply(): Promise<void> {
    this.running = true;
    try {
      while (this.requested && !this.stopped) {
        this.requested = false;
        // Navigation replaces the compositor phase. A failed attempt must be retried even
        // if an earlier document was aligned and the host clock has not moved since then.
        this.appliedOffsetNs = null;
        const offset = this.offsetNs();
        const epoch = BigInt(this.config.epochMs) * 1_000_000n - offset;
        const generation = this.generation;
        const applied = await alignFrameGrid(
          this.target,
          this.fps,
          this.config.phase,
          epoch,
          this.clock,
          () => generation !== this.generation,
        );
        if (generation !== this.generation) return;
        if (applied) {
          this.appliedOffsetNs = offset;
          this.clockDriftMs = 0;
          this.applications++;
          this.lastAppliedUnixMs = this.clock.unixMs();
        } else {
          this.failures++;
          this.report('browser frame sync deferred: timers woke too late');
        }
      }
    } finally {
      this.running = false;
    }
  }
}
