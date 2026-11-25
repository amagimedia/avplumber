/** Both replicas must use the same settings and synchronized host wall clocks. */
export interface BrowserSyncConfig {
  readonly epochMs: number;
  /** Fraction of Electron's frame period, independent of worker assignment. */
  readonly phase: number;
  /** Opt in separately: rephases infinite CSS/WAAPI animations in accessible documents. */
  readonly loopAnimations: boolean;
}

export interface BrowserSyncStatus {
  readonly config: BrowserSyncConfig;
  readonly applications: number;
  readonly failures: number;
  readonly lastAppliedUnixMs: number | null;
  /** Wall-clock change against monotonic since the last successful phase application. */
  readonly clockDriftMs: number;
}
