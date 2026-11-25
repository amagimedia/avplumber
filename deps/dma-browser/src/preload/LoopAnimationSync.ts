/**
 * Rephase infinite document-timeline animations without replacing any page clock or API.
 * Relative delays, iterationStart and alternating directions remain part of each effect.
 * Finite, paused, rate-controlled and scroll-timeline animations belong to the page.
 */
export function alignLoopAnimations(doc: Document, epochMs: number): void {
  const view = doc.defaultView;
  if (!view) return;
  // The document timeline uses this document's performance origin. Sampling wall time against
  // performance.now avoids treating the (possibly older) last animation-frame time as "now".
  const startTime = epochMs - (Date.now() - view.performance.now());
  for (const animation of doc.getAnimations()) {
    if (
      animation.timeline !== doc.timeline ||
      animation.playState !== 'running' ||
      animation.pending ||
      animation.playbackRate !== 1 ||
      !animation.effect
    )
      continue;
    const timing = animation.effect.getTiming();
    if (
      timing.iterations !== Infinity ||
      typeof timing.duration !== 'number' ||
      !Number.isFinite(timing.duration) ||
      timing.duration <= 0
    )
      continue;
    const previous = animation.startTime;
    if (typeof previous !== 'number' || Math.abs(previous - startTime) > 2) {
      animation.startTime = startTime;
    }
  }
  // Access is deliberately restricted to documents the browser lets us read. Cross-origin
  // frames need their own renderer integration; do not enable Node in subframes to reach them.
  for (const frame of Array.from(doc.querySelectorAll<HTMLIFrameElement>('iframe, frame'))) {
    try {
      const child = frame.contentDocument;
      if (child) alignLoopAnimations(child, epochMs);
    } catch {
      // An iframe can become inaccessible or navigate while this scan is running.
    }
  }
}

export class LoopAnimationSync {
  private timer: number | null = null;

  constructor(private readonly epochMs: number) {}

  public install(): void {
    this.start();
    window.addEventListener('pagehide', () => this.stop());
    window.addEventListener('pageshow', () => this.start());
  }

  private start(): void {
    if (this.timer !== null) return;
    const scan = (): void => alignLoopAnimations(document, this.epochMs);
    // Polling also finds WAAPI animations created without a DOM mutation, and animations
    // inside newly loaded same-origin frames. This is discovery/correction, not a frame pump.
    this.timer = window.setInterval(scan, 500);
    scan();
  }

  private stop(): void {
    if (this.timer !== null) window.clearInterval(this.timer);
    this.timer = null;
  }
}
