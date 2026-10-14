/**
 * Captures Chromium numbered but never delivered, counted from the gaps in the paint
 * textures' `textureInfo.metadata.frameCount` (Electron's copy of the capturer's
 * `capture_counter`).
 *
 * Captures skipped before a number is assigned (for example, because too many frames
 * are outstanding) cannot be counted here.
 *
 * A number at or below the last one is a new capturer (a reload or a new window): it starts
 * a new baseline and is not a gap.
 */
export class CaptureCounterGaps {
  private last: number | null = null;
  private total = 0;

  public get skipped(): number {
    return this.total;
  }

  public observe(frameCount: unknown): void {
    if (typeof frameCount !== 'number' || !Number.isSafeInteger(frameCount) || frameCount < 0) {
      return;
    }
    if (this.last !== null && frameCount > this.last) this.total += frameCount - this.last - 1;
    this.last = frameCount;
  }
}

/** `textureInfo.metadata.frameCount` of a paint texture, if Electron sent one. */
export function paintFrameCount(textureInfo: Record<string, unknown> | undefined): unknown {
  const metadata = textureInfo?.metadata;
  return typeof metadata === 'object' && metadata !== null
    ? (metadata as Record<string, unknown>).frameCount
    : undefined;
}
