import { describe, expect, it } from 'vitest';
import { CaptureCounterGaps, paintFrameCount } from '../../../src/main/capture/CaptureCounterGaps';

describe('CaptureCounterGaps', () => {
  it('counts the numbers Chromium took but never delivered', () => {
    const gaps = new CaptureCounterGaps();
    for (const n of [10, 11, 12, 15, 16, 20]) gaps.observe(n);
    expect(gaps.skipped).toBe(5); // 13, 14, 17, 18, 19
  });

  it('starts again on a new capturer and ignores paints without a number', () => {
    const gaps = new CaptureCounterGaps();
    for (const n of [100, 102]) gaps.observe(n);
    gaps.observe(undefined);
    gaps.observe('7');
    gaps.observe(-1);
    gaps.observe(1.5);
    expect(gaps.skipped).toBe(1);
    // A reload: the counter starts low again; its first frame is a baseline, not a gap.
    for (const n of [0, 1, 3]) gaps.observe(n);
    expect(gaps.skipped).toBe(2);
    gaps.observe(3); // the same number again is not a gap either
    expect(gaps.skipped).toBe(2);
  });

  it('reads metadata.frameCount from a paint texture', () => {
    expect(paintFrameCount({ metadata: { frameCount: 42 } })).toBe(42);
    expect(paintFrameCount({ metadata: null })).toBeUndefined();
    expect(paintFrameCount(undefined)).toBeUndefined();
  });
});
