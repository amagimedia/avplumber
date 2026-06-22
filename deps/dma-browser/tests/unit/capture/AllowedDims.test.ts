import { describe, expect, it } from 'vitest';
import { AllowedDims } from '../../../src/main/capture/AllowedDims';

describe('AllowedDims', () => {
  it('allows everything when list is empty', () => {
    const d = AllowedDims.fromList([]);
    expect(d.allows(1, 1)).toBe(true);
    expect(d.allows(9999, 9999)).toBe(true);
    expect(d.listForLogging()).toBe('any');
  });

  it('parses WxH pairs', () => {
    const d = AllowedDims.fromList(['1920x1080', '1280x720']);
    expect(d.allows(1920, 1080)).toBe(true);
    expect(d.allows(1280, 720)).toBe(true);
    expect(d.allows(640, 480)).toBe(false);
    expect(d.listForLogging()).toBe('1920x1080,1280x720');
  });

  it('skips malformed entries and reverts to "any" when none parse', () => {
    expect(AllowedDims.fromList(['junk', '0x0']).listForLogging()).toBe('any');
  });

  it('mixes good and bad entries', () => {
    const d = AllowedDims.fromList(['junk', '1920x1080']);
    expect(d.allows(1920, 1080)).toBe(true);
    expect(d.allows(1, 1)).toBe(false);
  });
});
