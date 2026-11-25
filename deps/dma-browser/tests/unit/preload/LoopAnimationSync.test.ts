import { afterEach, describe, expect, it, vi } from 'vitest';
import { alignLoopAnimations } from '../../../src/preload/LoopAnimationSync';

function documentFixture(age: number, children: { contentDocument: Document | null }[] = []) {
  const timeline = { currentTime: age - 10 };
  const animation = {
    timeline,
    playState: 'running',
    pending: false,
    playbackRate: 1,
    startTime: 0,
    effect: {
      getTiming: () => ({
        iterations: Infinity,
        duration: 1000,
        delay: -200,
        direction: 'alternate',
      }),
    },
  };
  const animations = [animation];
  const doc = {
    timeline,
    defaultView: { performance: { now: () => age } },
    getAnimations: () => animations,
    querySelectorAll: () => children,
  } as unknown as Document;
  return { doc, animation, animations };
}

afterEach(() => vi.restoreAllMocks());

describe('alignLoopAnimations', () => {
  it('gives pages loaded at different times the same loop position, including a rejoin', () => {
    vi.spyOn(Date, 'now').mockReturnValue(100_000);
    const a = documentFixture(20_000);
    const b = documentFixture(800);
    for (const f of [a, b]) {
      alignLoopAnimations(f.doc, 90_000);
      // Last animation-frame time is deliberately older than performance.now().
      expect(Number(f.doc.timeline.currentTime) - f.animation.startTime).toBe(9990);
      expect(f.animation.effect.getTiming().delay).toBe(-200);
      expect(f.animation.effect.getTiming().direction).toBe('alternate');
    }
    vi.mocked(Date.now).mockReturnValue(120_000);
    const rejoined = documentFixture(50);
    alignLoopAnimations(rejoined.doc, 90_000);
    expect(Number(rejoined.doc.timeline.currentTime) - rejoined.animation.startTime).toBe(29_990);
  });

  it('preserves finite, paused, pending, rate-controlled and custom-timeline animations', () => {
    vi.spyOn(Date, 'now').mockReturnValue(100_000);
    for (const overrides of [
      { playState: 'paused' },
      { playState: 'finished' },
      { pending: true },
      { playbackRate: 2 },
      { timeline: { currentTime: 42 } },
      { effect: { getTiming: () => ({ iterations: 1, duration: 1000 }) } },
      { effect: { getTiming: () => ({ iterations: Infinity, duration: 0 }) } },
    ]) {
      const f = documentFixture(800);
      Object.assign(f.animation, overrides);
      alignLoopAnimations(f.doc, 90_000);
      expect(f.animation.startTime).toBe(0);
    }
  });

  it('finds new animations and accessible iframe documents without failing on inaccessible frames', () => {
    vi.spyOn(Date, 'now').mockReturnValue(100_000);
    const child = documentFixture(300);
    const parent = documentFixture(800, [
      { contentDocument: child.doc },
      { contentDocument: null },
      {
        get contentDocument(): Document {
          throw new Error('cross-origin');
        },
      },
    ]);
    alignLoopAnimations(parent.doc, 90_000);
    expect(child.animation.startTime).toBe(-9700);
    const added = { ...parent.animation, startTime: 0 };
    parent.animations.push(added);
    alignLoopAnimations(parent.doc, 90_000);
    expect(added.startTime).toBe(-9200);
  });
});
