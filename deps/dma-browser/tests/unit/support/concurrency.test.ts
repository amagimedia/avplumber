import { describe, expect, it } from 'vitest';
import { forEachBounded, settleAll } from '../../../src/main/support/concurrency';

const tick = async (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 5));

describe('settleAll', () => {
  it('rethrows the first failure only after every task settled', async () => {
    let finished = false;
    const slow = (async () => {
      await tick();
      finished = true;
    })();
    await expect(settleAll([Promise.reject(new Error('first')), slow])).rejects.toThrow('first');
    expect(finished).toBe(true);
  });
});

describe('forEachBounded', () => {
  it('keeps at most the limit in flight and visits every item once', async () => {
    let inFlight = 0;
    let peak = 0;
    const seen: number[] = [];
    await forEachBounded([0, 1, 2, 3, 4, 5, 6], 3, async (item) => {
      inFlight += 1;
      peak = Math.max(peak, inFlight);
      await tick();
      seen.push(item);
      inFlight -= 1;
    });
    expect(peak).toBe(3);
    expect(seen.sort()).toEqual([0, 1, 2, 3, 4, 5, 6]);
  });

  it('lets the other lanes drain the queue when one task fails', async () => {
    const seen: number[] = [];
    const run = forEachBounded([0, 1, 2, 3], 2, async (item) => {
      await tick();
      if (item === 0) throw new Error('page 0');
      seen.push(item);
    });
    await expect(run).rejects.toThrow('page 0');
    expect(seen.sort()).toEqual([1, 2, 3]);
  });
});
