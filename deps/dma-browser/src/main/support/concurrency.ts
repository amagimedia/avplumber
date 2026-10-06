/**
 * Waits for every task, then rethrows the first failure. Unlike Promise.all, nothing is still
 * running when the caller sees the error, so a retry cannot overlap the tasks it retries.
 */
export async function settleAll(tasks: readonly Promise<unknown>[]): Promise<void> {
  for (const result of await Promise.allSettled(tasks)) {
    if (result.status === 'rejected') throw result.reason;
  }
}

/** Runs `task` over `items` with at most `limit` in flight, settling like settleAll. */
export async function forEachBounded<T>(
  items: readonly T[],
  limit: number,
  task: (item: T) => Promise<void>,
): Promise<void> {
  // The lanes share one iterator: each takes the next item when its previous task settles.
  const pending = items[Symbol.iterator]();
  const lane = async (): Promise<void> => {
    for (let next = pending.next(); next.done !== true; next = pending.next()) {
      await task(next.value);
    }
  };
  await settleAll(Array.from({ length: Math.min(limit, items.length) }, lane));
}
