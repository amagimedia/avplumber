import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';

/** Shipped with the patched Electron distribution; absent for stock Electron. */
export function fixedCapturePoolSize(
  manifest = join(
    dirname(require.resolve('electron/package.json')),
    'dist/resources/avp-capture-pool.json',
  ),
): number {
  let contents: string;
  try {
    contents = readFileSync(manifest, 'utf8');
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return 0;
    throw error;
  }
  const value = JSON.parse(contents) as { version?: unknown; size?: unknown } | null;
  if (value?.version !== 1 || value.size !== 11) {
    throw new Error('Unsupported fixed capture pool manifest');
  }
  return value.size;
}
