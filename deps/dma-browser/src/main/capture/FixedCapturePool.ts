import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { envList, type Env } from '../support/env';

/** Shipped with the patched Electron distribution; absent for stock Electron. */
export function fixedCapturePoolSize(
  manifest = join(
    dirname(require.resolve('electron/package.json')),
    'dist/resources/avp-capture-pool.json',
  ),
  env: Env = process.env,
): number {
  let contents: string;
  try {
    contents = readFileSync(manifest, 'utf8');
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return 0;
    throw error;
  }
  const value = JSON.parse(contents) as { version?: unknown; size?: unknown } | null;
  if ((value?.version !== 1 && value?.version !== 2) || value.size !== 11) {
    throw new Error('Unsupported fixed capture pool manifest');
  }
  // Version 1 is the hardcoded-11 binary. Version 2 certifies the optional
  // Chromium startup-size patch; reading a flag alone cannot certify it.
  if (value.version === 1) return value.size;
  const feature = 'AvpZeroCopyOffscreenExport';
  const selected = envList(env, 'DMA_BROWSER_CHROMIUM_EXTRA_FEATURES').filter(
    (entry) => entry.split(/[:<]/, 1)[0] === feature,
  );
  if (!selected.length) return 0;
  if (envList(env, 'DMA_BROWSER_CHROMIUM_DISABLE_FEATURES').includes(feature)) {
    throw new Error('Fixed capture pool feature is both enabled and disabled');
  }
  if (selected.length !== 1) throw new Error('Duplicate fixed capture pool feature');
  if (selected[0] === feature) return value.size;
  const match = /^AvpZeroCopyOffscreenExport:buffers\/(\d+)$/.exec(selected[0]!);
  const size = match ? Number(match[1]) : NaN;
  if (!Number.isInteger(size) || size < 2 || size > 16) {
    throw new Error('Fixed capture pool requires buffers/N with N from 2 to 16');
  }
  return size;
}
