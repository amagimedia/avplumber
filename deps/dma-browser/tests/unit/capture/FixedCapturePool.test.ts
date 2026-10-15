import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { fixedCapturePoolSize } from '../../../src/main/capture/FixedCapturePool';

const directories: string[] = [];
function manifest(value?: unknown): string {
  const directory = mkdtempSync(join(tmpdir(), 'capture-pool-'));
  directories.push(directory);
  const path = join(directory, 'pool.json');
  if (value !== undefined) writeFileSync(path, JSON.stringify(value));
  return path;
}
afterEach(() => directories.splice(0).forEach((path) => rmSync(path, { recursive: true })));

describe('installed capture pool contract', () => {
  it('keeps stock Electron on the existing path', () => {
    expect(fixedCapturePoolSize(manifest())).toBe(0);
  });
  it('reports the full fixed pool installed with the patched binary', () => {
    expect(fixedCapturePoolSize(manifest({ version: 1, size: 11 }))).toBe(11);
  });
  it.each([6, 9, 11])('reports the selected startup pool %i', (size) => {
    expect(
      fixedCapturePoolSize(manifest({ version: 2, size: 11 }), {
        DMA_BROWSER_CHROMIUM_EXTRA_FEATURES: `AvpZeroCopyOffscreenExport:buffers/${size},RenderableMappableSharedImageForceScanout`,
      }),
    ).toBe(size);
  });
  it('requires the feature to be enabled for the selectable binary', () => {
    const path = manifest({ version: 2, size: 11 });
    expect(fixedCapturePoolSize(path, {})).toBe(0);
    expect(
      fixedCapturePoolSize(path, {
        DMA_BROWSER_CHROMIUM_EXTRA_FEATURES: 'AvpZeroCopyOffscreenExport',
      }),
    ).toBe(11);
  });
  it.each(['buffers/1', 'buffers/17', 'buffers/no', 'buffers/6.5', '<trial'])(
    'rejects ambiguous or out-of-range startup settings %s',
    (params) => {
      expect(() =>
        fixedCapturePoolSize(manifest({ version: 2, size: 11 }), {
          DMA_BROWSER_CHROMIUM_EXTRA_FEATURES: `AvpZeroCopyOffscreenExport:${params}`,
        }),
      ).toThrow('Fixed capture pool requires');
    },
  );
  it('rejects conflicting feature settings', () => {
    expect(() =>
      fixedCapturePoolSize(manifest({ version: 2, size: 11 }), {
        DMA_BROWSER_CHROMIUM_EXTRA_FEATURES: 'AvpZeroCopyOffscreenExport:buffers/6',
        DMA_BROWSER_CHROMIUM_DISABLE_FEATURES: 'AvpZeroCopyOffscreenExport',
      }),
    ).toThrow('both enabled and disabled');
    expect(() =>
      fixedCapturePoolSize(manifest({ version: 2, size: 11 }), {
        DMA_BROWSER_CHROMIUM_EXTRA_FEATURES:
          'AvpZeroCopyOffscreenExport:buffers/6,AvpZeroCopyOffscreenExport:buffers/9',
      }),
    ).toThrow('Duplicate');
  });
  it.each([null, { version: 1, size: 12 }, { version: 3, size: 11 }, { size: '11' }])(
    'rejects an unsupported contract %j',
    (value) => {
      expect(() => fixedCapturePoolSize(manifest(value))).toThrow('Unsupported fixed capture pool');
    },
  );
});
