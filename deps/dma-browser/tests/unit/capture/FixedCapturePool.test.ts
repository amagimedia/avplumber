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
  it.each([null, { version: 1, size: 12 }, { version: 2, size: 11 }, { size: '11' }])(
    'rejects an unsupported contract %j',
    (value) => {
      expect(() => fixedCapturePoolSize(manifest(value))).toThrow('Unsupported fixed capture pool');
    },
  );
});
