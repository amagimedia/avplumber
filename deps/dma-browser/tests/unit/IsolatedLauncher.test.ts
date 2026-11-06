import { expect, it } from 'vitest';
import { chmodSync, copyFileSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';

it('selects sandboxed shell arguments and rejects extra isolated arguments', () => {
  const root = mkdtempSync(join(tmpdir(), 'isolated-launch-'));
  try {
    mkdirSync(join(root, 'bin'));
    mkdirSync(join(root, 'node_modules/.bin'), { recursive: true });
    copyFileSync('bin/run.sh', join(root, 'bin/run.sh'));
    const fake = join(root, 'node_modules/.bin/electron');
    writeFileSync(fake, '#!/bin/sh\nprintf "%s\\n" "$@"\n');
    chmodSync(fake, 0o700);
    const env = {
      PATH: process.env.PATH,
      DMA_BROWSER_DISABLE_NVIDIA_RUNTIME: '1',
      DMA_BROWSER_PROCESS_COUNT: '1',
    };
    const legacy = execFileSync('bash', [join(root, 'bin/run.sh')], { env, encoding: 'utf8' });
    expect(legacy).toContain('--no-sandbox');
    const isolated = { ...env, DMA_BROWSER_ISOLATED_SCENE_URL: 'http://127.0.0.1:43123/' };
    const strict = execFileSync('bash', [join(root, 'bin/run.sh')], {
      env: isolated,
      encoding: 'utf8',
    });
    expect(strict).not.toContain('--no-sandbox');
    expect(strict.trim()).toBe(join(root, 'dist/main/index.js'));
    expect(() =>
      execFileSync('bash', [join(root, 'bin/run.sh'), '--no-sandbox'], {
        env: isolated,
        stdio: 'pipe',
      }),
    ).toThrow();
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
